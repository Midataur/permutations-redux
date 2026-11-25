"""
NumPy re-implementation of scaling-generator/transformer.py (inference only).

Mirrors the repo exactly, including two quirks worth knowing about:
  * attention logits are scaled by n_embed**-0.5 (C is taken from x.shape, i.e. 402),
    NOT head_size**-0.5;
  * masking: the INPUT_LENGTH input tokens attend bidirectionally to each other and never
    to later positions; the START token and the output tokens attend to every input token
    and causally to earlier output tokens.

Weights are loaded from model.safetensors (stored bfloat16 -> upcast to float32, which is
what load_checkpoint_and_dispatch into a float32 model does).
"""
import numpy as np
from st_loader import load_safetensors


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)   # biased, like torch.nn.LayerNorm
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class NPTransformer:
    def __init__(self, path, max_group=25, input_len=25, n_head=6, n_blocks=5, num_trans=None):
        W, _ = load_safetensors(path)
        self.W = W
        self.n_head, self.n_blocks = n_head, n_blocks
        self.max_group, self.input_len = max_group, input_len
        # general transpositions: n^2 tokens; elementary: pass num_trans=max_group (e.g. NPTransformer(path, 16, 120, num_trans=16))
        self.num_trans = max_group ** 2 if num_trans is None else num_trans
        self.num_normal = self.num_trans + max_group
        self.START = self.num_normal + 1
        self.E = W["token_embedding_table.weight"]
        self.P = W["position_embedding.weight"]
        self.D = self.E.shape[1]
        self.hd = self.D // n_head
        self.scale = self.D ** -0.5               # NB: repo uses C=x.shape[-1]=n_embed
        self.layers = []
        for i in range(n_blocks):
            p = f"blocks.{i}."
            cat = lambda nm: np.concatenate(
                [W[p + f"sa.heads.{h}.{nm}.weight"] for h in range(n_head)], axis=0)
            self.layers.append(dict(
                ln1w=W[p + "ln1.weight"], ln1b=W[p + "ln1.bias"],
                ln2w=W[p + "ln2.weight"], ln2b=W[p + "ln2.bias"],
                Wq=cat("query"), Wk=cat("key"), Wv=cat("value"),
                Wo=W[p + "sa.proj.weight"], bo=W[p + "sa.proj.bias"],
                W1=W[p + "ffwd.net.0.weight"], b1=W[p + "ffwd.net.0.bias"],
                W2=W[p + "ffwd.net.2.weight"], b2=W[p + "ffwd.net.2.bias"],
            ))
        self.lnfw = W[f"blocks.{n_blocks}.weight"]
        self.lnfb = W[f"blocks.{n_blocks}.bias"]
        self.Wlm = W["lm_head.weight"]
        # sanity-check our mask against the one stored in the checkpoint
        stored = W["blocks.0.sa.heads.0.tril"]
        assert np.array_equal(stored, self.mask(stored.shape[0])), "mask mismatch with checkpoint"

    # ------------------------------------------------------------------ masks
    def mask(self, T):
        L = self.input_len
        M = np.zeros((T, T), dtype=np.float32)
        M[:L, :L] = 1
        if T > L:
            M[L:, :L] = 1
            M[L:, L:] = np.tril(np.ones((T - L, T - L), dtype=np.float32))
        return M

    # ------------------------------------------------------------- full forward
    def forward(self, tokens, return_hidden=False, return_attn=False):
        """tokens: (B,T) ints. Returns logits for the last max_group positions (B,max_group,V)."""
        B, T = tokens.shape
        x = self.E[tokens] + self.P[:T][None]
        M = self.mask(T)
        hidden, attns = [x.copy()], []
        for ly in self.layers:
            a = layer_norm(x, ly["ln1w"], ly["ln1b"])
            q = (a @ ly["Wq"].T).reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)
            k = (a @ ly["Wk"].T).reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)
            v = (a @ ly["Wv"].T).reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)
            s = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            s = np.where(M[None, None] == 0, -np.inf, s)
            w = softmax(s)
            if return_attn:
                attns.append(w)
            o = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, self.D)
            x = x + (o @ ly["Wo"].T + ly["bo"])
            f = layer_norm(x, ly["ln2w"], ly["ln2b"])
            x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
            hidden.append(x.copy())
        xf = layer_norm(x, self.lnfw, self.lnfb)
        logits = xf[:, -self.max_group:, :] @ self.Wlm.T
        out = [logits]
        if return_hidden:
            out.append(hidden)
        if return_attn:
            out.append(attns)
        return out[0] if len(out) == 1 else tuple(out)

    # ----------------------------------------- incremental (KV-cache) generation
    def _encode_inputs(self, seq):
        """Run the 25 input tokens + START through the stack once; return per-layer K,V caches."""
        B = seq.shape[0]
        L = self.input_len
        toks = np.concatenate([seq, np.full((B, 1), self.START, dtype=seq.dtype)], axis=1)
        T = L + 1
        x = self.E[toks] + self.P[:T][None]
        M = self.mask(T)
        caches = []
        for ly in self.layers:
            a = layer_norm(x, ly["ln1w"], ly["ln1b"])
            q = (a @ ly["Wq"].T).reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)
            k = (a @ ly["Wk"].T).reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)
            v = (a @ ly["Wv"].T).reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)
            s = np.where(M[None, None] == 0, -np.inf, (q @ k.transpose(0, 1, 3, 2)) * self.scale)
            o = (softmax(s) @ v).transpose(0, 2, 1, 3).reshape(B, T, self.D)
            x = x + (o @ ly["Wo"].T + ly["bo"])
            f = layer_norm(x, ly["ln2w"], ly["ln2b"])
            x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
            caches.append([k, v])
        xf = layer_norm(x[:, -1], self.lnfw, self.lnfb)      # START position predicts p_0
        return caches, xf @ self.Wlm.T

    def _step(self, tok, pos, caches):
        """Append one output token at position `pos`; return logits for the next output."""
        B = tok.shape[0]
        x = self.E[tok] + self.P[pos][None]
        x = x[:, None, :]
        for ly, c in zip(self.layers, caches):
            a = layer_norm(x, ly["ln1w"], ly["ln1b"])
            q = (a @ ly["Wq"].T).reshape(B, 1, self.n_head, self.hd).transpose(0, 2, 1, 3)
            k = (a @ ly["Wk"].T).reshape(B, 1, self.n_head, self.hd).transpose(0, 2, 1, 3)
            v = (a @ ly["Wv"].T).reshape(B, 1, self.n_head, self.hd).transpose(0, 2, 1, 3)
            c[0] = np.concatenate([c[0], k], axis=2)
            c[1] = np.concatenate([c[1], v], axis=2)
            s = (q @ c[0].transpose(0, 1, 3, 2)) * self.scale     # attends to everything so far
            o = (softmax(s) @ c[1]).transpose(0, 2, 1, 3).reshape(B, 1, self.D)
            x = x + (o @ ly["Wo"].T + ly["bo"])
            f = layer_norm(x, ly["ln2w"], ly["ln2b"])
            x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
        xf = layer_norm(x[:, 0], self.lnfw, self.lnfb)
        return xf @ self.Wlm.T

    def generate(self, seq, force_valid=True, return_logits=False):
        """Greedy autoregressive decoding, identical in logic to Transformer.generate.
        seq: (B,input_len) token ids. Returns permutations (B,max_group) as 0-based entries."""
        seq = np.asarray(seq)
        B = seq.shape[0]
        caches, logits = self._encode_inputs(seq)
        used = np.zeros((B, self.max_group), dtype=bool)
        out, all_logits = np.zeros((B, self.max_group), dtype=np.int64), []
        for i in range(self.max_group):
            all_logits.append(logits)
            if force_valid:
                perm_logits = logits[:, self.num_trans:self.num_normal].copy()
                perm_logits[used] = -np.inf
                chosen = perm_logits.argmax(1)                 # permutation index 0..n-1
            else:
                chosen = logits.argmax(1) - self.num_trans
            out[:, i] = chosen
            used[np.arange(B), np.clip(chosen, 0, self.max_group - 1)] = True
            if i < self.max_group - 1:
                logits = self._step((chosen + self.num_trans).astype(np.int64),
                                    self.input_len + 1 + i, caches)
        return (out, np.stack(all_logits, 1)) if return_logits else out

    def teacher_forced_predictions(self, seq, perm):
        """Argmax prediction (over the whole vocab) at each output position given the TRUE prefix.
        Returns predicted permutation entries (may be out of range if the model predicts a
        non-permutation token) of shape (B,max_group)."""
        seq, perm = np.asarray(seq), np.asarray(perm)
        B = seq.shape[0]
        start = np.full((B, 1), self.START, dtype=seq.dtype)
        toks = np.concatenate([seq, start, perm[:, :-1] + self.num_trans], axis=1)
        logits = self.forward(toks)
        return logits.argmax(-1) - self.num_trans, logits
