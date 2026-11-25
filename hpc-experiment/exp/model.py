"""The repo's transformer (scaling-generator/transformer.py), re-implemented with fused projections and
scaled_dot_product_attention so that it trains faster, with the SAME maths and the SAME checkpoint layout on save.

Same as the repo:
  * pre-LN blocks, n_embed=402, 6 heads, ReLU feed-forward with 4x width, final LayerNorm, no dropout;
  * token + learned position embeddings, default PyTorch initialisation (nn.Embedding: N(0,1));
  * attention logits are scaled by n_embed**-0.5 (the repo takes C from x.shape, so it is NOT head_size**-0.5);
  * mask: the `input_len` word positions attend bidirectionally to each other and to nothing else; START and the
    output positions attend to every word position and causally to earlier output positions;
  * logits are returned for the last `n_out` positions only.
`original_state_dict()` converts to the repo's key names (per-head query/key/value, `tril` buffers), so the saved
safetensors file can be read by the repo's code and by claude-analysis/np_transformer.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H, self.hd = H, D // H
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D)
        self.ff1, self.ff2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        # SDPA divides by sqrt(head_size); the repo divides by sqrt(n_embed).  Pre-scale q to turn one into the other.
        self.q_scale = math.sqrt(self.hd / D)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.H, self.hd).transpose(1, 2) * self.q_scale
        k = k.view(B, T, self.H, self.hd).transpose(1, 2)
        v = v.view(B, T, self.H, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(F.relu(self.ff1(self.ln2(x))))


class Net(nn.Module):
    def __init__(self, vocab, n_ctx, input_len, n_out, D=402, H=6, n_blocks=5):
        super().__init__()
        assert D % H == 0
        self.vocab, self.n_ctx, self.input_len, self.n_out = vocab, n_ctx, input_len, n_out
        self.D, self.H, self.n_blocks = D, H, n_blocks
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(n_ctx, D)
        self.blocks = nn.ModuleList([Block(D, H) for _ in range(n_blocks)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        allowed = torch.zeros(n_ctx, n_ctx, dtype=torch.bool)
        allowed[:input_len, :input_len] = True
        allowed[input_len:, :input_len] = True
        allowed[input_len:, input_len:] = torch.tril(torch.ones(n_ctx - input_len, n_ctx - input_len)).bool()
        self.register_buffer("mask", allowed, persistent=False)

    def forward(self, idx, return_hidden=False):
        B, T = idx.shape
        assert T == self.n_ctx
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        hidden = [x] if return_hidden else None
        for b in self.blocks:
            x = b(x, self.mask)
            if return_hidden:
                hidden.append(x)
        logits = self.head(self.lnf(x)[:, -self.n_out:])
        return (logits, hidden) if return_hidden else logits

    # ------------------------------------------------------------------ checkpoint layout of the repo
    def original_state_dict(self):
        D, H, hd = self.D, self.H, self.D // self.H
        sd = {"token_embedding_table.weight": self.tok.weight, "position_embedding.weight": self.pos.weight}
        tril = self.mask.float()
        for i, b in enumerate(self.blocks):
            p = f"blocks.{i}."
            q, k, v = b.qkv.weight.chunk(3, dim=0)
            for h in range(H):
                sl = slice(h * hd, (h + 1) * hd)
                sd[p + f"sa.heads.{h}.query.weight"] = q[sl]
                sd[p + f"sa.heads.{h}.key.weight"] = k[sl]
                sd[p + f"sa.heads.{h}.value.weight"] = v[sl]
                sd[p + f"sa.heads.{h}.tril"] = tril
            sd[p + "sa.proj.weight"], sd[p + "sa.proj.bias"] = b.proj.weight, b.proj.bias
            sd[p + "ln1.weight"], sd[p + "ln1.bias"] = b.ln1.weight, b.ln1.bias
            sd[p + "ln2.weight"], sd[p + "ln2.bias"] = b.ln2.weight, b.ln2.bias
            sd[p + "ffwd.net.0.weight"], sd[p + "ffwd.net.0.bias"] = b.ff1.weight, b.ff1.bias
            sd[p + "ffwd.net.2.weight"], sd[p + "ffwd.net.2.bias"] = b.ff2.weight, b.ff2.bias
        sd[f"blocks.{self.n_blocks}.weight"], sd[f"blocks.{self.n_blocks}.bias"] = self.lnf.weight, self.lnf.bias
        sd["lm_head.weight"] = self.head.weight
        return {k: v.detach() for k, v in sd.items()}

    def load_original_state_dict(self, sd):
        H, hd = self.H, self.D // self.H
        with torch.no_grad():
            self.tok.weight.copy_(sd["token_embedding_table.weight"]); self.pos.weight.copy_(sd["position_embedding.weight"])
            for i, b in enumerate(self.blocks):
                p = f"blocks.{i}."
                for j, nm in enumerate(("query", "key", "value")):
                    w = torch.cat([sd[p + f"sa.heads.{h}.{nm}.weight"] for h in range(H)], 0)
                    b.qkv.weight[j * self.D:(j + 1) * self.D].copy_(w)
                b.proj.weight.copy_(sd[p + "sa.proj.weight"]); b.proj.bias.copy_(sd[p + "sa.proj.bias"])
                b.ln1.weight.copy_(sd[p + "ln1.weight"]); b.ln1.bias.copy_(sd[p + "ln1.bias"])
                b.ln2.weight.copy_(sd[p + "ln2.weight"]); b.ln2.bias.copy_(sd[p + "ln2.bias"])
                b.ff1.weight.copy_(sd[p + "ffwd.net.0.weight"]); b.ff1.bias.copy_(sd[p + "ffwd.net.0.bias"])
                b.ff2.weight.copy_(sd[p + "ffwd.net.2.weight"]); b.ff2.bias.copy_(sd[p + "ffwd.net.2.bias"])
            self.lnf.weight.copy_(sd[f"blocks.{self.n_blocks}.weight"]); self.lnf.bias.copy_(sd[f"blocks.{self.n_blocks}.bias"])
            self.head.weight.copy_(sd["lm_head.weight"])


def save_original_safetensors(model, path, dtype=torch.bfloat16):
    """Write the repo-format checkpoint (bfloat16, like the repo's own accelerate save)."""
    from safetensors.torch import save_file
    sd = {k: v.detach().to(dtype).clone().contiguous().cpu() for k, v in model.original_state_dict().items()}
    save_file(sd, path)
