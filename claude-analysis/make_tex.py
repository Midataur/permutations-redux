#!/usr/bin/env python3
"""Convert report.md into report.tex (XeLaTeX).

    python make_tex.py [report.md [report.tex]]
    xelatex report.tex && xelatex report.tex        # two passes for the contents and the references

The Markdown file stays the master copy: every number and sentence in the .tex comes from it, so a new
version of the report is regenerated, not retyped.  What this script decides is layout only:
  * section references such as "§3.6-3.8" become hyperlinked \\ref's;
  * [NEW] / [UPDATED] become coloured tags (also in headings and in the table of contents);
  * leading bold run-in headings become \\lead{...}; "What it shows" and "Limits"/"Caveats" become shaded boxes;
  * tables get booktabs rules and per-table column specifications (TABLE_SPECS below);
  * two tables (patching) are re-arranged so the five layer values sit in five columns.
"""
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "report.md"
DST = sys.argv[2] if len(sys.argv) > 2 else "report.tex"

# ----------------------------------------------------------------------------------------------------------
# Edit these when a new version of the report is made.
VERSION = "Version 8"
SUBTITLE = "Results of the Spartan training run"
DATE = "27 November 2026"
EXPECTED_TITLE = "Tests on two trained models, and a training experiment on four more: `torn-bigger-13` (general transpositions, $S_{25}$) and `elem-partitioned-long-boosted-1` (elementary transpositions)"
FIGURE_CAPTION_FOR_FILE_LINE = {
    # the line "**Figure:** `fig_accuracy_vs_hops.png` (...)" in section 3.3 stands for the picture itself
    "fig_accuracy_vs_hops.png": ("Planted-chase accuracy against hop count $L$ (top), and the number of training slots with at "
                                 "least $L$ hops, from one real training file scaled to 8M words (bottom). Teacher-forced, "
                                 "400 words per point, 95\\% Wilson intervals."),
}
FIGURE_WIDTH = {"fig_accuracy_vs_hops.png": 0.88, "fig_probe_layers.png": 1.0, "fig_influence_map.png": 1.0,
                "fig_elem_cells.png": 1.0, "fig_input_positions.png": 1.0, "fig_patching_setup.png": 1.0, "fig_input_positions_elem.png": 1.0,
                "fig_attention_heads.png": 1.0, "fig_attention_profiles.png": 1.0,
                "fig_train_learning_curves.png": 1.0, "fig_train_hop_cliff.png": 1.0, "fig_ebase_blocks.png": 1.0}

ARR = r">{\raggedright\arraybackslash}"
RARR = r">{\raggedleft\arraybackslash}"
TABLE_SPECS = {
    # index in the order of appearance: column spec, optional long=True (longtable), optional size, optional special
    0: dict(cols="l r r"),
    1: dict(cols="l r r r l r", long=True),
    2: dict(cols="r r l r", long=True),
    3: dict(cols="l r r r r r"),
    4: dict(cols=ARR + "p{6.6cm} r r"),
    5: dict(cols="l *{8}{r}"),
    6: dict(cols="l c r r r r"),
    7: dict(cols="l r r"),
    8: dict(cols="l *{5}{r}"),
    9: dict(cols="l *{6}{r}"),
    10: dict(cols="l *{4}{r}"),
    11: dict(cols="l r r"),
    12: dict(cols=ARR + "p{6.0cm} r r r r"),
    13: dict(cols="l *{8}{r}"),
    14: dict(cols="r l r r *{5}{r}", special="patch_all"),
    15: dict(cols="l " + ARR + "p{5.4cm} *{5}{r}", special="patch_subsets", long=True),
    # section 3.11 (elementary model): probes on natural words, planted long ranges, patching all positions, patching subsets
    16: dict(cols="l *{9}{r}"),
    17: dict(cols="l l *{6}{r}"),
    18: dict(cols="r l r r *{5}{r}", special="patch_all"),
    19: dict(cols="l " + ARR + "p{5.4cm} *{5}{r}", special="patch_subsets", long=True),
    # section 3.12 (attention and head ablation): hop offsets of the elementary model; ablation tables for the general and the elementary model
    20: dict(cols="r r " + ARR + "p{4.6cm} l"),
    21: dict(cols="r r l " + ARR + "p{4.0cm} l r", long=True, size="footnotesize"),
    22: dict(cols="r r l " + ARR + "p{3.8cm} l r", long=True, size="footnotesize"),
    # sections 3.13-3.16 (version 8)
    23: dict(cols="r " + RARR + "p{6.4cm} r r"),                                     # Test 13: hops, training slots, accuracy at widths 8 and 4
    24: dict(cols="l r r r"),                                     # Test 14: heads zeroed together
    25: dict(cols="l r " + ARR + "p{1.9cm} r " + ARR + "p{2.2cm} " + ARR + "p{2.1cm} " + ARR + "p{2.1cm}", size="small"),   # learning milestones
    26: dict(cols="l " + ARR + "p{2.6cm} " + ARR + "p{2.8cm} " + ARR + "p{4.2cm}", size="small"),                          # final accuracy on natural words
    27: dict(cols="r r r r r"),                                   # planted-chain accuracy of the three general arms
    28: dict(cols="r " + ARR + "p{3.3cm} " + ARR + "p{4.2cm} " + ARR + "p{4.2cm}", size="small"),                       # tail of the hop-count distribution
    29: dict(cols=ARR + "p{6.0cm} " + ARR + "p{8.6cm}", size="small"),                                                    # predictions against outcomes
    30: dict(cols=ARR + "p{5.0cm} " + ARR + "p{4.0cm} " + ARR + "p{4.6cm}", size="small"),             # e_base against the original: neighbour cosine
    31: dict(cols="l *{8}{r}", size="small"),                     # probes of the retrained models
    32: dict(cols=ARR + "p{2.9cm} " + ARR + "p{3.2cm} " + ARR + "p{8.0cm}", size="small"),                                # single-head ablation, four arms
    33: dict(cols=ARR + "p{2.6cm} r r " + ARR + "p{9.0cm}", size="small"),                                                # greedy ablation
    34: dict(cols="r r r r r"),                                   # patching, first hop corrupted, four models
    # section 6.1 and section 7
    35: dict(cols=ARR + "p{2.9cm} " + ARR + "p{5.8cm} " + ARR + "p{5.2cm}", size="small"),
    36: dict(cols=ARR + "p{5.6cm} " + ARR + "p{8.6cm}", long=True, size="footnotesize"),
}

CALLOUTS = {"What it shows": "find", "What this does and does not say": "find",
            "Limits": "lim", "Caveats": "lim", "Limitations of the design": "lim"}

# ----------------------------------------------------------------------------------------------------------
# inline conversion


def esc_code(c):
    out = []
    for i, ch in enumerate(c):
        nxt = c[i + 1] if i + 1 < len(c) else ""
        brk = r"\allowbreak{}" if nxt else ""      # where a long file name may be broken
        if ch == "_":
            out.append(r"\_" + brk)
        elif ch == "/":
            out.append("/" + brk)
        elif ch in "-.=:":
            out.append(ch + (brk if nxt.isalnum() else ""))
        elif ch in "#%&${}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def plain(s):
    """text for the PDF bookmarks"""
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("\\", ""), s)
    return s.replace("**", "").replace("*", "")


SCAN = re.compile(r"\[NEW\]|\[UPDATED\]|§(\d+(?:\.\d+)?)(?:–(\d+(?:\.\d+)?))?|Section (\d+\.\d+)|[\"'%&#_→×±−]|.", re.S)
OPENERS = " \n([{—–/"


def text(t, prev):
    out = []
    for m in SCAN.finditer(t):
        g = m.group(0)
        if g == "[NEW]":
            out.append(r"\tagnew{}")
            continue
        if g == "[UPDATED]":
            out.append(r"\tagupd{}")
            continue
        if g.startswith("§"):
            s = r"\S\,\ref{sec:%s}" % m.group(1)
            if m.group(2):
                s += r"–\ref{sec:%s}" % m.group(2)
            out.append(s)
            prev = "0"
            continue
        if g.startswith("Section "):
            out.append(r"Section~\ref{sec:%s}" % m.group(3))
            prev = "0"
            continue
        if g == '"':
            out.append("“" if (prev == "" or prev in OPENERS) else "”")
        elif g == "'":
            out.append("’")
        elif g in "%&#_":
            out.append("\\" + g)
        elif g == "→":
            out.append(r"\ensuremath{\to}")
        elif g == "×":
            out.append(r"\ensuremath{\times}")
        elif g == "±":
            out.append(r"\ensuremath{\pm}")
        elif g == "−":
            out.append(r"\ensuremath{-}")
        else:
            out.append(g)
        prev = g[-1]
    return "".join(out), prev


def inl(s, lead_ok=True, heading=False):
    toks = re.split(r"(`[^`]+`|\$[^$]+\$|\*\*|\*)", s)
    out, stack, prev, visible = [], [], "", False
    for t in toks:
        if t == "":
            continue
        if t == "**":
            if stack and stack[-1] == "b":
                stack.pop()
                out.append("}")
                prev = "a"
            else:
                out.append(r"\lead{" if (lead_ok and not visible) else r"\textbf{")
                stack.append("b")
                prev = " "
        elif t == "*":
            if stack and stack[-1] == "i":
                stack.pop()
                out.append("}")
                prev = "a"
            else:
                out.append(r"\emph{")
                stack.append("i")
                prev = " "
        elif t.startswith("`"):
            code = r"\code{%s}" % esc_code(t[1:-1])
            out.append(r"\texorpdfstring{%s}{%s}" % (code, plain(t)) if heading else code)
            visible, prev = True, "a"
        elif t.startswith("$"):
            out.append(r"\texorpdfstring{%s}{%s}" % (t, plain(t)) if heading else t)
            visible, prev = True, "a"
        else:
            conv, prev = text(t, prev)
            out.append(conv)
            if re.sub(r"\[NEW\]|\[UPDATED\]|\s", "", t):
                visible = True
    assert not stack, "unbalanced emphasis in: " + s
    return "".join(out)


# ----------------------------------------------------------------------------------------------------------
# block parsing

lines = open(SRC, encoding="utf-8").read().split("\n")
ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
SEP = re.compile(r"^\|[-: |]+\|$")


def is_block_start(l):
    return (l.strip() == "" or l.startswith("#") or l.startswith("![") or l.startswith("$$") or l.startswith("|")
            or ITEM.match(l) is not None)


blocks, i = [], 0
while i < len(lines):
    l = lines[i]
    if l.strip() == "":
        i += 1
    elif l.startswith("# "):
        blocks.append(("title", l[2:]))
        i += 1
    elif l.startswith("## ") or l.startswith("### "):
        level = 2 if l.startswith("## ") else 3
        blocks.append(("h%d" % level, l[level + 1:]))
        i += 1
    elif l.startswith("!["):
        m = re.match(r"^!\[(.*)\]\((.*?)\)$", l)
        assert m, l
        blocks.append(("figure", (m.group(1), m.group(2))))
        i += 1
    elif l.startswith("$$"):
        assert l.rstrip().endswith("$$") and len(l) > 4, "display math must sit on one line: " + l
        blocks.append(("math", l.strip()[2:-2].strip()))
        i += 1
    elif l.startswith("|") and i + 1 < len(lines) and SEP.match(lines[i + 1]):
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in lines[i:j] if not SEP.match(r)]
        blocks.append(("table", rows))
        i = j
    elif ITEM.match(l):
        items = []
        top = ITEM.match(l)
        top_kind = "ol" if top.group(2)[0].isdigit() else "ul"
        while i < len(lines):
            l = lines[i]
            m = ITEM.match(l)
            if m:
                kind = "ol" if m.group(2)[0].isdigit() else "ul"
                items.append([len(m.group(1)), kind, m.group(3), m.group(2)])
                i += 1
            elif l.strip() == "":
                j = i
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                nm = ITEM.match(lines[j]) if j < len(lines) else None
                if nm and nm.group(1) == "" and ("ol" if nm.group(2)[0].isdigit() else "ul") == top_kind:
                    i = j
                else:
                    break
            elif l.startswith(" ") and not l.startswith("|"):
                items[-1][2] += " " + l.strip()
                i += 1
            else:
                break
        blocks.append(("list", items))
    else:
        par = []
        while i < len(lines) and not (par and is_block_start(lines[i])) and lines[i].strip() != "":
            par.append(lines[i].strip())
            i += 1
        blocks.append(("para", " ".join(par)))

# ----------------------------------------------------------------------------------------------------------
# rendering


def render_list(items):
    out, stack = [], []
    counters = {}

    def close():
        kind, _ = stack.pop()
        out.append(r"\end{%s}" % ("itemize" if kind == "ul" else "enumerate"))

    for indent, kind, txt, mark in items:
        while stack and indent < stack[-1][1]:
            close()
        if not stack or indent > stack[-1][1] or kind != stack[-1][0]:
            if stack and indent == stack[-1][1]:
                close()
            stack.append((kind, indent))
            out.append(r"\begin{%s}" % ("itemize" if kind == "ul" else "enumerate"))
            counters[indent] = 0
            if kind == "ol":
                assert mark == "1.", "numbered lists must start at 1 (found %s)" % mark
        if kind == "ol":
            counters[indent] += 1
            assert mark == "%d." % counters[indent], "list numbering is not consecutive near: " + txt[:50]
        out.append(r"\item " + inl(txt))
    while stack:
        close()
    return "\n".join(out)


def callout_of(par):
    m = re.match(r"^\*\*([^*]+?)\*\*(.*)$", par)
    if not m:
        return None
    head = m.group(1).strip().rstrip(".:")
    if head in CALLOUTS:
        return CALLOUTS[head], head, m.group(2).strip()
    return None


def hdr(c):
    return r"\hdr{%s}" % inl(c, lead_ok=False)


def render_table(idx, rows):
    spec = TABLE_SPECS[idx]
    size = spec.get("size", "small")
    special = spec.get("special")
    ncol = len(rows[0])
    body = rows[1:]
    out = []
    if special == "patch_all":
        head = [r"& & & & \multicolumn{5}{c}{\hdr{restored when patching at layer $\ell$}} \\", r"\cmidrule(lr){5-9}",
                " & ".join(hdr(c) for c in rows[0]) + r" \\", r"\midrule"]
        # the five layer columns are labelled 0..4 in the Markdown header
        trs, prev_L = [], None
        for ri, r in enumerate(body):
            if prev_L is not None and r[0] != prev_L:
                trs.append(r"\addlinespace[4pt]")
            # \\* forbids a page break inside a group of rows with the same L
            end = r"\\*" if (ri + 1 < len(body) and body[ri + 1][0] == r[0]) else r"\\"
            trs.append(" & ".join(inl(c, lead_ok=False) for c in r) + " " + end)
            prev_L = r[0]
        content = head + trs
    elif special == "patch_subsets":
        names = rows[0][1:]
        head = [r"\hdr{corrupted hop} & \hdr{patched positions} & \multicolumn{5}{c}{\hdr{restored when patching at layer $\ell$}} \\",
                r"\cmidrule(lr){3-7}", r" & & " + " & ".join(hdr(str(k)) for k in range(5)) + r" \\", r"\midrule"]
        trs = []
        for n, r in enumerate(body):
            if n:
                trs.append(r"\addlinespace[4pt]")
            for k, c in enumerate(r[1:]):
                vals = [v.strip() for v in c.split("/")]
                if c.endswith("for all"):
                    cells = r"\multicolumn{5}{c}{%s at every layer}" % c.replace(" for all", "").strip()
                else:
                    assert len(vals) == 5, c
                    cells = " & ".join(vals)
                label = r"\textbf{%s}" % inl(r[0], lead_ok=False) if k == 0 else ""
                # \\* forbids a page break inside one corrupted-hop group (the table is a longtable and may break between groups)
                end = r"\\" if k == len(r) - 2 else r"\\*"
                trs.append("%s & %s & %s %s" % (label, inl(names[k], lead_ok=False), cells, end))
        content = head + trs
    else:
        head = [" & ".join(hdr(c) for c in rows[0]) + r" \\", r"\midrule"]
        trs = [" & ".join(inl(c, lead_ok=False) for c in r) + r" \\" for r in body]
        content = head + trs
    if spec.get("long") or len(rows) >= 7:
        first = [r"\toprule"] + content[:len(head)] if special is None else [r"\toprule"] + content[:len(head)]
        out.append(r"{\%s\nopagebreak[4]" % size)       # keep the lead-in line with the start of the table
        out.append(r"\begin{longtable}{@{}%s@{}}" % spec["cols"])
        out += first
        out.append(r"\endfirsthead")
        out += first
        out.append(r"\endhead")
        out.append(r"\bottomrule")
        out.append(r"\endlastfoot")
        out += content[len(head):]
        out.append(r"\end{longtable}}")
    else:
        out.append(r"\begin{center}\%s" % size)
        out.append(r"\begin{tabular}{@{}%s@{}}" % spec["cols"])
        out.append(r"\toprule")
        out += content
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{center}")
    return "\n".join(out)


body_parts = []        # (kind, tex)
sec_counter = [0, 0]
table_index = 0
first_section_seen = False
front_blocks = []

# title check
assert blocks[0][0] == "title"
if blocks[0][1].strip() != EXPECTED_TITLE:
    print("WARNING: the title changed; update EXPECTED_TITLE and the title block in the preamble.", file=sys.stderr)
blocks = blocks[1:]

k = 0
while k < len(blocks):
    kind, data = blocks[k]
    if kind in ("h2", "h3"):
        m = re.match(r"^(\d+(?:\.\d+)?)\.? (.*)$", data)
        assert m, data
        num, title = m.group(1), m.group(2)
        if kind == "h2":
            sec_counter[0] += 1
            sec_counter[1] = 0
            expect = str(sec_counter[0])
        else:
            sec_counter[1] += 1
            expect = "%d.%d" % tuple(sec_counter)
        assert num == expect, "heading number %s, expected %s" % (num, expect)
        if not first_section_seen:
            first_section_seen = True
            body_parts.append(("toc", r"\tableofcontents"))
        cmd = "section" if kind == "h2" else "subsection"
        banner = ("% " + "=" * 100 + "\n%% %s %s\n%% " % (num, plain(title).strip()) + "=" * 100 + "\n") if kind == "h2" else ""
        body_parts.append((kind, banner + r"\%s{%s}\label{sec:%s}" % (cmd, inl(title, lead_ok=False, heading=True), num)))
    elif not first_section_seen and kind in ("para", "list"):
        front_blocks.append((kind, data))
    elif kind == "para":
        mfig = re.match(r"^\*\*Figure:\*\* `([^`]+\.png)`", data)
        if mfig:
            fn = mfig.group(1)
            body_parts.append(("figure", r"\begin{figure}[htbp]\centering\includegraphics[width=%.2f\linewidth]{%s}"
                               r"\caption{%s}\end{figure}" % (FIGURE_WIDTH[fn], fn, FIGURE_CAPTION_FOR_FILE_LINE[fn])))
        else:
            co = callout_of(data)
            if co:
                ck, head, rest = co
                inner = []
                if rest:
                    inner.append(inl(rest, lead_ok=False))
                if k + 1 < len(blocks) and blocks[k + 1][0] == "list":
                    inner.append(render_list(blocks[k + 1][1]))
                    k += 1
                tex = "\\begin{%sbox}{%s}\n%s\n\\end{%sbox}" % (ck, head, "\n\n".join(inner), ck)
                body_parts.append(("box", tex))
            else:
                t = inl(data)
                if data.count("`") >= 12:      # a paragraph that is mostly file names: do not justify it
                    t = "{\\raggedright " + t + "\\par}"
                body_parts.append(("para", t))
    elif kind == "list":
        body_parts.append(("list", render_list(data)))
    elif kind == "math":
        body_parts.append(("math", r"\[ %s \]" % data))
    elif kind == "figure":
        cap, fn = data
        body_parts.append(("figure", r"\begin{figure}[htbp]\centering\includegraphics[width=%.2f\linewidth]{%s}"
                           r"\caption{%s}\end{figure}" % (FIGURE_WIDTH[fn], fn, inl(cap, lead_ok=False))))
    elif kind == "table":
        assert table_index in TABLE_SPECS, "no column specification for table %d (%s)" % (table_index, data[0][:2])
        body_parts.append(("table", render_table(table_index, data)))
        table_index += 1
    k += 1
assert table_index == len(TABLE_SPECS), "expected %d tables, found %d" % (len(TABLE_SPECS), table_index)

# front matter: the first paragraph is the intro, the rest goes into the grey box
front_tex = []
intro_done = False
inbox = []
for kind, data in front_blocks:
    if not intro_done and kind == "para" and not data.startswith("**"):
        front_tex.append(inl(data))
        intro_done = True
        continue
    inbox.append(inl(data) if kind == "para" else render_list(data))
front_tex.append("\\begin{frontbox}\n" + "\n\n".join(inbox) + "\n\\end{frontbox}")

import textwrap


def wrap_tex(txt, width=112):
    out = []
    for line in txt.split("\n"):
        if len(line) <= width or " & " in line or line.startswith(("%", "\\begin{figure}", "\\includegraphics")):
            out.append(line)
            continue
        ind = len(line) - len(line.lstrip())
        cont = ind + (2 if line.lstrip().startswith("\\item") else 0)
        words = textwrap.wrap(line.strip(), width=width - ind, break_long_words=False, break_on_hyphens=False)
        out.append(" " * ind + words[0])
        for w2 in textwrap.wrap(" ".join(words[1:]), width=width - cont, break_long_words=False, break_on_hyphens=False):
            out.append(" " * cont + w2)
    return "\n".join(out)


# join blocks: no blank line around display maths so the sentence continues after it
out_body = []
for n, (kind, tex) in enumerate(body_parts):
    nxt = body_parts[n + 1][0] if n + 1 < len(body_parts) else None
    if kind == "para" and nxt == "table" and "restored when patching at layer" in body_parts[n + 1][1]:
        # keep a lead-in line with the start of its table: reserve room for the line, the header and the first rows
        ttex = body_parts[n + 1][1]
        nrows = ttex.count("\\\\")
        need = (min(nrows, 8) + 5) if "longtable" in ttex else (nrows + 4)
        tex = r"\Needspace*{%d\baselineskip}" % (need + 3) + tex
    out_body.append(tex)
    if kind == "math" or nxt == "math":
        out_body.append("\n")
    elif kind in ("h2", "h3"):
        out_body.append("\n")
    else:
        out_body.append("\n\n")
body_tex = "".join(out_body)


# ----------------------------------------------------------------------------------------------------------
# document

PREAMBLE = r"""% !TEX program = xelatex
% Generated by make_tex.py from report.md.  Compile with XeLaTeX (or LuaLaTeX), twice:  xelatex report.tex
% Fonts: Latin Modern (text, math, code) and TeX Gyre Heros (headings); both ship with TeX Live.
\documentclass[paper=a4,fontsize=11pt,DIV=11,parskip=half,headings=small,numbers=noendperiod]{scrartcl}

\usepackage{amsmath,mathtools}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{Latin Modern Roman}
\setsansfont{TeX Gyre Heros}[Scale=MatchLowercase]
\setmonofont{Latin Modern Mono}[Scale=0.94,HyphenChar=None]
\setmathfont{Latin Modern Math}
\usepackage{microtype}
\frenchspacing
\emergencystretch=2em

\usepackage{xcolor}
\definecolor{accent}{HTML}{1F4E79}
\definecolor{newcol}{HTML}{1B7F4B}
\definecolor{updcol}{HTML}{B45309}
\definecolor{findbg}{HTML}{EFF4FA}
\definecolor{limbg}{HTML}{FBF4E6}
\definecolor{limcol}{HTML}{C58B2A}
\definecolor{frontbg}{HTML}{F3F3F1}
\definecolor{rulegray}{HTML}{7A7A7A}

\usepackage{graphicx}
\graphicspath{{./}{figures/}}
\usepackage{booktabs,array,longtable,multirow}
\usepackage{enumitem}
\usepackage{needspace}
\usepackage{caption}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

\usepackage[headsepline,footsepline=false,plainheadsepline=false]{scrlayer-scrpage}
\clearpairofpagestyles
\ihead{\sffamily\small Two trained models and a training experiment}
\ohead{\sffamily\small @@VERSION@@ \textperiodcentered{} @@DATE@@}
\cfoot*{\sffamily\small\pagemark}

\setkomafont{disposition}{\sffamily\bfseries\color{accent}}
\addtokomafont{section}{\Large}
\addtokomafont{subsection}{\large}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{2}

\captionsetup{font=small,labelfont={sf,bf},labelsep=period,justification=raggedright,singlelinecheck=false,skip=6pt}

\setlist{itemsep=0.28em,topsep=0.3em,parsep=0pt,partopsep=0pt}
\setlist[itemize,1]{leftmargin=1.4em,label={\color{accent}\textbullet}}
\setlist[itemize,2]{leftmargin=1.3em,label={\color{accent}\textendash},itemsep=0.15em}
\setlist[enumerate,1]{leftmargin=2.0em,label={\color{accent}\sffamily\bfseries\small\arabic*.},itemsep=0.45em}

\renewcommand{\arraystretch}{1.16}
\setlength{\LTpre}{4pt}
\setlength{\LTpost}{8pt}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.12}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{2}
\setlength{\tabcolsep}{5pt}
\setlength{\heavyrulewidth}{0.9pt}
\setlength{\lightrulewidth}{0.5pt}

% text macros (robust, so they may appear in headings, captions and tables)
\DeclareRobustCommand{\lead}[1]{\textsf{\bfseries\color{accent}#1}}
\DeclareRobustCommand{\code}[1]{{\fontseries{m}\ttfamily #1}}
\DeclareRobustCommand{\hdr}[1]{\textsf{\bfseries #1}}
\DeclareRobustCommand{\boxlabel}[2]{{\sffamily\bfseries\small\color{#1}#2}}

% [NEW] and [UPDATED] tags
\newcommand{\tagpill}[2]{\tcbox[on line,colback=#1,colframe=#1,boxrule=0pt,arc=1.6pt,left=2.4pt,right=2.4pt,top=0.9pt,bottom=0.9pt,boxsep=0pt,tcbox raise base,fontupper={\sffamily\bfseries\fontsize{6.8}{7}\selectfont\color{white}}]{#2}}
\DeclareRobustCommand{\tagnew}{\texorpdfstring{\tagpill{newcol}{NEW}}{[NEW]}}
\DeclareRobustCommand{\tagupd}{\texorpdfstring{\tagpill{updcol}{UPDATED}}{[UPDATED]}}

% boxes: findings (blue), limits and caveats (amber), front matter (grey)
\newtcolorbox{findbox}[1]{enhanced,breakable,colback=findbg,colframe=findbg,boxrule=0pt,sharp corners,
  borderline west={2.4pt}{0pt}{accent},left=9pt,right=8pt,top=5pt,bottom=4pt,before skip=8pt,after skip=8pt,
  before upper={\boxlabel{accent}{#1}\par\nopagebreak\vspace{1pt}}}
\newtcolorbox{limbox}[1]{enhanced,breakable,colback=limbg,colframe=limbg,boxrule=0pt,sharp corners,
  borderline west={2.4pt}{0pt}{limcol},left=9pt,right=8pt,top=5pt,bottom=4pt,before skip=8pt,after skip=8pt,
  before upper={\boxlabel{limcol}{#1}\par\nopagebreak\vspace{1pt}}}
\newtcolorbox{frontbox}{enhanced,breakable,colback=frontbg,colframe=frontbg,boxrule=0pt,arc=3pt,
  left=10pt,right=10pt,top=6pt,bottom=6pt,before skip=10pt,after skip=10pt}

\usepackage[colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent,bookmarksnumbered=true,
  pdftitle={Tests on two trained models, and a training experiment on four more: torn-bigger-13 and elem-partitioned-long-boosted-1},pdfsubject={@@VERSION@@}]{hyperref}

\begin{document}

\begin{flushleft}
{\sffamily\bfseries\color{accent}\small\MakeUppercase{@@VERSION@@}\hspace{0.7em}\textcolor{rulegray}{\textperiodcentered}\hspace{0.7em}\textcolor{rulegray}{@@DATE@@}\par}
\vspace{8pt}
{\sffamily\bfseries\fontsize{24}{28}\selectfont Tests on two trained models,\par}
\vspace{0pt}
{\sffamily\bfseries\fontsize{24}{28}\selectfont and a training experiment on four more\par}
\vspace{8pt}
{\ttfamily\fontsize{15}{19}\selectfont torn-bigger-13}\hspace{0.5em}{\sffamily\large (general transpositions, $S_{25}$)\par}
\vspace{3pt}
{\ttfamily\fontsize{15}{19}\selectfont elem-partitioned-long-boosted-1}\hspace{0.5em}{\sffamily\large (elementary transpositions)\par}
\vspace{6pt}
{\sffamily\color{rulegray}@@SUBTITLE@@\par}
\vspace{6pt}
{\color{accent}\rule{\linewidth}{1.1pt}}
\end{flushleft}

"""

POSTAMBLE = "\\end{document}\n"

doc = (PREAMBLE.replace("@@VERSION@@", VERSION).replace("@@SUBTITLE@@", SUBTITLE).replace("@@DATE@@", DATE)
       + wrap_tex("\n\n".join(front_tex)) + "\n\n" + wrap_tex(body_tex.rstrip()) + "\n\n" + POSTAMBLE)
open(DST, "w", encoding="utf-8").write(doc)
print("wrote %s: %d lines, %d tables, %d section counters %s" % (DST, doc.count("\n"), table_index, sec_counter[0], sec_counter))
