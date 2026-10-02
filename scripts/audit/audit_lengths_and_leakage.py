"""Read-only audit of the processed fine-tuning datasets.

Reports, per task: token lengths and how much of the answer survives
truncation, the shape of the target outputs, and train/val overlap.

Needs `data/processed/{detection,fixing}/{train,val}` (not in git) and
`pyarrow`, `tokenizers`, `numpy`. Run from anywhere:

    python scripts/audit/audit_lengths_and_leakage.py
"""
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
from tokenizers import Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

local_tok = ROOT / "models" / "detection_merged" / "tokenizer.json"
tok = Tokenizer.from_file(str(local_tok)) if local_tok.exists() else Tokenizer.from_pretrained(BASE_MODEL)
tok.no_truncation()
tok.no_padding()


def load(p):
    with ipc.open_stream(str(p / "data-00000-of-00001.arrow")) as r:
        return r.read_all().column("text").to_pylist()


def ntok(texts):
    return np.array([len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)])


def split(text):
    """(prompt, answer) -- the answer is everything after the last [/INST]."""
    i = text.rfind("[/INST]")
    return text[:i], text[i + len("[/INST]"):].strip()


def clean_out(o):
    o = o.replace("<|EOT|>", "").strip()
    gap = o.startswith("<fim_middle>")
    if gap:
        o = o[len("<fim_middle>"):].strip()
    return gap, o


def pct(a, q):
    return int(np.percentile(a, q))


def md5(s):
    return hashlib.md5(s.encode("utf-8", "replace")).hexdigest()


def audit(name, train, val, rng):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"rows: train={len(train)} val={len(val)}")
    sample = [train[i] for i in rng.choice(len(train), size=min(3000, len(train)), replace=False)]
    _, outs = zip(*(split(t) for t in sample))

    # --- length / truncation ---
    full = ntok(list(sample))
    out_t = ntok(list(outs))
    # the system prompt is the prefix every sample shares
    a, b = sample[0], sample[1]
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    print(f"shared prefix (system prompt) tokens: {len(tok.encode(a[:k], add_special_tokens=False).ids)}")
    print(f"total tokens  p50={pct(full, 50)} p90={pct(full, 90)} p99={pct(full, 99)} max={full.max()}")
    print(f"answer tokens p50={pct(out_t, 50)} p90={pct(out_t, 90)} p99={pct(out_t, 99)} max={out_t.max()}")
    for lim in (1024, 2048, 4096):
        # the answer sits at the end, so right-truncation at `lim` eats it first
        surv = np.clip(lim - (full - out_t), 0, out_t)
        print(f"  max_length={lim}: {(full > lim).mean():.1%} truncated | "
              f"answer partly cut {(surv < out_t).mean():.1%} | answer fully cut {(surv == 0).mean():.1%}")
    print(f"answer share of supervised tokens (no masking): {out_t.sum() / full.sum():.1%}")

    # --- output format ---
    kinds, schemas, smell_labels, n_smells, examples = Counter(), Counter(), Counter(), Counter(), {}
    gaps = 0
    for o in outs:
        gap, body = clean_out(o)
        gaps += gap
        try:
            j = json.loads(body)
        except Exception:
            kinds["not JSON (raw text/code)"] += 1
            examples.setdefault("not JSON", body[:160])
            continue
        if not isinstance(j, dict):
            kinds["JSON non-object"] += 1
            continue
        kinds["valid JSON object"] += 1
        keys = tuple(sorted(j.keys()))
        schemas[keys] += 1
        examples.setdefault(str(keys), body[:220])
        sm = j.get("smells")
        if isinstance(sm, list):
            n_smells[min(len(sm), 5)] += 1
            for s in sm:
                smell_labels[s.get("type", "<dict w/o type>") if isinstance(s, dict) else f"str: {s}"] += 1
    n = len(outs)
    print(f"gap (FIM) samples: {gaps / n:.1%}")
    print("output kinds:")
    for key, v in kinds.most_common():
        print(f"  {v / n:6.1%}  {key}")
    if schemas:
        print("JSON key sets:")
        for key, v in schemas.most_common(8):
            print(f"  {v / n:6.1%}  {key}")
    if n_smells:
        print("smells per sample (5 = 5+):", {key: f"{v / n:.1%}" for key, v in sorted(n_smells.items())})
        print(f"distinct smell labels: {len(smell_labels)}; top 10:")
        for key, v in smell_labels.most_common(10):
            print(f"  {v:5d}  {key[:110]}")
    print("examples of each output shape:")
    for key, v in examples.items():
        print(f"  [{key}] {v!r}")

    # --- leakage ---
    tr_full = {md5(t) for t in train}
    tr_prompt = {md5(split(t)[0]) for t in train}
    v_full = sum(md5(t) in tr_full for t in val)
    v_prompt = sum(md5(split(t)[0]) in tr_prompt for t in val)
    print(f"val rows identical to a train row: {v_full}/{len(val)} ({v_full / len(val):.1%}); "
          f"same prompt: {v_prompt / len(val):.1%}")
    print(f"duplicate rows inside train: {len(train) - len(tr_full)} ({1 - len(tr_full) / len(train):.1%})")

    # overlapping-window check: 6-line shingles of the code part
    def shingles(t):
        code = split(t)[0][k:]
        lines = [line.strip() for line in code.splitlines() if len(line.strip()) > 12]
        return {md5("\n".join(lines[i:i + 6])) for i in range(0, max(len(lines) - 5, 0))}

    tr_sh = set()
    for t in train:
        tr_sh |= shingles(t)
    hit = cnt = 0
    for i in rng.choice(len(val), size=min(1000, len(val)), replace=False):
        s = shingles(val[i])
        if not s:
            continue
        cnt += 1
        hit += len(s & tr_sh) / len(s) >= 0.5
    print(f"val samples sharing >=50% of their 6-line code windows with train: {hit}/{cnt} ({hit / max(cnt, 1):.1%})")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for task in ("detection", "fixing"):
        audit(task.upper(), load(DATA / task / "train"), load(DATA / task / "val"), rng)
