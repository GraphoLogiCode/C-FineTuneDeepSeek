"""What the labels actually contain, and what the 'fixing' targets are.

Needs `data/processed/{detection,fixing}/train` (not in git) and `pyarrow`,
`numpy`. Run from anywhere:

    python scripts/audit/audit_labels.py
"""
import difflib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def load(p):
    with ipc.open_stream(str(p / "data-00000-of-00001.arrow")) as r:
        return r.read_all().column("text").to_pylist()


def split(text):
    """(prompt, answer) -- the answer is everything after the last [/INST]."""
    i = text.rfind("[/INST]")
    return text[:i], text[i + 7:].replace("<|EOT|>", "").strip()


def shared_prefix_len(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def norm(s):
    return " ".join(s.split())


def audit_detection(det):
    print("=== DETECTION labels ===")
    sev, loc, checks, atoms = Counter(), Counter(), Counter(), Counter()
    smelly_gap = smelly_full = n_gap = n_full = 0
    for t in det:
        _, o = split(t)
        gap = o.startswith("<fim_middle>")
        j = json.loads(o[len("<fim_middle>"):] if gap else o)
        has = bool(j.get("smells"))
        if gap:
            n_gap += 1
            smelly_gap += has
            loc[j.get("location")] += 1
        else:
            n_full += 1
            smelly_full += has
            sev[j.get("severity")] += 1
            for c in str(j.get("explanation", "")).split(","):
                checks[c.strip()] += 1
        for s in j.get("smells", []):
            for a in str(s).split(","):
                atoms[a.strip()] += 1
    print(f"full-snippet: {n_full}, smelly {smelly_full / n_full:.1%} | gap: {n_gap}, smelly {smelly_gap / n_gap:.1%}")
    print("severity values (full-snippet):", sev.most_common(8))
    print("location values (gap):", loc.most_common(8))
    print(f"atomic label vocabulary ({len(atoms)}):", atoms.most_common())
    print(f"'explanation' is a clang-tidy check list; {len(checks)} distinct; top 15:")
    for c, v in checks.most_common(15):
        print(f"  {v:5d}  {c}")


def audit_fixing(fix):
    print("\n=== FIXING targets ===")
    k = shared_prefix_len(fix[0], fix[1])
    rng = np.random.default_rng(1)
    idx = rng.choice(len(fix), 3000, replace=False)
    kinds, shown = Counter(), Counter()
    ratios = []
    contained = n_fullcode = 0
    for i in idx:
        p, o = split(fix[i])
        ins = p[k:]
        gap = o.startswith("<fim_middle>")
        body = o[len("<fim_middle>"):] if gap else o
        try:
            json.loads(body)
            is_json = True
        except Exception:
            is_json = False
        kind = ("gap" if gap else "full") + (" -> detection-style JSON" if is_json else " -> raw code")
        kinds[kind] += 1
        if not gap and not is_json:
            n_fullcode += 1
            code_in = ins.split(":", 1)[1] if ":" in ins else ins
            ci, co = norm(code_in), norm(body)
            contained += co[:400] in ci
            ratios.append(difflib.SequenceMatcher(None, ci[:3000], co[:3000], autojunk=False).quick_ratio())
        if shown[kind] < 1:
            shown[kind] += 1
            print(f"\n--- example [{kind}] ---")
            print("INSTRUCTION (first 300):", repr(ins[:300]))
            print("TARGET (first 300):", repr(body[:300]))
    print("\ntarget kinds:", {key: f"{v / len(idx):.1%}" for key, v in kinds.most_common()})
    if n_fullcode:
        r = np.array(ratios)
        print(f"full-snippet 'fix' targets: {n_fullcode}; "
              f"target's first 400 chars found verbatim in the input: {contained / n_fullcode:.1%}")
        print(f"char-similarity input vs target: p10={np.percentile(r, 10):.2f} "
              f"p50={np.percentile(r, 50):.2f} p90={np.percentile(r, 90):.2f}")
    print("targets containing 'fixed_code':", sum('"fixed_code"' in split(t)[1] for t in fix), "of", len(fix))


if __name__ == "__main__":
    audit_detection(load(DATA / "detection" / "train"))
    audit_fixing(load(DATA / "fixing" / "train"))
