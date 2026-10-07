"""How much can the labels in data/clean/detection be trusted?

Three questions, answered from the clean JSONL files:
  1. How well does snippet length alone predict "smelly"?
  2. When a check is named, is the construct it reports actually in the snippet?
  3. Does the same check always map to the same category?

    python scripts/audit/audit_clean_dataset.py
"""
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "clean" / "detection"
USER_PREFIX = "Detect code smells in this C++ snippet:\n\n"

# checks whose trigger is a construct that must be visible in the flagged code
VISIBLE = {
    "cppcoreguidelines-macro-usage": re.compile(r"#\s*define\b"),
    "cppcoreguidelines-macro-to-enum": re.compile(r"#\s*define\b"),
    "bugprone-macro-parentheses": re.compile(r"#\s*define\b"),
    "performance-enum-size": re.compile(r"\benum\b"),
    "cppcoreguidelines-special-member-functions": re.compile(r"\b(class|struct)\b"),
    "cppcoreguidelines-avoid-c-arrays": re.compile(r"\w\s*\[[^\]]*\]"),
    "cppcoreguidelines-pro-type-vararg": re.compile(r"\.\.\.|\b\w*printf\w*\s*\(|\b\w*scanf\w*\s*\(|va_(list|start|arg)"),
    "bugprone-reserved-identifier": re.compile(r"\b(_[A-Z]\w*|\w*__\w*)\b"),
}


def read(name):
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def code(r):
    return r["messages"][1]["content"][len(USER_PREFIX):]


train, val, test = read("train"), read("val"), read("test")
test_matched = read("test_length_matched")
rows = train + val + test

print("=== 1. length as a predictor of 'smelly' ===")
x = np.array([r["n_tokens"] for r in train])
y = np.array([bool(r["smells"]) for r in train])
cands = np.unique(x)
acc = [((x >= t) == y).mean() for t in cands]
t = int(cands[int(np.argmax(acc))])
print(f"rule fitted on train: smelly if the conversation has >= {t} tokens")
for name, part in (("train", train), ("test", test), ("test_length_matched", test_matched)):
    px = np.array([r["n_tokens"] for r in part]) >= t
    py = np.array([bool(r["smells"]) for r in part])
    tp, fp, fn = (px & py).sum(), (px & ~py).sum(), (~px & py).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"  {name:20s} accuracy {(px == py).mean():.1%}  F1 {f1:.3f}  (majority class {max(py.mean(), 1 - py.mean()):.1%})")

print("\n=== 2. is the reported construct in the snippet? ===")
clean_codes = [code(r) for r in rows if not r["smells"]]
print(f"{'check':46s} {'rows':>5s}  construct present   present in clean rows")
for chk, rx in VISIBLE.items():
    flagged = [code(r) for r in rows if chk in r["checks"]]
    if not flagged:
        continue
    present = np.mean([bool(rx.search(c)) for c in flagged])
    base = np.mean([bool(rx.search(c)) for c in clean_codes])
    print(f"{chk:46s} {len(flagged):5d} {present:14.1%} {base:20.1%}")

print("\n=== 3. does a check always map to the same category? ===")
single = defaultdict(Counter)
for r in rows:
    if len(r["checks"]) == 1 and len(r["smells"]) == 1:
        single[r["checks"][0]][r["smells"][0]] += 1
n = sum(sum(v.values()) for v in single.values())
agree = sum(v.most_common(1)[0][1] for v in single.values())
print(f"rows with exactly one check and one category: {n}")
print(f"rows whose category is that check's most common one: {agree / n:.1%}")
print("least consistent frequent checks:")
worst = sorted((v.most_common(1)[0][1] / sum(v.values()), sum(v.values()), c) for c, v in single.items() if sum(v.values()) >= 40)
for share, cnt, c in worst[:6]:
    print(f"  {c:46s} n={cnt:4d}  top category {share:.0%}  {dict(single[c].most_common(3))}")

print("\n=== checks per smelly row ===")
k = Counter(min(len(r["checks"]), 5) for r in rows if r["smells"])
print({(f"{a}+" if a == 5 else a): b for a, b in sorted(k.items())})
print("most frequent checks:", Counter(c for r in rows for c in r["checks"]).most_common(8))
