"""Build the clean detection dataset from the legacy processed data.

Reads  data/processed/{detection,fixing}/{train,val}   (Arrow, one `text` column
       with the old system prompt baked in)
Writes data/clean/detection/{train,val,test}.jsonl
       data/clean/detection/{val,test}_length_matched.jsonl
       data/clean/detection/stats.json

Only full-snippet detection rows are kept. Gap rows ask about code that is not
in the input, and the fixing rows contain no real fixes; docs/DATASET.md has
the evidence. Run from anywhere:

    python scripts/build_clean_dataset.py
    python scripts/build_clean_dataset.py --max-tokens 2560

Needs `pyarrow`, `numpy` and `transformers` (tokenizer only).
"""
import argparse
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed"
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

HEADS = {
    "Detect code smells in this C++ snippet:": "detection, full snippet",
    "Detect smells in the gap:": "detection, gap",
    "Refactor this C++ code to fix smells:": "fixing, full snippet",
    "Fill and fix the gap:": "fixing, gap",
}
KEEP = "Detect code smells in this C++ snippet:"

# Label vocabulary found in the source data, most severe first.
CATEGORIES = [
    "High Severity (80%+)",
    "Medium-High Severity (60-79%)",
    "Medium Severity (40-59%)",
    "Maintainability Smells",
    "Performance Smells",
]
CHECK_NAME = re.compile(r"[a-z][a-z0-9]*(-[A-Za-z0-9.+]+)+")

# Ends with a newline because the model's chat template puts "### Instruction:"
# directly after the system text.
SYSTEM_PROMPT = (
    "You are a C++ static-analysis assistant for game-engine code. "
    "Given a C++ snippet, report the clang-tidy findings it would trigger.\n"
    "Answer with JSON only, in this form:\n"
    '{"smells": [<categories>], "severity": <0-100>, "checks": [<clang-tidy check names>]}\n'
    'Categories: "High Severity (80%+)", "Medium-High Severity (60-79%)", '
    '"Medium Severity (40-59%)", "Maintainability Smells", "Performance Smells".\n'
    'For clean code answer {"smells": [], "severity": 0, "checks": []}\n'
)
USER_TEMPLATE = "Detect code smells in this C++ snippet:\n\n{code}"

# The legacy builder dropped the newline where it re-joined chunks, which glues a
# preprocessor directive onto the previous token ("#pragma once#include ...").
GLUED_DIRECTIVE = re.compile(r"(?<=[A-Za-z0-9_\">)])(#[ \t]*(?:define|include|ifdef|ifndef|endif|pragma|undef)\b)")
STRAY_MARKERS = ("<fim_", "[INST]", "[/INST]", "<|EOT|>", "<｜begin▁of▁sentence｜>", "### Response:", "\x00")


def load_arrow(path):
    with ipc.open_stream(str(path / "data-00000-of-00001.arrow")) as r:
        return r.read_all().column("text").to_pylist()


def norm(s):
    return " ".join(s.split())


def md5(s):
    return hashlib.md5(s.encode("utf-8", "replace")).hexdigest()


def split_row(text):
    """(head, body, answer) of one legacy row, or None if it has no known instruction."""
    end = text.rfind("[/INST]")
    best = None
    for head in HEADS:
        # the last occurrence is the real instruction; earlier ones are prompt examples
        p = text.rfind(head, 0, end)
        if p >= 0 and (best is None or p > best[0]):
            best = (p, head)
    if best is None:
        return None
    p, head = best
    answer = text[end + len("[/INST]"):].replace("<|EOT|>", "").strip()
    return head, text[p + len(head):end], answer


def parse_label(answer):
    """Normalised (smells, severity, checks), or a string naming why the label is rejected."""
    try:
        j = json.loads(answer)
    except json.JSONDecodeError:
        return "label is not JSON"
    if not isinstance(j, dict) or set(j) != {"smells", "severity", "explanation"}:
        return "unexpected label keys"
    atoms = {a.strip() for s in j["smells"] for a in str(s).split(",")}
    if not atoms <= set(CATEGORIES):
        return "category outside the vocabulary ('Other')"
    smells = [c for c in CATEGORIES if c in atoms]
    if not smells:
        if j["severity"] != 0 or j["explanation"] != "No smells detected":
            return "clean label with severity or checks set"
        return [], 0, []
    checks = sorted({c.strip() for c in j["explanation"].split(",") if c.strip()})
    if not checks or not all(CHECK_NAME.fullmatch(c) for c in checks):
        return "smelly label without valid check names"
    return smells, int(j["severity"]), checks


def clean_code(code):
    code = code.replace("\r\n", "\n").strip()
    fixed, n = GLUED_DIRECTIVE.subn(r"\n\1", code)
    return fixed, n > 0


def code_windows(code, n=6):
    """Hashes of every run of n consecutive substantial, non-comment lines."""
    lines = []
    for line in code.splitlines():
        s = line.strip()
        if len(s) < 12 or s.startswith(("//", "/*", "*", "#include")):
            continue
        lines.append(s)
    return {md5("\n".join(lines[i:i + n])) for i in range(max(len(lines) - n + 1, 0))}


def group_by_shared_code(codes):
    """Union rows that share any code window, so overlapping chunks stay in one split."""
    parent = list(range(len(codes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for i, code in enumerate(codes):
        for w in code_windows(code):
            if w in owner:
                parent[find(i)] = find(owner[w])
            else:
                owner[w] = i
    roots = [find(i) for i in range(len(codes))]
    ids = {r: g for g, r in enumerate(sorted(set(roots)))}
    return [ids[r] for r in roots]


def assign_splits(groups, rng, fractions=(("train", 0.8), ("val", 0.1), ("test", 0.1)), big=50):
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    total = len(groups)
    target = {name: f * total for name, f in fractions}
    have = Counter()
    split_of = {}
    order = sorted(members)
    rng.shuffle(order)
    order.sort(key=lambda g: len(members[g]) <= big)  # big groups first, into train
    for g in order:
        size = len(members[g])
        if size > big:
            name = "train"
        else:
            name = max(target, key=lambda s: (target[s] - have[s]) / target[s])
        have[name] += size
        split_of[g] = name
    return [split_of[g] for g in groups]


def length_matched(rows, edges, rng):
    """Within each length bin keep equally many smelly and clean rows."""
    bins = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        bins[int(np.searchsorted(edges, r["n_tokens"], side="right"))][bool(r["smells"])].append(r)
    out = []
    for b in sorted(bins):
        k = min(len(bins[b][True]), len(bins[b][False]))
        for cls in (True, False):
            out += rng.sample(bins[b][cls], k)
    return sorted(out, key=lambda r: r["id"])


def count_tokens(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return len(ids)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(args):
    from transformers import AutoTokenizer

    local = ROOT / "models" / "detection_merged"
    tokenizer = AutoTokenizer.from_pretrained(str(local) if local.exists() else BASE_MODEL)
    rng = random.Random(args.seed)
    stats = {"seed": args.seed, "max_tokens": args.max_tokens, "source_rows": {}, "dropped": Counter()}

    # 1. pull instruction / answer pairs back out of the legacy text
    candidates = []
    for task in ("detection", "fixing"):
        for part in ("train", "val"):
            for text in load_arrow(SRC / task / part):
                parsed = split_row(text)
                kind = HEADS[parsed[0]] if parsed else "unrecognised"
                key = f"{task} file: {kind}"
                stats["source_rows"][key] = stats["source_rows"].get(key, 0) + 1
                if parsed and parsed[0] == KEEP:
                    candidates.append((parsed[1], parsed[2], task))
    stats["candidates"] = len(candidates)

    # 2. normalise labels and code
    rows, repaired = [], 0
    for body, answer, task in candidates:
        label = parse_label(answer)
        if isinstance(label, str):
            stats["dropped"][label] += 1
            continue
        code, was_repaired = clean_code(body)
        if not code:
            stats["dropped"]["empty code"] += 1
            continue
        if any(m in code for m in STRAY_MARKERS):
            stats["dropped"]["prompt marker inside the code"] += 1
            continue
        repaired += was_repaired
        rows.append({"code": code, "label": label, "source": task})
    stats["glued_directives_repaired"] = repaired

    # 3. exact duplicates (ignoring whitespace); conflicting labels drop every copy
    by_code = defaultdict(list)
    for r in rows:
        by_code[norm(r["code"])].append(r)
    rows = []
    for copies in by_code.values():
        if len({json.dumps(c["label"]) for c in copies}) > 1:
            stats["dropped"]["duplicate snippet with conflicting labels"] += len(copies)
            continue
        stats["dropped"]["exact duplicate"] += len(copies) - 1
        rows.append(copies[0])

    # 4. render the conversation and enforce the token budget
    kept = []
    for r in rows:
        smells, severity, checks = r["label"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(code=r["code"])},
            {"role": "assistant", "content": json.dumps({"smells": smells, "severity": severity, "checks": checks})},
        ]
        n = count_tokens(tokenizer, messages)
        if n > args.max_tokens:
            stats["dropped"][f"longer than {args.max_tokens} tokens"] += 1
            continue
        kept.append({"id": "det-" + md5(norm(r["code"]))[:12], "code": r["code"], "smells": smells,
                     "severity": severity, "checks": checks, "n_tokens": n, "source_file": r["source"],
                     "messages": messages})
    rows = sorted(kept, key=lambda r: r["id"])

    # 5. split by groups of overlapping code
    groups = group_by_shared_code([r["code"] for r in rows])
    splits = assign_splits(groups, rng)
    for r, g, s in zip(rows, groups, splits):
        r["group"] = g
        r["split"] = s

    # 6. write
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ("id", "group", "smells", "severity", "checks", "n_tokens", "source_file", "messages")
    by_split = {s: [{k: r[k] for k in fields} for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    for name, part in by_split.items():
        write_jsonl(out / f"{name}.jsonl", part)
    lengths = np.array([r["n_tokens"] for r in rows])
    edges = [int(x) for x in np.percentile(lengths, [20, 40, 60, 80])]
    matched = {}
    for name in ("val", "test"):
        matched[name] = length_matched(by_split[name], edges, rng)
        write_jsonl(out / f"{name}_length_matched.jsonl", matched[name])

    # 7. stats
    sizes = Counter(groups)
    stats["dropped"] = dict(stats["dropped"])
    stats["rows"] = len(rows)
    stats["groups"] = {"count": len(sizes), "multi_row_groups": sum(v > 1 for v in sizes.values()),
                       "largest": sorted(sizes.values(), reverse=True)[:5]}
    stats["splits"] = {}
    for name, part in {**by_split, **{f"{k}_length_matched": v for k, v in matched.items()}}.items():
        n_smelly = sum(bool(r["smells"]) for r in part)
        stats["splits"][name] = {"rows": len(part), "smelly": n_smelly, "clean": len(part) - n_smelly}
    stats["category_rows"] = {c: sum(c in r["smells"] for r in rows) for c in CATEGORIES}
    stats["distinct_checks"] = len({c for r in rows for c in r["checks"]})
    stats["tokens"] = {f"p{q}": int(np.percentile(lengths, q)) for q in (50, 90, 99)} | {"max": int(lengths.max())}
    stats["fits_max_length"] = {}
    smelly = np.array([bool(r["smells"]) for r in rows])
    for lim in (1024, 2048, 2560, 3072, 4096):
        fits = lengths <= lim
        stats["fits_max_length"][str(lim)] = {"all": round(float(fits.mean()), 4),
                                              "smelly": round(float(fits[smelly].mean()), 4),
                                              "clean": round(float(fits[~smelly].mean()), 4)}
    stats["length_bin_edges"] = edges
    stats["smelly_share_by_length_bin"] = {}
    for b in range(len(edges) + 1):
        m = np.searchsorted(edges, lengths, side="right") == b
        lo = 0 if b == 0 else edges[b - 1]
        hi = edges[b] if b < len(edges) else int(lengths.max())
        stats["smelly_share_by_length_bin"][f"{lo}-{hi}"] = {"rows": int(m.sum()), "smelly": round(float(smelly[m].mean()), 3)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out, stats


def verify(out, max_tokens):
    """Re-read what was written and check it independently of the build step."""
    from transformers import AutoTokenizer

    local = ROOT / "models" / "detection_merged"
    tokenizer = AutoTokenizer.from_pretrained(str(local) if local.exists() else BASE_MODEL)
    parts = {name: read_jsonl(out / f"{name}.jsonl") for name in ("train", "val", "test")}
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' -- ' + detail if detail else ''}")

    rows = [r for part in parts.values() for r in part]
    keys = {"id", "group", "smells", "severity", "checks", "n_tokens", "source_file", "messages"}
    check("every row has the expected fields", all(set(r) == keys for r in rows))
    check("messages are system / user / assistant",
          all([m["role"] for m in r["messages"]] == ["system", "user", "assistant"] for r in rows))
    check("one system prompt throughout", len({r["messages"][0]["content"] for r in rows}) == 1)

    bad = 0
    for r in rows:
        a = json.loads(r["messages"][2]["content"])
        ok = (list(a) == ["smells", "severity", "checks"] and a["smells"] == r["smells"] and a["checks"] == r["checks"]
              and a["smells"] == [c for c in CATEGORIES if c in a["smells"]]
              and (bool(a["smells"]) == bool(a["checks"])) and (a["smells"] or a["severity"] == 0))
        bad += not ok
    check("every answer is valid JSON in the one schema", bad == 0, f"{bad} bad")

    codes = {r["id"]: r["messages"][1]["content"][len(USER_TEMPLATE.format(code="")):] for r in rows}
    check("ids are unique", len(codes) == len(rows))
    check("no snippet appears twice", len({norm(c) for c in codes.values()}) == len(rows))
    check("no prompt markers inside the code", not any(m in c for c in codes.values() for m in STRAY_MARKERS))
    check("no glued preprocessor directives left", not any(GLUED_DIRECTIVE.search(c) for c in codes.values()))

    group_split = defaultdict(set)
    for name, part in parts.items():
        for r in part:
            group_split[r["group"]].add(name)
    check("no group spans two splits", all(len(s) == 1 for s in group_split.values()))
    windows = {name: set().union(*(code_windows(codes[r["id"]]) for r in part)) for name, part in parts.items()}
    shared = len(windows["train"] & windows["val"]) + len(windows["train"] & windows["test"]) + len(windows["val"] & windows["test"])
    check("no 6-line code window is shared between splits", shared == 0, f"{shared} shared")

    recount = [count_tokens(tokenizer, r["messages"]) for r in rows]
    check("stored token counts match a fresh count", recount == [r["n_tokens"] for r in rows])
    check(f"every conversation fits in {max_tokens} tokens", max(recount) <= max_tokens, f"max {max(recount)}")

    # the answer is the tail of the rendered conversation, so it must survive right-truncation
    sample = random.Random(0).sample(rows, 200)
    tails_ok = all(
        tokenizer.apply_chat_template(r["messages"], tokenize=False).endswith(r["messages"][2]["content"] + "\n<|EOT|>\n")
        for r in sample)
    check("the rendered conversation ends with the answer and <|EOT|>", tails_ok)

    for name in ("val", "test"):
        sub = read_jsonl(out / f"{name}_length_matched.jsonl")
        ids = {r["id"] for r in parts[name]}
        n_smelly = sum(bool(r["smells"]) for r in sub)
        check(f"{name}_length_matched is a balanced subset of {name}",
              all(r["id"] in ids for r in sub) and n_smelly * 2 == len(sub), f"{len(sub)} rows")
    return all(results)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "clean" / "detection"))
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="drop conversations longer than this (code is never truncated)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out, stats = build(args)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out}\nverifying:")
    if not verify(out, args.max_tokens):
        sys.exit(1)


if __name__ == "__main__":
    main()
