"""Step 3 of 3: turn the labelled functions into training files.

    python scripts/collect/build_dataset.py
    python scripts/collect/build_dataset.py --eligibility relaxed

Reads data/collect/<engine>/units.jsonl, writes

    data/collected/detection/{train,val,test}.jsonl          function -> checks it triggers
    data/collected/detection/{val,test}_length_matched.jsonl
    data/collected/fixing/{train,val,test}.jsonl             function -> function with clang-tidy's own fixes applied
    data/collected/stats.json

and then verifies what it wrote. Splits are by engine, so no engine's code is
in two splits. Which checks count as labels is decided here, from the config,
so the label set can be changed without scanning again.
"""
import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (COGNITIVE_CHECK, ROOT, SIZE_CHECK, engine_root, load_config, matches_any, md5_text, norm_code,
                    read_jsonl, selected_engines, utf8_stdout, work_dir, write_jsonl)

BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Share the existing detection format with the interactive demo.
sys.path.insert(0, str(ROOT))
from src.detection_prompt import DETECT_SYSTEM, DETECT_USER
from src.fixing_prompt import FIX_SYSTEM, FIX_USER

# The system prompt ends with a newline before the chat template's instruction marker.
STRAY_MARKERS = ("### Instruction:", "### Response:", "<|EOT|>", "<｜begin▁of▁sentence｜>", "\x00")

_parser = None


def has_parse_error(code):
    global _parser
    if _parser is None:
        import tree_sitter_cpp
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_cpp.language()))
    return _parser.parse(code.encode("utf-8")).root_node.has_error


def tokenizer():
    from transformers import AutoTokenizer
    # The legacy merged tokenizer produces different IDs/counts from the current
    # base tokenizer. Match train_lora.py so context filtering uses actual tokens.
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def count_tokens(tok, messages):
    ids = tok.apply_chat_template(messages, tokenize=True)
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def dedent(code, indent):
    """Normalise line endings and shift a class member's body back to the margin."""
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if indent:
        lines = [lines[0]] + [ln[len(indent):] if ln.startswith(indent) else ln for ln in lines[1:]]
    return "\n".join(lines)


def selects(policy, check):
    return matches_any(check, policy["include"]) and not matches_any(check, policy["exclude"])


def label_unit(unit, policy):
    """The checks that count as labels for this function, and the findings behind them."""
    limits = policy["thresholds"]

    def wanted(check):
        return selects(policy, check)

    checks, kept = set(), []
    for f in unit["findings"]:
        if wanted(f["check"]):
            checks.add(f["check"])
            kept.append(f)
    m = unit["metrics"]
    size_keys = ("lines", "statements", "branches", "parameters", "variables", "nesting")
    if wanted(SIZE_CHECK) and any(m.get(k, 0) > limits[k] for k in size_keys if k in limits):
        checks.add(SIZE_CHECK)
    if wanted(COGNITIVE_CHECK) and m.get("cognitive", 0) > limits["cognitive"]:
        checks.add(COGNITIVE_CHECK)
    return sorted(checks), kept


def apply_fixes(code, findings):
    """Apply clang-tidy's edits for as many findings as do not collide. Returns (new code, checks fixed)."""
    raw = code.encode("utf-8")
    taken, edits, fixed = [], [], set()
    for f in sorted((f for f in findings if f.get("fix")), key=lambda f: min(e[0] for e in f["fix"])):
        spans = [(o, o + n) for o, n, _ in f["fix"]]
        # Zero-length insertions at the same spot count as a collision: their order would be a guess.
        if any(a <= t1 and t0 <= b for a, b in spans for t0, t1 in taken):
            continue
        taken += spans
        # clang-tidy leaves re-indenting to clang-format. Give any line it starts (typically a closing
        # brace) the indentation of the statement being fixed.
        first = min(o for o, _, _ in f["fix"])
        line_start = raw.rfind(b"\n", 0, first) + 1
        line_end = raw.find(b"\n", line_start)
        line = raw[line_start:line_end if line_end != -1 else len(raw)]
        indent = line[:len(line) - len(line.lstrip(b" \t"))].decode("ascii")
        edits += [[o, n, text.replace("\n", "\n" + indent)] for o, n, text in f["fix"]]
        fixed.add(f["check"])
    if not edits:
        return None, []
    out, pos = [], 0
    for o, n, text in sorted(edits, key=lambda e: e[0]):
        out += [raw[pos:o], text.encode("utf-8")]
        pos = o + n
    out.append(raw[pos:])
    try:
        return b"".join(out).decode("utf-8"), sorted(fixed)
    except UnicodeDecodeError:
        return None, []


def length_bins(values, n_bins):
    qs = np.unique(np.percentile(values, np.linspace(0, 100, n_bins + 1)[1:-1]).astype(int))
    return [int(q) for q in qs]


def bin_of(edges, value):
    return int(np.searchsorted(edges, value, side="right"))


def sample_engine(units, ds, rng):
    """All smelly functions (up to the cap) plus clean ones drawn to follow the same length distribution."""
    smelly = [u for u in units if u["checks"]]
    clean = [u for u in units if not u["checks"]]
    ratio = ds["clean_per_smelly"]
    cap = int(ds["max_units_per_engine"] / (1 + ratio))
    if len(smelly) > cap:
        smelly = rng.sample(smelly, cap)
    if not smelly:
        return []
    edges = length_bins([u["code_tokens"] for u in smelly], ds["length_bins"])
    by_bin = defaultdict(list)
    for u in clean:
        by_bin[bin_of(edges, u["code_tokens"])].append(u)
    picked = []
    for b, n in Counter(bin_of(edges, u["code_tokens"]) for u in smelly).items():
        pool = by_bin.get(b, [])
        picked += rng.sample(pool, min(len(pool), round(n * ratio)))
    return smelly + picked


def length_matched(rows, edges, rng):
    """Equal numbers of smelly and clean rows inside every length bin."""
    bins = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        bins[bin_of(edges, r["n_tokens"])][bool(r["checks"])].append(r)
    out = []
    for b in sorted(bins):
        k = min(len(bins[b][True]), len(bins[b][False]))
        for cls in (True, False):
            out += rng.sample(bins[b][cls], k)
    return sorted(out, key=lambda r: r["id"])


def engine_commit(cfg, engine):
    try:
        r = subprocess.run(["git", "-C", str(engine_root(cfg, engine)), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build(cfg, engines, args):
    ds, policy = cfg["dataset"], cfg["labels"]
    eligibility = dict(ds["eligibility"])
    if args.eligibility:
        eligibility = {"train": args.eligibility, "eval": args.eligibility}
    rng = random.Random(ds["seed"])
    tok = tokenizer()
    split_of = {e: s for s, names in ds["splits"].items() for e in names}
    stats = {"eligibility": eligibility, "engines": {}, "dropped": Counter()}

    # 1. load, filter, label
    units = []
    for engine in engines:
        path = work_dir(cfg, engine) / "units.jsonl"
        if not path.exists():
            print(f"{engine}: no units.jsonl, run extract.py first")
            continue
        n_seen = 0
        mode = eligibility["train" if split_of.get(engine, "train") == "train" else "eval"]
        for u in read_jsonl(path):
            n_seen += 1
            u["file_error_free"] = not (u["file_errors_main"] or u["file_errors_other"])
            if mode == "strict" and not u["file_error_free"]:
                stats["dropped"]["file had a compile error elsewhere (strict eligibility)"] += 1
                continue
            if any(m in u["code"] for m in STRAY_MARKERS):
                stats["dropped"]["chat-template marker inside the code"] += 1
                continue
            aliases = policy["aliases"]
            u["findings"] = [{**f, "check": aliases.get(f["check"], f["check"])} for f in u["findings"]]
            # Fixing has its own policy: the checks worth auto-fixing are not the ones worth detecting.
            u["fixable"] = [f for f in u["findings"] if f.get("fix") and selects(cfg["fixes"], f["check"])]
            u["checks"], u["findings"] = label_unit(u, policy)
            u["split"] = split_of.get(engine, "train")
            units.append(u)
        stats["engines"][engine] = {"split": split_of.get(engine, "train"), "commit": engine_commit(cfg, engine),
                                    "functions_extracted": n_seen}

    # 2. token budget (code only; the prompt and answer are added later)
    lengths = [len(ids) for ids in tok([u["code"] for u in units], add_special_tokens=False)["input_ids"]] if units else []
    kept = []
    for u, n in zip(units, lengths):
        u["code_tokens"] = n
        if n < ds["min_code_tokens"]:
            stats["dropped"][f"fewer than {ds['min_code_tokens']} code tokens"] += 1
        elif n > ds["max_code_tokens"]:
            stats["dropped"][f"more than {ds['max_code_tokens']} code tokens"] += 1
        else:
            kept.append(u)
    units = kept

    # 3. duplicates across files and engines (vendored or copied code)
    by_code = defaultdict(list)
    for u in units:
        by_code[norm_code(u["code"])].append(u)
    units = []
    for copies in by_code.values():
        if len({tuple(c["checks"]) for c in copies}) > 1:
            stats["dropped"]["duplicate function with conflicting labels"] += len(copies)
            continue
        stats["dropped"]["duplicate function"] += len(copies) - 1
        units.append(copies[0])

    # 4. label vocabulary: checks seen often enough in the training engines
    counts = Counter(c for u in units if u["split"] == "train" for c in u["checks"])
    vocab = sorted(c for c, n in counts.items() if n >= policy["min_label_count"])
    vocab_set = set(vocab)
    kept = []
    for u in units:
        if any(c not in vocab_set for c in u["checks"]):
            # It has a finding we do not label, so it cannot honestly be called clean either.
            if not any(c in vocab_set for c in u["checks"]):
                stats["dropped"]["only rare checks (below min_label_count)"] += 1
                continue
            u["checks"] = [c for c in u["checks"] if c in vocab_set]
            u["findings"] = [f for f in u["findings"] if f["check"] in vocab_set]
        kept.append(u)
    units = kept

    # 5. sample per engine, then write detection rows
    out = ROOT / (args.out or cfg["dataset_dir"])
    (out / "detection").mkdir(parents=True, exist_ok=True)
    (out / "fixing").mkdir(parents=True, exist_ok=True)
    by_engine = defaultdict(list)
    for u in units:
        by_engine[u["engine"]].append(u)
    detection = defaultdict(list)
    for engine in sorted(by_engine):
        pool = by_engine[engine]
        chosen = sample_engine(pool, ds, rng)
        info = stats["engines"][engine]
        info.update(eligible=len(pool), eligible_smelly=sum(bool(u["checks"]) for u in pool), selected=len(chosen),
                    selected_smelly=sum(bool(u["checks"]) for u in chosen))
        for u in chosen:
            code = dedent(u["code"], u["indent"])
            messages = [{"role": "system", "content": DETECT_SYSTEM},
                        {"role": "user", "content": DETECT_USER.format(code=code)},
                        {"role": "assistant", "content": json.dumps({"checks": u["checks"]})}]
            detection[u["split"]].append({
                "id": "fn-" + md5_text(norm_code(code))[:12], "engine": engine, "path": u["path"], "name": u["name"],
                "start_line": u["start_line"], "end_line": u["end_line"], "checks": u["checks"],
                "findings": [{"check": f["check"], "line": f["line"], "message": f["message"]} for f in u["findings"]],
                "metrics": u["metrics"], "file_error_free": u["file_error_free"],
                "n_tokens": count_tokens(tok, messages), "messages": messages})
    all_lengths = [r["n_tokens"] for rows in detection.values() for r in rows]
    edges = length_bins(all_lengths, ds["length_bins"]) if all_lengths else []
    for split in ("train", "val", "test"):
        rows = sorted(detection[split], key=lambda r: r["id"])
        write_jsonl(out / "detection" / f"{split}.jsonl", rows)
        if split != "train":
            write_jsonl(out / "detection" / f"{split}_length_matched.jsonl", length_matched(rows, edges, rng))
    (out / "detection" / "labels.json").write_text(
        json.dumps({"checks": vocab, "train_counts": {c: counts[c] for c in vocab}}, indent=1), encoding="utf-8")

    # 6. fixing rows: clang-tidy's own edits applied inside the function
    fixing = defaultdict(list)
    fix_dropped = Counter()
    fix_counts = Counter(c for u in units for c in {f["check"] for f in u["fixable"]})
    for engine in sorted(by_engine):
        rows = []
        candidates = [u for u in by_engine[engine] if u["fixable"]]
        rng.shuffle(candidates)
        # Rarest fixes first, so one very common check (braces) does not crowd out the others.
        candidates.sort(key=lambda u: min(fix_counts[f["check"]] for f in u["fixable"]))
        for u in candidates:
            if len(rows) >= ds["max_fix_pairs_per_engine"]:
                break
            fixed_raw, fixed_checks = apply_fixes(u["code"], u["fixable"])
            if fixed_raw is None:
                fix_dropped["edits could not be applied"] += 1
                continue
            before, after = dedent(u["code"], u["indent"]), dedent(fixed_raw, u["indent"])
            if norm_code(before) == norm_code(after):
                fix_dropped["fix changed nothing but whitespace"] += 1
                continue
            if has_parse_error(after) and not has_parse_error(before):
                fix_dropped["fixed code no longer parses"] += 1
                continue
            messages = [{"role": "system", "content": FIX_SYSTEM},
                        {"role": "user", "content": FIX_USER.format(checks=", ".join(fixed_checks), code=before)},
                        {"role": "assistant", "content": after}]
            n = count_tokens(tok, messages)
            if n > args.max_tokens:
                fix_dropped[f"longer than {args.max_tokens} tokens"] += 1
                continue
            rows.append({"id": "fix-" + md5_text(norm_code(before))[:12], "engine": engine, "path": u["path"],
                         "name": u["name"], "start_line": u["start_line"], "end_line": u["end_line"],
                         "fixed_checks": fixed_checks, "file_error_free": u["file_error_free"], "n_tokens": n,
                         "messages": messages})
        fixing[by_engine[engine][0]["split"]] += rows
        stats["engines"][engine]["fix_pairs"] = len(rows)
    for split in ("train", "val", "test"):
        write_jsonl(out / "fixing" / f"{split}.jsonl", sorted(fixing[split], key=lambda r: r["id"]))

    # 7. statistics
    stats["dropped"] = dict(stats["dropped"])
    stats["fixing_dropped"] = dict(fix_dropped)
    stats["label_vocabulary"] = len(vocab)
    stats["detection"], stats["fixing"] = {}, {}
    for split in ("train", "val", "test"):
        rows = detection[split]
        n_smelly = sum(bool(r["checks"]) for r in rows)
        stats["detection"][split] = {"rows": len(rows), "smelly": n_smelly, "clean": len(rows) - n_smelly,
                                     "engines": sorted({r["engine"] for r in rows})}
        stats["fixing"][split] = {"rows": len(fixing[split])}
    label_rows = Counter(c for rows in detection.values() for r in rows for c in r["checks"])
    stats["rows_per_check"] = dict(label_rows.most_common())
    stats["fix_pairs_per_check"] = dict(Counter(c for rows in fixing.values() for r in rows
                                                for c in r["fixed_checks"]).most_common())
    if all_lengths:
        stats["tokens"] = {"p50": int(np.percentile(all_lengths, 50)), "p90": int(np.percentile(all_lengths, 90)),
                           "max": int(max(all_lengths))}
        stats["length_bin_edges"] = edges
        stats["length_rule"] = length_rule(detection)
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return out, stats


def length_rule(detection):
    """How well 'long means smelly' does: the shortcut any trained model has to beat."""
    train = detection["train"]
    if not train:
        return {}
    x = np.array([r["n_tokens"] for r in train])
    y = np.array([bool(r["checks"]) for r in train])
    cands = np.unique(x)
    best = int(cands[int(np.argmax([((x >= t) == y).mean() for t in cands]))])
    result = {"threshold_tokens": best}
    for split, rows in detection.items():
        if rows:
            px = np.array([r["n_tokens"] for r in rows]) >= best
            py = np.array([bool(r["checks"]) for r in rows])
            result[f"{split}_accuracy"] = round(float((px == py).mean()), 3)
            result[f"{split}_majority_class"] = round(float(max(py.mean(), 1 - py.mean())), 3)
    return result


def verify(out, cfg, max_tokens, eligibility):
    """Re-read the written files and check them independently of the build."""
    tok = tokenizer()
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' -- ' + str(detail) if detail != '' else ''}")

    det = {s: list(read_jsonl(out / "detection" / f"{s}.jsonl")) for s in ("train", "val", "test")}
    fix = {s: list(read_jsonl(out / "fixing" / f"{s}.jsonl")) for s in ("train", "val", "test")}
    vocab = set(json.loads((out / "detection" / "labels.json").read_text(encoding="utf-8"))["checks"])
    rows = [r for part in det.values() for r in part]
    check("detection has rows in every split", all(det.values()), {s: len(v) for s, v in det.items()})

    bad = 0
    for r in rows:
        a = json.loads(r["messages"][2]["content"])
        bad += not (list(a) == ["checks"] and a["checks"] == r["checks"] == sorted(set(r["checks"]))
                    and set(r["checks"]) <= vocab and [m["role"] for m in r["messages"]] == ["system", "user", "assistant"])
    check("every detection answer is valid JSON with sorted, known checks", bad == 0, f"{bad} bad")
    check("one detection prompt throughout", len({r["messages"][0]["content"] for r in rows}) <= 1)

    prefix = DETECT_USER.format(code="")
    codes = [norm_code(r["messages"][1]["content"][len(prefix):]) for r in rows]
    check("no function appears twice", len(set(codes)) == len(rows), f"{len(rows) - len(set(codes))} duplicates")
    check("ids are unique", len({r["id"] for r in rows}) == len(rows))
    check("no chat-template markers inside the code",
          not any(m in r["messages"][1]["content"] for r in rows for m in STRAY_MARKERS))

    engines = {s: {r["engine"] for r in part} for s, part in det.items()}
    overlap = (engines["train"] & engines["val"]) | (engines["train"] & engines["test"]) | (engines["val"] & engines["test"])
    check("no engine appears in two splits", not overlap, sorted(overlap) or "")
    split_of = {e: s for s, names in cfg["dataset"]["splits"].items() for e in names}
    check("every row is in the split its engine is assigned to",
          all(split_of.get(r["engine"], "train") == s for group in (det, fix) for s, part in group.items() for r in part))

    if eligibility["eval"] == "strict":
        check("val and test rows come only from files with no compile error",
              all(r["file_error_free"] for s in ("val", "test") for r in det[s]))
    findings_ok = all(1 <= f["line"] <= r["end_line"] - r["start_line"] + 1 for r in rows for f in r["findings"])
    check("every finding's line lies inside its function", findings_ok)
    labelled = all({f["check"] for f in r["findings"]} <= set(r["checks"]) for r in rows)
    check("every finding belongs to one of the row's checks", labelled)

    sample = random.Random(0).sample(rows, min(400, len(rows)))
    check("stored token counts match a fresh count",
          all(count_tokens(tok, r["messages"]) == r["n_tokens"] for r in sample))
    longest = max([r["n_tokens"] for r in rows] + [r["n_tokens"] for part in fix.values() for r in part] + [0])
    check(f"every conversation fits in {max_tokens} tokens", longest <= max_tokens, f"max {longest}")

    for name in ("val", "test"):
        sub = list(read_jsonl(out / "detection" / f"{name}_length_matched.jsonl"))
        ids = {r["id"] for r in det[name]}
        check(f"{name}_length_matched is a balanced subset of {name}",
              all(r["id"] in ids for r in sub) and sum(bool(r["checks"]) for r in sub) * 2 == len(sub), f"{len(sub)} rows")

    fix_rows = [r for part in fix.values() for r in part]
    changed = all(norm_code(r["messages"][1]["content"].split("\n\n", 1)[1]) != norm_code(r["messages"][2]["content"])
                  for r in fix_rows)
    check("every fixing target differs from its input", changed, f"{len(fix_rows)} pairs")
    check("every fixing row names what was fixed", all(r["fixed_checks"] for r in fix_rows))
    return all(results)


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engines", nargs="*")
    ap.add_argument("--config")
    ap.add_argument("--out", help="output folder relative to the repository (default: dataset_dir in the config)")
    ap.add_argument("--eligibility", choices=["strict", "relaxed"],
                    help="override the config for every split. strict: only functions from files with no compile "
                         "error at all; relaxed: also functions from files that have errors elsewhere")
    ap.add_argument("--max-tokens", type=int, default=4096, help="longest conversation allowed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out, stats = build(cfg, selected_engines(cfg, args.engines), args)
    print(json.dumps({k: v for k, v in stats.items() if k != "rows_per_check"}, indent=1))
    print("rows per check (top 25):", dict(list(stats.get("rows_per_check", {}).items())[:25]))
    print(f"\nwrote {out}\nverifying:")
    if not verify(out, cfg, args.max_tokens, stats["eligibility"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
