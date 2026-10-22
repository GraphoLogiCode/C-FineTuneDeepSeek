"""Are the findings of the collection pipeline attached to the right line of the right function?

For checks whose trigger is a visible token, test whether that token is on the line the finding
points to. Reads data/collect/<engine>/units.jsonl.

    python scripts/audit/audit_collected_units.py            # every engine with units
    python scripts/audit/audit_collected_units.py irrlicht
"""
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "collect"

# check -> what must appear on the reported line
ON_LINE = {
    "readability-else-after-return": re.compile(r"\belse\b"),
    "modernize-use-nullptr": re.compile(r"\b(0|NULL|0L|0x0+)\b"),
    "cppcoreguidelines-avoid-goto": re.compile(r"\bgoto\b"),
    "cppcoreguidelines-avoid-do-while": re.compile(r"\bdo\b"),
    "modernize-avoid-c-style-cast": re.compile(r"\("),
    "readability-uppercase-literal-suffix": re.compile(r"\d[0-9a-fA-FxX.']*[fulFUL]"),
    "modernize-use-auto": re.compile(r"\bnew\b|_cast<|\biterator\b|\(\s*[\w:<>, ]+[*&\s]*\)\s*[\w(*&]"),
    "bugprone-switch-missing-default-case": re.compile(r"\bswitch\b"),
    "readability-braces-around-statements": re.compile(r"\b(if|else|for|while|do)\b|\)"),
}
# checks that quote the identifier they are about: 'name'
QUOTED = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'")
QUOTING = ("readability-identifier-length", "cppcoreguidelines-init-variables", "misc-unused-parameters",
           "readability-non-const-parameter", "misc-const-correctness")

engines = sys.argv[1:] or sorted(p.name for p in WORK.iterdir() if (p / "units.jsonl").exists())
total, hit = Counter(), Counter()
examples = defaultdict(list)
n_units = 0
for engine in engines:
    with open(WORK / engine / "units.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            n_units += 1
            lines = u["code"].replace("\r\n", "\n").split("\n")
            for fnd in u["findings"]:
                check = fnd["check"]
                if not 1 <= fnd["line"] <= len(lines):
                    total["<line outside the function>"] += 1
                    continue
                text = lines[fnd["line"] - 1]
                if check in ON_LINE:
                    ok = bool(ON_LINE[check].search(text))
                elif check in QUOTING:
                    m = QUOTED.search(fnd["message"])
                    if not m:
                        continue
                    ok = re.search(r"(?<![A-Za-z0-9_])" + re.escape(m.group(1)) + r"(?![A-Za-z0-9_])", text) is not None
                else:
                    continue
                total[check] += 1
                hit[check] += ok
                if not ok and len(examples[check]) < 2:
                    examples[check].append((engine, u["path"], u["start_line"] + fnd["line"] - 1, text.strip()[:80]))

print(f"engines: {', '.join(engines)}; functions: {n_units}")
print(f"{'check':46s} {'findings':>9s}  token on the reported line")
for check, n in total.most_common():
    if check.startswith("<"):
        print(f"{check:46s} {n:9d}")
    else:
        print(f"{check:46s} {n:9d}  {hit[check] / n:7.1%}")
tested = sum(n for c, n in total.items() if not c.startswith("<"))
print(f"\nall tested findings: {sum(hit.values())} of {tested} on the right line ({sum(hit.values()) / max(tested, 1):.2%})")
for check, ex in examples.items():
    for e in ex:
        print(f"  miss [{check}] {e[0]}/{e[1]}:{e[2]}  {e[3]!r}")
