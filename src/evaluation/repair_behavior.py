"""Compiled behavioral checks on illustrative fixtures, not held-out engine evidence."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CLANG = Path("C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin/clang++.exe")
LIMITS = "static_assert(sizeof(int) == 4 && sizeof(unsigned int) == 4, \"32-bit fixture integers required\");\n"


def _case(name, before, reference, check, harness, test_count, bad):
    return dict(name=name, before=before, reference=reference, checks=[check],
                harness=harness, test_count=test_count, bad=bad)


CASES = [
    _case("else_after_return", """int classify(int value) {
    if (value < 0) { return -1; }
    else { return value == 0 ? 0 : 1; }
}""", """int classify(int value) {
    if (value < 0) { return -1; }
    return value == 0 ? 0 : 1;
}""", "readability-else-after-return", """int main() {
    int inputs[] = {-2147483647 - 1, -999, -1, 0, 1, 999, 2147483647};
    int expected[] = {-1, -1, -1, 0, 1, 1, 1};
    for (int i = 0; i < 7; ++i) { if (classify(inputs[i]) != expected[i]) return 1; }
    return 0;
}""", 7, "int classify(int value) { return 0; }"),
    _case("braces", """int bounded_sum(int limit) {
    int total = 0;
    for (int i = 0; i < limit; ++i)
        total += i;
    return total;
}""", """int bounded_sum(int limit) {
    int total = 0;
    for (int i = 0; i < limit; ++i) {
        total += i;
    }
    return total;
}""", "readability-braces-around-statements", """int main() {
    int inputs[] = {-2147483647 - 1, -5, 0, 1, 2, 64, 1000};
    int expected[] = {0, 0, 0, 0, 1, 2016, 499500};
    for (int i = 0; i < 7; ++i) { if (bounded_sum(inputs[i]) != expected[i]) return 1; }
    return 0;
}""", 7, "int bounded_sum(int limit) { return 0; }"),
    _case("nullptr", """int pointer_value(const int* value) {
    if (value == 0) { return -17; }
    return *value;
}""", """int pointer_value(const int* value) {
    if (value == nullptr) { return -17; }
    return *value;
}""", "modernize-use-nullptr", """int main() {
    if (pointer_value(nullptr) != -17) return 1;
    int inputs[] = {-2147483647 - 1, -17, -1, 0, 1, 2147483647};
    for (int i = 0; i < 6; ++i) { if (pointer_value(&inputs[i]) != inputs[i]) return 1; }
    return 0;
}""", 7, "int pointer_value(const int* value) { return -17; }"),
    _case("boolean_simplification", """bool is_nonzero(int value) {
    if (value != 0) { return true; }
    else { return false; }
}""", """bool is_nonzero(int value) {
    return value != 0;
}""", "readability-simplify-boolean-expr", """int main() {
    int inputs[] = {-2147483647 - 1, -999, -1, 0, 1, 999, 2147483647};
    bool expected[] = {true, true, true, false, true, true, true};
    for (int i = 0; i < 7; ++i) { if (is_nonzero(inputs[i]) != expected[i]) return 1; }
    return 0;
}""", 7, "bool is_nonzero(int value) { return value > 0; }"),
    _case("c_style_cast", """unsigned int to_unsigned(int value) {
    return (unsigned int)value;
}""", """unsigned int to_unsigned(int value) {
    return static_cast<unsigned int>(value);
}""", "modernize-avoid-c-style-cast", """int main() {
    int inputs[] = {-2147483647 - 1, -2, -1, 0, 1, 2147483647};
    unsigned int expected[] = {2147483648u, 4294967294u, 4294967295u, 0u, 1u, 2147483647u};
    for (int i = 0; i < 6; ++i) { if (to_unsigned(inputs[i]) != expected[i]) return 1; }
    return 0;
}""", 6, "unsigned int to_unsigned(int value) { return 0u; }"),
    _case("redundant_void", """int sentinel(void) {
    return 37;
}""", """int sentinel() {
    return 37;
}""", "modernize-redundant-void-arg", """int main() {
    if (sentinel() != 37) return 1;
    if (sentinel() != 37) return 1;
    return 0;
}""", 2, "int sentinel() { return 0; }"),
]


def find_compiler():
    return str(CLANG) if CLANG.is_file() else shutil.which("clang++") or shutil.which("g++")


def validate_behavior(case, candidate, compiler=None):
    """Compile the exact candidate with the independent fixture and run its assertions."""
    compiler = compiler or find_compiler()
    result = {"compile_ok": False, "tests_passed": False, "test_count": case["test_count"],
              "compiler": compiler, "evidence": "illustrative_standalone_fixture"}
    if not compiler:
        return dict(result, status="compiler_unavailable")
    source = LIMITS + candidate + "\n" + case["harness"] + "\n"
    result["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    temp_root = ROOT / "data/repair_behavior_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=temp_root) as directory:
        source_path, executable = Path(directory) / "candidate.cpp", Path(directory) / "candidate.exe"
        source_path.write_text(source, encoding="utf-8")
        command = [str(compiler), "-std=c++17", "-O0", str(source_path), "-o", str(executable)]
        # Keep compiler intermediates inside the writable fixture directory too.
        process_env = dict(os.environ, TMP=directory, TEMP=directory, TMPDIR=directory)
        try:
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=45, errors="replace",
                                      cwd=directory, env=process_env)
            result.update(compile_returncode=compiled.returncode,
                          compile_stdout=compiled.stdout[-4000:], compile_stderr=compiled.stderr[-4000:])
            result["compile_ok"] = compiled.returncode == 0
            if not result["compile_ok"]:
                result["status"] = "compile_failed"
            else:
                executed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=3,
                                          errors="replace")
                result.update(run_returncode=executed.returncode, run_stdout=executed.stdout[-4000:],
                              run_stderr=executed.stderr[-4000:], tests_passed=executed.returncode == 0)
                result["status"] = "passed_fixture" if result["tests_passed"] else "assertion_failed"
        except subprocess.TimeoutExpired as error:
            result["status"] = "run_timeout" if result["compile_ok"] else "compile_timeout"
            result["timeout_seconds"] = error.timeout
        except OSError as error:
            result.update(status="execution_error", error=str(error))
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Optional learned adapter; otherwise validate fixture controls only.")
    parser.add_argument("--out", default="results/collected_fixing/behavior")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fixer = None
    if args.adapter:
        from src.inference.fixing import Fixer
        fixer = Fixer(ROOT / args.adapter)
    for case in CASES:
        candidates = {"original": case["before"], "reference": case["reference"], "intentional_bad": case["bad"]}
        inference = None
        if fixer:
            inference = fixer.repair(case["before"], case["checks"], max_new_tokens=768)
            candidates["adapter"] = inference["candidate"]
        for system, candidate in candidates.items():
            row = {"case": case["name"], "system": system, "checks": case["checks"],
                   "candidate": candidate, "validation": validate_behavior(case, candidate)}
            if system == "adapter":
                row["inference"] = inference
            rows.append(row)
            print(f"{case['name']} / {system}: {row['validation']['status']}", flush=True)
    controls_ok = all(r["validation"]["compile_ok"] and
                      r["validation"]["tests_passed"] == (r["system"] != "intentional_bad")
                      for r in rows if r["system"] != "adapter")
    summary = {"fixture_count": len(CASES), "assertions_per_system": sum(c["test_count"] for c in CASES),
               "controls_ok": controls_ok, "adapter": args.adapter,
               "systems": {s: {"cases": sum(r["system"] == s for r in rows),
                                "compiled": sum(r["system"] == s and r["validation"]["compile_ok"] for r in rows),
                                "passed": sum(r["system"] == s and r["validation"]["tests_passed"] for r in rows)}
                           for s in sorted({r["system"] for r in rows})},
               "limitations": ["Hand-written illustrative cases, not held-out engine functions.",
                               "Passing finite fixture inputs is not a proof of semantic equivalence.",
                               "This suite checks behavior, not clang-tidy warning removal."]}
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if controls_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
