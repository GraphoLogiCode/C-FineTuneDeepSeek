"""Native tests for one preselected held-out Spring engine component.

The complete original AIInterfaceKey.cpp is copied to a temporary workspace;
only the selected function is replaced. Compilation uses actual engine headers,
without collection stubs, in the engine-supported BUILDING_AI configuration.
This is a finite native component test, not an engine build or equivalence proof.

    python -m src.evaluation.repair_engine_component
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from src.evaluation.repair_validation import ROOT, RepairValidator, original_code, splice_function

ROW_ID = "fix-31c269426bd1"
CLANG = Path("C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin/clang++.exe")
ASSERTIONS = 31468
BAD_CANDIDATE = "bool AIInterfaceKey::isEqual(const AIInterfaceKey& otherKey) const { return true; }"
PLAUSIBLE_BAD_CANDIDATE = (
    "bool AIInterfaceKey::isEqual(const AIInterfaceKey& otherKey) const { "
    "return shortName == otherKey.shortName; }")

# These expectations are hand-written from public key identity/lexicographic
# ordering requirements. Expected results do not execute the original function.
HARNESS = r'''#include "AIInterfaceKey.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static unsigned long assertions = 0;
static void require(bool condition, const char* operation, std::size_t i, std::size_t j) {
    ++assertions;
    if (!condition) {
        std::cerr << "FAILED " << operation << " indices=" << i << "," << j
                  << " assertion=" << assertions << "\n";
        std::exit(7);
    }
}

int main() {
    const std::vector<std::string> names = {
        "", "A", "AI", "ai", "B", "name with spaces",
        std::string("\0", 1), std::string("A\0B", 3), std::string(512, 'x')
    };
    const std::vector<std::string> versions = {
        "", "0", "1", "01", "1.0", "2",
        std::string("\0", 1), std::string("v\0x", 3)
    };
    std::vector<std::pair<std::string, std::string>> expected;
    std::vector<AIInterfaceKey> keys;
    for (const auto& name : names) {
        for (const auto& version : versions) {
            expected.emplace_back(name, version);
            keys.emplace_back(name, version);
        }
    }
    AIInterfaceKey empty;
    require(empty.GetShortName().empty(), "default name", 0, 0);
    require(empty.GetVersion().empty(), "default version", 0, 0);
    require(empty.IsUnspecified(), "default unspecified", 0, 0);
    require(empty == AIInterfaceKey("", ""), "default identity", 0, 0);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const auto& key = keys[i];
        require(key.GetShortName() == expected[i].first, "get name", i, i);
        require(key.GetVersion() == expected[i].second, "get version", i, i);
        require(key.IsUnspecified() == expected[i].first.empty(), "unspecified", i, i);
        require(key.ToString() == expected[i].first + " " + expected[i].second, "string", i, i);
        const AIInterfaceKey copied(key);
        require(copied == key, "copy identity", i, i);
        for (std::size_t j = 0; j < keys.size(); ++j) {
            const auto& other = keys[j];
            require((key == other) == (expected[i] == expected[j]), "==", i, j);
            require((key != other) == (expected[i] != expected[j]), "!=", i, j);
            require((key < other) == (expected[i] < expected[j]), "<", i, j);
            require((key > other) == (expected[i] > expected[j]), ">", i, j);
            require((key <= other) == (expected[i] <= expected[j]), "<=", i, j);
            require((key >= other) == (expected[i] >= expected[j]), ">=", i, j);
        }
    }
    std::cout << "PASS assertions=" << assertions << " keys=" << keys.size()
              << " pairs=" << keys.size() * keys.size() << "\n";
    return assertions == 31468 ? 0 : 9;
}
'''


def load_case(dataset, validator=None):
    rows = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row["id"] == ROW_ID]
    if len(matches) != 1:
        raise ValueError("The frozen component row is missing or duplicated")
    row = matches[0]
    if (row["engine"], row["path"], row["name"]) != (
            "spring", "rts/ExternalAI/AIInterfaceKey.cpp", "AIInterfaceKey::isEqual"):
        raise ValueError("Frozen row metadata differs from the selected component")
    validator = validator or RepairValidator()
    source, raw, _, _, provenance = validator._context(row)
    _, start, end, _ = splice_function(raw, row, original_code(row), original_code(row))
    engine = source.parents[2]
    header_paths = [source.with_suffix(".h"), engine / "rts/System/creg/creg_cond.h",
                    engine / "rts/System/creg/ISerializer.h"]
    headers = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in header_paths}
    return {"row": row, "source": source, "raw": raw, "engine_root": engine,
            "start": start, "end": end, "provenance": provenance, "header_sha256": headers}


def test_candidate(case, candidate, compiler=CLANG, timeout=45):
    started = time.perf_counter()
    result = {"compile_ok": False, "tests_passed": False, "assertion_count": ASSERTIONS,
              "evidence": "native_heldout_engine_component", "status": "unavailable",
              "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest()}
    try:
        patched, start, end, new_end = splice_function(case["raw"], case["row"],
                                                      original_code(case["row"]), candidate)
        if patched[:start] != case["raw"][:start] or patched[new_end:] != case["raw"][end:]:
            raise ValueError("Candidate changed surrounding component source")
        result.update(component_source_sha256=hashlib.sha256(patched).hexdigest(),
                      surrounding_source_unchanged=True)
        temp_root = ROOT / "results/collected_fixing/engine_component/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-component-", dir=temp_root) as directory:
            folder = Path(directory)
            source, harness, executable = folder / "AIInterfaceKey.cpp", folder / "harness.cpp", folder / "component.exe"
            source.write_bytes(patched)
            harness.write_text(HARNESS, encoding="utf-8", newline="\n")
            engine = case["engine_root"]
            flags = ["-std=c++17", "-O0", "-DBUILDING_AI", "-I" + str(engine / "rts"),
                     "-I" + str(engine / "rts/ExternalAI")]
            command = [str(compiler), *flags, str(source), str(harness), "-o", str(executable)]
            result["compile_flags"] = flags
            # Direct compiler intermediates to this writable temporary project.
            process_env = dict(os.environ, TMP=directory, TEMP=directory, TMPDIR=directory)
            built = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout,
                                   cwd=directory, env=process_env)
            result.update(compile_returncode=built.returncode, compile_stderr=built.stderr[-4000:],
                          compile_stdout=built.stdout[-4000:], compile_ok=built.returncode == 0)
            if built.returncode:
                result["status"] = "compile_failed"
                return result
            ran = subprocess.run([str(executable)], capture_output=True, text=True, errors="replace",
                                 timeout=5, cwd=directory)
            count = re.search(r"PASS assertions=(\d+) keys=(\d+) pairs=(\d+)", ran.stdout)
            passed = ran.returncode == 0 and count is not None and int(count.group(1)) == ASSERTIONS
            result.update(run_returncode=ran.returncode, run_stdout=ran.stdout[-4000:], run_stderr=ran.stderr[-4000:],
                          tests_passed=passed, status="passed_component_tests" if passed else "assertion_failed")
            if count:
                result.update(assertions_completed=int(count.group(1)), tested_keys=int(count.group(2)),
                              tested_ordered_pairs=int(count.group(3)))
            return result
    except subprocess.TimeoutExpired as exc:
        result.update(status="run_timeout" if result["compile_ok"] else "compile_timeout", timeout_seconds=exc.timeout)
        return result
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        result.update(status="unavailable", reason=str(exc))
        return result
    finally:
        unchanged = case["source"].read_bytes() == case["raw"]
        result["engine_source_unchanged"] = unchanged
        headers_same = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                           for path, digest in case["header_sha256"].items())
        result["engine_headers_unchanged"] = headers_same
        if not unchanged or not headers_same:
            result.update(status="source_changed", tests_passed=False)
        result["seconds"] = round(time.perf_counter() - started, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/fixing_pilot/test.jsonl")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/collected_fixing/pilot_test/predictions.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/collected_fixing/engine_component")
    parser.add_argument("--compiler", type=Path, default=CLANG)
    args = parser.parse_args()
    validator = RepairValidator()
    case = load_case(args.dataset, validator)
    row = case["row"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "harness.cpp").write_text(HARNESS, encoding="utf-8", newline="\n")
    candidates = {"original": original_code(row), "reference": row["messages"][-1]["content"],
                  "intentional_bad": BAD_CANDIDATE,
                  "plausible_bad_ignores_version": PLAUSIBLE_BAD_CANDIDATE}
    metadata = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "predictions_sha256": None, "model_predictions": "not_available"}
    if args.predictions.is_file():
        predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        metadata["predictions_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
        selected = [pred for pred in predictions if pred["id"] == ROW_ID and pred["system"] in {"adapter", "base"}]
        if len({pred["system"] for pred in selected}) != len(selected):
            raise ValueError("Duplicate model outputs for the frozen component row")
        for pred in selected:
            if pred["dataset_sha256"] != metadata["dataset_sha256"]:
                raise ValueError("Predictions do not belong to the frozen evaluation manifest")
            if pred.get("truncated"):
                raise ValueError("Frozen component model output was truncated; do not present it as a completed repair")
            candidates[pred["system"]] = pred["candidate"]
        metadata["model_predictions"] = sorted(pred["system"] for pred in selected)
    compiler_version = subprocess.run([str(args.compiler), "--version"], capture_output=True, text=True,
                                      timeout=15, check=True).stdout.strip()
    results = []
    for system, candidate in candidates.items():
        result = {"id": ROW_ID, "system": system, "engine": row["engine"], "path": row["path"],
                  "candidate": candidate, "requested_checks": row["fixed_checks"],
                  "validation": test_candidate(case, candidate, args.compiler)}
        if system == "plausible_bad_ignores_version":
            result["static_validation"] = validator.validate(row, candidate)
        results.append(result)
        print(f"{ROW_ID} / {system}: {result['validation']['status']}", flush=True)
    control_systems = {"original", "reference", "intentional_bad", "plausible_bad_ignores_version"}
    passing_controls = {"original", "reference"}
    controls_ok = all(result["validation"]["compile_ok"] and result["validation"]["engine_source_unchanged"]
                      and result["validation"]["engine_headers_unchanged"]
                      and result["validation"]["tests_passed"] == (result["system"] in passing_controls)
                      for result in results if result["system"] in control_systems)
    plausible = next(result for result in results if result["system"] == "plausible_bad_ignores_version")
    summary = {"evidence": "native_heldout_engine_component", "id": ROW_ID,
               "selection": "preselected GUI default equality example before model results; no outcome-based search",
               "component": "Spring AIInterfaceKey", "functions_repaired": 1,
               "keys": 72, "ordered_key_pairs": 5184, "assertions_per_passing_candidate": ASSERTIONS,
               "controls_ok": controls_ok, "compiler": str(args.compiler), "compiler_version": compiler_version,
               "configuration": "BUILDING_AI: real engine creg_cond.h disables serialization metadata",
               "uses_collection_header_stubs": False, "actual_engine_headers_sha256": case["header_sha256"],
               "source_provenance": case["provenance"], "generation": metadata,
               "harness_sha256": hashlib.sha256(HARNESS.encode()).hexdigest(),
               "systems": {result["system"]: result["validation"] for result in results},
               "negative_control_static_validation": {"plausible_bad_ignores_version": plausible["static_validation"]},
               "negative_control_findings": {"plausible_bad_ignores_version": {
                   "intentional_fault": "compare short name but ignore version",
                   "static_validated": plausible["static_validation"]["static_validated"],
                   "native_tests_passed": plausible["validation"]["tests_passed"]}},
               "limitations": ["One preselected small held-out engine component, not a representative benchmark.",
                               "Finite identity/ordering and boundary assertions do not establish general semantic equivalence.",
                               "This compiles the complete component source with real headers in BUILDING_AI mode; it is not an engine build.",
                               "Serialization metadata and runtime engine integration are not exercised.",
                               "Behavior tests and clang-tidy warning-removal checks establish separate properties."]}
    (args.out / "predictions.jsonl").write_text("".join(json.dumps(result) + "\n" for result in results), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"controls_ok": controls_ok, "systems": {key: value["status"] for key, value in summary["systems"].items()}}, indent=2))
    return 0 if controls_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
