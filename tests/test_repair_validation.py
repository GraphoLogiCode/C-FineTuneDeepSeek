"""Validation must reject warning disappearance caused by errors or missing context."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from src.evaluation.repair_validation import (
    ROOT, RepairValidator, ValidationUnavailable, WITNESS, original_code, splice_function,
)

CHECK = "readability-else-after-return"
BEFORE = "int f(bool flag)\n{\n    if (flag) {\n        return 1;\n    } else {\n        return 2;\n    }\n}"
AFTER = "int f(bool flag)\n{\n    if (flag) {\n        return 1;\n    }\n    return 2;\n}"


def row_for(code=BEFORE, start=1):
    return {"id": "fixture", "engine": "fixture", "path": "fixture.cpp", "name": "f",
            "start_line": start, "end_line": start + code.count("\n"), "fixed_checks": [CHECK],
            "messages": [{"role": "user", "content": "Fix these clang-tidy findings: " + CHECK + "\n\n" + code},
                         {"role": "assistant", "content": AFTER}]}


def diag(check, offset=4, message="warning"):
    return {"check": check, "offset": offset, "path": "fixture.cpp", "message": message}


def analysis(*diags, errors=(), ok=True):
    return {"ok": ok, "reason": "compiler errors" if not ok else "", "errors": list(errors),
            "diagnostics": list(diags), "returncode": 0 if ok else 1}


class SpliceTests(unittest.TestCase):
    def test_splice_preserves_crlf_indentation_unicode_and_neighboring_code(self):
        before = BEFORE.replace("return 1", "return 1 /* caf\u00e9 */")
        after = AFTER.replace("return 1", "return 1 /* caf\u00e9 */")
        # The dataset dedents later lines but the AST starts after first-line indentation.
        raw = ("// neighbour\r\n  " + before.replace("\n", "\r\n  ") + "\r\n// untouched\r\n").encode()
        patched, start, end, new_end = splice_function(raw, row_for(before, 2), before, after)
        self.assertEqual(patched[:start], raw[:start])
        self.assertEqual(patched[new_end:], raw[end:])
        self.assertIn("caf\u00e9".encode(), patched)
        self.assertNotIn(b"\n", patched.replace(b"\r\n", b""))

    def test_stale_text_or_lines_cannot_be_overwritten(self):
        for row, before in ((row_for(), BEFORE.replace("return 1", "return 3")), (row_for(start=2), BEFORE)):
            with self.assertRaises(ValidationUnavailable):
                splice_function(BEFORE.encode(), row, before, AFTER)

    def test_candidate_cannot_remove_function_or_inject_another_definition(self):
        for candidate in ("", AFTER.replace("f(", "g("), AFTER + "\nint g() { return 0; }"):
            with self.assertRaises(ValueError):
                splice_function(BEFORE.encode(), row_for(), BEFORE, candidate)

    def test_suppressing_diagnostic_is_not_a_repair(self):
        for candidate in (AFTER.replace("return 2;", "return 2; // NOLINT"),
                          AFTER.replace("return 2;", "#pragma clang diagnostic ignored \"-Weverything\"\nreturn 2;")):
            with self.assertRaises(ValueError):
                splice_function(BEFORE.encode(), row_for(), BEFORE, candidate)

    def test_user_input_is_read_without_reference_answer(self):
        row = row_for()
        row["messages"][-1]["content"] = "THIS MUST NEVER BECOME THE INPUT"
        self.assertEqual(original_code(row), BEFORE)

    def test_defaulted_assignment_definition_keeps_exact_original_signature(self):
        before = "Thing& operator=(const Thing& other) {\n    value = other.value;\n    return *this;\n}"
        after = "Thing& operator=(const Thing& other) = default;"
        patched, _, _, _ = splice_function(before.encode(), row_for(before), before, after)
        self.assertEqual(patched, after.encode())
        with self.assertRaises(ValueError):
            splice_function(before.encode(), row_for(before), before, after.replace("Thing& operator", "Other& operator"))


class StaticGateTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / "data/repair_validation_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "fixture.cpp"
        self.source.write_bytes(BEFORE.encode())
        self.validator = RepairValidator()
        self.validator._context = Mock(return_value=(
            self.source, BEFORE.encode(), Path(self.temp.name), self.source,
            {"source_sha1": hashlib.sha1(BEFORE.encode()).hexdigest(), "scan_config_hash": "fixture"},
        ))
        self.good_before = analysis(diag(WITNESS), diag(CHECK, BEFORE.index("else")))
        self.good_after = analysis(diag(WITNESS))

    def run_result(self, before=None, after=None, candidate=AFTER):
        self.validator._analyse = Mock(side_effect=[before or self.good_before, after or self.good_after])
        return self.validator.validate(row_for(), candidate)

    def test_valid_static_result_explicitly_does_not_claim_behavior_correctness(self):
        result = self.run_result()
        self.assertEqual(result["status"], "passed_static_validation")
        self.assertTrue(result["static_validated"])
        self.assertEqual(result["behavioral_correctness"], "not_tested")
        self.assertEqual(result["engine_build"], "not_tested")

    def test_no_op_fails_and_reuses_original_analysis(self):
        result = self.run_result(candidate=BEFORE)
        self.assertEqual(result["status"], "rejected")
        self.assertIn(CHECK, result["remaining_checks"])
        self.assertEqual(self.validator._analyse.call_count, 1)

    def test_original_compile_error_is_inconclusive_never_a_clean_baseline(self):
        result = self.run_result(before=analysis(diag(WITNESS), errors=[{"message": "unknown type"}], ok=False))
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["static_validated"])
        self.assertEqual(self.validator._analyse.call_count, 1)

    def test_original_missing_target_or_analysis_witness_is_inconclusive(self):
        for before in (analysis(diag(WITNESS)), analysis(diag(CHECK))):
            with self.subTest(before=before):
                self.validator._analyses.clear()
                self.assertEqual(self.run_result(before=before)["status"], "inconclusive")

    def test_candidate_compiler_error_cannot_count_as_diagnostic_removed(self):
        result = self.run_result(after=analysis(errors=[{"message": "undeclared identifier"}], ok=False))
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["static_validated"])

    def test_candidate_tool_failure_or_absent_witness_is_inconclusive(self):
        for after in (analysis(ok=False), analysis()):
            with self.subTest(after=after):
                self.validator._analyses.clear()
                self.assertEqual(self.run_result(after=after)["status"], "inconclusive")

    def test_new_non_target_warning_is_rejected(self):
        result = self.run_result(after=analysis(diag(WITNESS), diag("bugprone-narrowing-conversions")))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["new_diagnostics"][0]["check"], "bugprone-narrowing-conversions")

    def test_existing_warning_after_function_tracks_byte_shift(self):
        old_offset = len(BEFORE.encode()) + 30
        new_offset = old_offset + len(AFTER.encode()) - len(BEFORE.encode())
        before = analysis(diag(WITNESS), diag(CHECK, BEFORE.index("else")), diag("misc-unused-parameters", old_offset))
        after = analysis(diag(WITNESS), diag("misc-unused-parameters", new_offset))
        self.assertTrue(self.run_result(before=before, after=after)["static_validated"])

    def test_new_warning_outside_repaired_function_is_rejected(self):
        after = analysis(diag(WITNESS), diag(CHECK, len(AFTER.encode()) + 15))
        result = self.run_result(after=after)
        self.assertEqual(result["remaining_checks"], [])
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["new_diagnostics"])

    def test_identical_candidate_is_cached(self):
        self.validator._analyse = Mock(side_effect=[self.good_before, self.good_after])
        self.assertTrue(self.validator.validate(row_for(), AFTER)["static_validated"])
        self.assertTrue(self.validator.validate(row_for(), AFTER)["static_validated"])
        self.assertEqual(self.validator._analyse.call_count, 2)

    def test_compiled_defaulted_assignment_needs_no_body_size_witness(self):
        before = "Thing& operator=(const Thing& other) {\n    value = other.value;\n    return *this;\n}"
        after = "Thing& operator=(const Thing& other) = default;"
        raw = ("struct Thing {\nint value;\n" + before + "\n};\n").encode()
        self.source.write_bytes(raw)
        context = list(self.validator._context.return_value)
        context[1] = raw
        self.validator._context.return_value = tuple(context)
        row = row_for(before, 3)
        row["fixed_checks"] = ["modernize-use-equals-default"]
        offset = raw.index(b"operator")
        self.validator._analyse = Mock(side_effect=[
            analysis(diag(WITNESS, offset), diag("modernize-use-equals-default", offset)), analysis(),
        ])
        result = self.validator.validate(row, after)
        self.assertTrue(result["static_validated"], result)
        self.assertEqual(result["candidate_analysis_witness"], "compiled defaulted definition")

    def test_parser_or_stale_provenance_failure_is_inconclusive(self):
        self.validator._context.side_effect = ValidationUnavailable("source SHA1 is stale")
        result = self.validator.validate(row_for(), AFTER)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["static_validated"])


class RealClangTidyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = RepairValidator._find_tool()
        if not cls.tool:
            raise unittest.SkipTest("clang-tidy not installed")

    def test_full_source_overlay_resolves_sibling_include_and_never_changes_source(self):
        import yaml
        temp_root = ROOT / "data/repair_validation_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            root = Path(temp)
            engine = root / "engines/fixture"
            engine.mkdir(parents=True)
            source = engine / "fixture.cpp"
            before = BEFORE.replace("return 1", "return helper()")
            after = AFTER.replace("return 1", "return helper()")
            raw = ('#include "sibling.h"\n' + before + "\n").encode()
            source.write_bytes(raw)
            (engine / "sibling.h").write_text("inline int helper() { return 1; }\n", encoding="utf-8")
            work = root / "work/fixture"
            flags_dir = work / "flags/c++17"
            flags_dir.mkdir(parents=True)
            flags = ["-xc++", "-std=c++17", "-fsyntax-only"]
            (flags_dir / "compile_flags.txt").write_text("\n".join(flags) + "\n", encoding="utf-8")
            tool_config = {"Checks": "-*," + CHECK + "," + WITNESS, "WarningsAsErrors": "",
                           "CheckOptions": {WITNESS + ".LineThreshold": "0"}}
            (work / "clang-tidy.yaml").write_text(yaml.safe_dump(tool_config), encoding="utf-8")
            version = subprocess.run([self.tool, "--version"], capture_output=True, text=True, check=True).stdout
            import re
            version = re.search(r"version (\S+)", version).group(1)
            cfg_hash = hashlib.sha1(json.dumps([version, tool_config, flags], sort_keys=True).encode()).hexdigest()[:12]
            (work / "scan_meta.json").write_text(json.dumps({"std": "c++17", "cfg_hash": cfg_hash,
                                                            "clang_tidy": version}), encoding="utf-8")
            scan = {"path": "fixture.cpp", "cfg": cfg_hash, "sha1": hashlib.sha1(raw).hexdigest(), "status": "ok"}
            (work / "scan.jsonl").write_text(json.dumps(scan) + "\n", encoding="utf-8")
            config = {"engines_root": str(root / "engines"), "engines": {"fixture": {}},
                      "work_dir": str(root / "work"), "dataset_dir": str(root / "dataset")}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            validator = RepairValidator(config_path, self.tool, timeout=30)
            row = row_for(before, 2)
            valid = validator.validate(row, after)
            self.assertTrue(valid["static_validated"], json.dumps(valid, indent=2))
            no_op = validator.validate(row, before)
            self.assertEqual(no_op["status"], "rejected")
            broken = validator.validate(row, after.replace("return 2;", "return unknown_variable;"))
            self.assertEqual(broken["status"], "rejected")
            self.assertTrue(broken["after"]["errors"])
            self.assertEqual(source.read_bytes(), raw)
            # A changed main file may not reuse cached baseline success.
            source.write_bytes(raw + b"// changed since collection\n")
            stale = validator.validate(row, after)
            self.assertEqual(stale["status"], "inconclusive")
            self.assertIn("SHA1", stale["reasons"][0])
            source.write_bytes(raw)
            # A cached compile-flags file may not silently drift either.
            (flags_dir / "compile_flags.txt").write_text("\n".join(flags + ["-DCHANGED=1"]) + "\n", encoding="utf-8")
            changed_flags = validator.validate(row, after)
            self.assertEqual(changed_flags["status"], "inconclusive")
            self.assertIn("configuration differ", changed_flags["reasons"][0])


if __name__ == "__main__":
    unittest.main()
