"""Independent executable checks for the repair demonstration's behavior claims."""
import unittest

from src.evaluation.repair_behavior import CASES, find_compiler, validate_behavior


@unittest.skipUnless(find_compiler(), "A C++ compiler is required for behavioral fixtures")
class CompiledBehaviorTests(unittest.TestCase):
    def test_originals_and_references_preserve_explicit_expected_values(self):
        for case in CASES:
            for version in ("before", "reference"):
                with self.subTest(case=case["name"], version=version):
                    result = validate_behavior(case, case[version])
                    self.assertTrue(result["compile_ok"], result)
                    self.assertTrue(result["tests_passed"], result)

    def test_compilable_behavior_changes_are_detected(self):
        for case in CASES:
            with self.subTest(case=case["name"]):
                result = validate_behavior(case, case["bad"])
                self.assertTrue(result["compile_ok"], result)
                self.assertFalse(result["tests_passed"], result)
                self.assertEqual(result["status"], "assertion_failed")

    def test_compilation_failure_cannot_pass_behavior_validation(self):
        result = validate_behavior(CASES[0], "int classify(int value) { return missing_name; }")
        self.assertFalse(result["compile_ok"], result)
        self.assertFalse(result["tests_passed"], result)
        self.assertEqual(result["status"], "compile_failed")


if __name__ == "__main__":
    unittest.main()
