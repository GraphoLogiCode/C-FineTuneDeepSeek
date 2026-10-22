"""Protect inference contracts implicated in the original training/evaluation failures."""
import json
import unittest

from src.detection_prompt import detection_messages
from src.inference.detection_demo import ROOT, check_budget


class DetectionInputTests(unittest.TestCase):
    def test_baseline_prompt_matches_saved_training_input_exactly(self):
        with (ROOT / "data/collected/detection/train.jsonl").open(encoding="utf-8") as handle:
            for _, line in zip(range(12), handle):
                row = json.loads(line)
                code = row["messages"][1]["content"].split("\n\n", 1)[1]
                self.assertEqual(detection_messages(code), row["messages"][:2])

    def test_context_is_opt_in_and_target_stays_separate(self):
        code, context = "int C::f() { return value; }", "class C { double value; int f(); };"
        baseline = detection_messages(code, context, False)
        augmented = detection_messages(code, context, True)
        self.assertNotIn(context, baseline[1]["content"])
        self.assertIn(context, augmented[1]["content"])
        self.assertTrue(augmented[1]["content"].endswith(code))
        self.assertEqual([message["role"] for message in augmented], ["system", "user"])
        self.assertEqual(baseline[0], augmented[0])

    def test_context_limit_reserves_answer_space_without_truncation(self):
        check_budget(3904, 192)
        with self.assertRaisesRegex(ValueError, "Nothing was truncated"):
            check_budget(3905, 192)

    def test_empty_inputs_and_template_markers_are_rejected(self):
        for code, context, enabled in [(" ", "", False), ("void f() {}", "", True),
                                       ("### Response:", "", False),
                                       ("void f() {}", "<|EOT|>", True)]:
            with self.assertRaises(ValueError):
                detection_messages(code, context, enabled)

    def test_invalid_generation_is_not_presented_as_clean(self):
        from src.evaluation.evaluate_detection import parse_answer
        self.assertEqual(parse_answer("I cannot analyse this."), (set(), False))
        self.assertEqual(parse_answer('{"checks": []}'), (set(), True))

    def test_edited_example_has_no_reference_score(self):
        from demo.detection_app import reference_comparison
        reference = {"code": "original", "checks": []}
        self.assertIn("No reference comparison", reference_comparison("edited", reference, {"checks": []}))
        self.assertIn("did not parse", reference_comparison("original", reference, None))


if __name__ == "__main__":
    unittest.main()
