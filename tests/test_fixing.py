import json
import tempfile
import unittest
from pathlib import Path

from src.fixing_prompt import fixing_messages
from src.inference.fixing import ROOT, Fixer, candidate_code, syntax_check


class RepairContracts(unittest.TestCase):
    def test_prompt_matches_collected_training_without_reference_answer(self):
        with (ROOT / "data/collected/fixing/train.jsonl").open(encoding="utf-8") as handle:
            for _, line in zip(range(30), handle):
                row = json.loads(line)
                original = row["messages"][1]["content"].split("\n\n", 1)[1]
                self.assertEqual(fixing_messages(original, row["fixed_checks"]), row["messages"][:2])

    def test_rejects_empty_request_and_chat_boundaries(self):
        for code, checks in [("", ["modernize-use-nullptr"]), ("void f() {}", []),
                             ("### Response:", ["modernize-use-nullptr"])]:
            with self.assertRaises(ValueError):
                fixing_messages(code, checks)

    def test_preserves_plain_and_fenced_code_without_rewriting_it(self):
        code = 'const char* f() { return "a  b"; }'
        self.assertEqual(candidate_code(code), code)
        self.assertEqual(candidate_code("```cpp\n" + code + "\n```"), code)
        self.assertEqual(candidate_code("Here is the fix:\n```cpp\n" + code + "\n```\nExplanation."), code)
        multiple = "```cpp\nint f() { return 1; }\n```\n```cpp\nint g() { return 2; }\n```"
        self.assertEqual(candidate_code(multiple), multiple)

    def test_syntax_gate_distinguishes_prose_empty_and_multiple_functions(self):
        self.assertTrue(syntax_check("int f() { return 1; }")["single_function"])
        self.assertFalse(syntax_check("")["single_function"])
        self.assertFalse(syntax_check("int f() { return 1; } int g() { return 2; }")["single_function"])
        self.assertFalse(syntax_check("int f( {")["parse_ok"])

    def test_unsupported_check_abstains_before_loading_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as folder:
            Path(folder, "trained_checks.json").write_text(json.dumps({"checks": ["readability-else-after-return"]}))
            fixer = Fixer(folder)
            with self.assertRaisesRegex(ValueError, "No repair training support"):
                fixer.repair("int f() { return 1; }", ["readability-function-size"])
            self.assertIsNone(fixer.model)


if __name__ == "__main__":
    unittest.main()
