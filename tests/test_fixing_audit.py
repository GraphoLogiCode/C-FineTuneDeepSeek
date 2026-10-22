"""Protect sampling, split boundaries, and exact prompt preservation."""
import copy
import unittest
from collections import Counter

from scripts.audit.audit_fixing_dataset import (
    FIX_SYSTEM, code_pair, leakage_summary, prepare_pilot, stratified_sample, token_count_comparison,
)


def row(ident, check="readability-else-after-return", engine="train-engine", strict=True, tokens=200):
    before = f"int f_{ident}(int x) {{ if (x) return 1; else return 0; }}"
    after = f"int f_{ident}(int x) {{ if (x) return 1; return 0; }}"
    return {
        "id": ident, "engine": engine, "fixed_checks": [check],
        "file_error_free": strict, "n_tokens": tokens,
        "messages": [
            {"role": "system", "content": FIX_SYSTEM},
            {"role": "user", "content": f"Fix these clang-tidy findings: {check}\n\n{before}"},
            {"role": "assistant", "content": after},
        ],
    }


class FixingAuditTests(unittest.TestCase):
    def test_input_output_are_extracted_without_reformatting(self):
        item = row("one")
        before, after = code_pair(item)
        self.assertEqual(before, item["messages"][1]["content"].split("\n\n", 1)[1])
        self.assertEqual(after, item["messages"][2]["content"])
        item["messages"][1]["content"] = "an unexpected prompt\n\n" + before
        with self.assertRaisesRegex(ValueError, "collector template"):
            code_pair(item)

    def test_stratification_is_stable_under_input_reordering(self):
        rows = [row(str(i), check="rare" if i < 3 else "common", engine=f"e{i % 2}") for i in range(30)]
        counts = Counter(c for item in rows for c in item["fixed_checks"])
        first = stratified_sample(rows, 12, counts, 42, "train")
        second = stratified_sample(list(reversed(rows)), 12, counts, 42, "train")
        self.assertEqual(first, second)
        self.assertEqual(len({item["id"] for item in first}), 12)
        self.assertEqual(sum(item["fixed_checks"] == ["rare"] for item in first), 3)
        self.assertEqual({item["engine"] for item in first}, {"e0", "e1"})

    def test_filters_are_applied_before_sampling_and_prompts_are_unchanged(self):
        parts = {
            "train": [row("keep"), row("long", tokens=2049), row("bad", strict=False)],
            "val": [row("v", engine="heldout-val"), row("unseen", check="unknown", engine="heldout-val")],
            "test": [row("t", engine="heldout-test")],
        }
        original = copy.deepcopy(parts)
        selected, design = prepare_pilot(parts, {"train": 20, "val": 20, "test": 20}, 2048, 7)
        self.assertEqual(parts, original)
        self.assertEqual([r["id"] for r in selected["train"]], ["keep"])
        self.assertEqual(selected["train"][0], original["train"][0])
        self.assertEqual(design["excluded_eval_rows_with_checks_absent_from_train_candidates"], {"val": 1, "test": 0})
        self.assertEqual(design["candidate_sizes"], {"train": 1, "val": 1, "test": 1})

    def test_sampling_does_not_inspect_targets_or_test_frequencies(self):
        rows = [row(str(i)) for i in range(20)]
        counts = {"readability-else-after-return": 20}
        first = stratified_sample(rows, 5, counts, 8, "test")
        altered = copy.deepcopy(rows)
        for item in altered:
            item["messages"][2]["content"] = "arbitrarily incorrect generated target"
        second = stratified_sample(altered, 5, counts, 8, "test")
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])

    def test_overlap_reports_copy_even_with_whitespace_changes(self):
        train = row("train")
        val = copy.deepcopy(train)
        val["id"] = "val"
        val["engine"] = "heldout-val"
        val["messages"][1]["content"] = val["messages"][1]["content"].replace(" { ", " {\n ")
        result = leakage_summary({"train": [train], "val": [val], "test": [row("test", engine="heldout-test")]})
        self.assertEqual(result["train:val"]["shared_normalized_inputs"], 1)
        self.assertEqual(result["train:val"]["shared_engines"], [])
        self.assertEqual(result["train:val"]["shared_ids"], 0)

    def test_runtime_budget_counts_actual_tokens_without_changing_frozen_rows(self):
        class FakeTokenizer:
            eos_token = "<EOS>"

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text)}

        item = row("runtime", tokens=1)
        original = copy.deepcopy(item)
        result = token_count_comparison([item], FakeTokenizer(), max_tokens=5)
        self.assertEqual(result["stored_count_mismatches"], 1)
        self.assertEqual(result["excluded_ids"], ["runtime"])
        self.assertEqual(result["retained_rows_at_limit"], 0)
        self.assertEqual(item, original)


if __name__ == "__main__":
    unittest.main()
