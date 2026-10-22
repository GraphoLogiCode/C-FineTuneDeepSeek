"""Build a CPU-only evidence report from completed repair artifacts; mark pending work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"adapter": "Trained 1.3B adapter", "base": "Untuned 1.3B base",
          "no_op": "Return input unchanged", "clang_tidy_recorded_edits": "Recorded clang-tidy targets",
          "fresh_clang_tidy": "Fresh clang-tidy (function scope)"}


class Artifacts:
    def __init__(self):
        self.hashes, self.notes = {}, []

    def read(self, path, lines=False):
        file = ROOT / path
        if not file.exists():
            self.notes.append(f"Pending: `{path}`.")
            return [] if lines else None
        raw = file.read_bytes()
        self.hashes[path] = hashlib.sha256(raw).hexdigest()
        try:
            if lines:
                return [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            return json.loads(raw.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.notes.append(f"Pending or incomplete write: `{path}`; excluded from calculations.")
            return [] if lines else None

    def hash(self, path):
        file = ROOT / path
        if file.exists():
            self.hashes[path] = hashlib.sha256(file.read_bytes()).hexdigest()


def link(path, label=None):
    return f"[{label or path}]({Path('..', path).as_posix()})"


def fraction(k, n):
    return f"{k}/{n} ({100*k/n:.1f}%)" if n else "not available"


def passed(row):
    return bool(row.get("validation", {}).get("static_validated"))


def paired_bootstrap(left, right, samples=10000, seed=20261015):
    """Percentile interval for paired binary outcomes, resampling functions."""
    if not left or len(left) != len(right):
        raise ValueError("Paired nonempty outcomes must have equal lengths")
    differences = [int(a)-int(b) for a, b in zip(left, right)]
    n, rng = len(differences), random.Random(seed)
    estimates = sorted(sum(rng.choices(differences, k=n))/n for _ in range(samples))
    return sum(differences)/n, estimates[int(.025*samples)], estimates[min(samples-1, int(.975*samples))]


def complete(records, ids):
    counts = Counter(r["id"] for r in records)
    return bool(ids) and set(counts) == set(ids) and all(n == 1 for n in counts.values()) and all(
        isinstance(r.get("validation", {}).get("static_validated"), bool) for r in records)


def failure_categories(row):
    validation = row.get("validation", {})
    categories = []
    if row.get("truncated"):
        categories.append("truncated generation")
    if validation.get("remaining_checks"):
        categories.append("requested findings remain")
    if validation.get("new_diagnostics"):
        categories.append("new enabled warnings")
    if (validation.get("compiler_errors_after", 0) > validation.get("compiler_errors_before", 0)
            or validation.get("after", {}).get("errors")):
        categories.append("new compiler errors")
    if not categories and not passed(row):
        categories.append(validation.get("status", "not validated"))
    return categories


def build_report():
    artifacts = Artifacts()
    train = artifacts.read("models/collected_fixing_pilot_r16/run_settings.json") or {}
    manifest = artifacts.read("data/fixing_pilot/manifest.json") or {}
    audit = artifacts.read("data/fixing_pilot/audit.json") or {}
    runtime = artifacts.read("data/fixing_pilot/runtime_tokenization.json") or {}
    data = artifacts.read("data/fixing_pilot/test.jsonl", lines=True)
    by_id, ids = {row["id"]: row for row in data}, [row["id"] for row in data]
    main = artifacts.read("results/collected_fixing/pilot_test/metrics.json")
    raw_main = artifacts.read("results/collected_fixing/pilot_test/predictions.jsonl", lines=True)
    tool = artifacts.read("results/collected_fixing/tool_test/metrics.json")
    raw_tool = artifacts.read("results/collected_fixing/tool_test/predictions.jsonl", lines=True)
    behavior = artifacts.read("results/collected_fixing/behavior/metrics.json") or {}
    component = artifacts.read("results/collected_fixing/engine_component/metrics.json")
    pipeline = artifacts.read("results/collected_fixing/pipeline_test/metrics.json")
    support = artifacts.read("models/collected_fixing_pilot_r16/trained_checks.json") or {}
    detector_labels = artifacts.read("data/collected/detection/labels.json") or {}
    shared_checks = set(support.get("checks", [])) & set(detector_labels.get("checks", []))
    eligible_pipeline = [r for r in data if r.get("fixed_checks") and set(r["fixed_checks"]) <= shared_checks]
    references = artifacts.read("data/fixing_pilot/reference_static_validation.jsonl", lines=True)
    systems = {system: [r for r in raw_main if r.get("system") == system]
               for system in ("adapter", "base", "no_op", "clang_tidy_recorded_edits")}
    for system, key in (("no_op", "no_op"), ("clang_tidy_recorded_edits", "reference")):
        if not complete(systems[system], ids):
            systems[system] = [{"id": r["id"], "engine": by_id[r["id"]]["engine"],
                                "exact_reference": system == "clang_tidy_recorded_edits",
                                "changed": system == "clang_tidy_recorded_edits", "validation": r[key]}
                               for r in references if r["id"] in by_id]
    systems["fresh_clang_tidy"] = raw_tool
    finalized = {s: complete(rows, ids) and (s in ("no_op", "clang_tidy_recorded_edits") or
                                          (tool is not None if s == "fresh_clang_tidy" else main is not None))
                 for s, rows in systems.items()}
    expected_hash = manifest.get("output_sha256", {}).get("test")
    actual_hash = artifacts.hashes.get("data/fixing_pilot/test.jsonl")
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Frozen test file does not match its manifest hash")
    for row in raw_main + raw_tool:
        if row.get("dataset_sha256") != actual_hash:
            raise ValueError("Predictions belong to a different test dataset")
    lines = ["# Fixing pilot: measured results", "",
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
             "`scripts/report_fixing_pilot.py`. Regenerate this report from the saved evidence with the same script.", ""]
    if finalized["adapter"]:
        n_pass = sum(passed(r) for r in systems["adapter"])
        lines += [f"The trained adapter passes static validation on **{fraction(n_pass, len(ids))}** frozen "
                  "held-out functions. This measures compile/diagnostic checks under collection flags; "
                  "behavioral validation across the full test set and independent human acceptance "
                  "remain unmeasured. Any separate component experiment is scoped below.", ""]
    else:
        lines += ["Training has completed. Final learned-model test results are **pending**; "
                  "completed controls below must not be presented as learned-model performance.", ""]
    lines += ["## Experiment and actual training", "",
              "The target task is conditional repair: source code and reference clang-tidy check names "
              "are provided, and a complete replacement function is generated. Targets are tool edits, "
              "not human-validated fixes. The purposive pilot balances check/engine groups and selects "
              "short, strict-source functions. It does not estimate natural warning prevalence.", "",
              "| Setting | Observed value |", "| --- | --- |",
              f"| Base model | `{train.get('model', 'pending')}` |",
              f"| Frozen sample sizes | train {len(manifest.get('ids', {}).get('train', []))}; "
              f"validation {len(manifest.get('ids', {}).get('val', []))}; test {len(ids)} |",
              f"| Rows actually used by trainer | train {train.get('train_rows', 'pending')}; "
              f"validation {train.get('val_rows', 'pending')} |",
              f"| Training | {train.get('epochs_run', 'pending')} epochs; seed {train.get('seed', 'pending')}; "
              f"{train.get('steps_run', 'pending')} optimizer steps |",
              f"| LoRA | rank {train.get('lora_r', 'pending')}; {train.get('trainable_parameters', 'pending'):,} "
              "trainable parameters; attention and MLP projections |" if train.get('trainable_parameters') else "| LoRA | pending |",
              f"| Hardware and measured training cost | {train.get('gpu', 'pending')}; "
              f"{train.get('minutes', 'pending')} minutes; {train.get('peak_gpu_gb', 'pending')} GB peak allocated GPU memory |",
              f"| Best validation loss | {train.get('best_val_loss', 'pending')} at epoch {train.get('best_epoch', 'pending')} |", ""]
    engine_info = audit.get("pilot", {})
    lines += ["Train engines: " + ", ".join(sorted(engine_info.get("train", {}).get("engines", {}))) +
              ". Validation: Irrlicht. Test: " + ", ".join(f"{e} ({n} functions)" for e,n in
              sorted(Counter(r["engine"] for r in data).items())) + ".", ""]
    comparisons = runtime.get("comparison", {})
    if comparisons:
        lines += ["The tokenizer audit found stale stored token counts. The trainer excluded "
                  f"{comparisons.get('train', {}).get('excluded_rows_at_limit', 'pending')} training and "
                  f"{comparisons.get('val', {}).get('excluded_rows_at_limit', 'pending')} validation rows over its "
                  f"{train.get('max_length', 'pending')}-token limit rather than truncating answers. "
                  f"The frozen {len(ids)}-function test set was retained; its inference budget is separately checked.", ""]
        lines += ["The collection builder's tokenizer selection has now been corrected to use the "
                  "same pretrained base tokenizer as training for future builds. Frozen pilot files "
                  "were not rebuilt, so their hashes and actual retained training counts above "
                  "remain unchanged.", ""]
    lines += ["## Oracle-check repair comparison", "",
              "A static pass requires the generated patch in its original translation unit to be analyzed "
              "successfully, remove requested findings, and introduce no new compiler errors or enabled "
              "main-file diagnostics. Collection flags can contain stub headers. This is neither a native "
              "engine build nor a behavioral-equivalence test. Standalone snippet parsing is not used as "
              "a substitute for full-context compilation.", "",
              "| Method | Static pass / all cases | Exact recorded target | Changed input | Truncated |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for system, rows in systems.items():
        if finalized[system]:
            cells = [fraction(sum(passed(r) for r in rows), len(rows))]
            cells += [fraction(sum(bool(r.get(key)) for r in rows), len(rows))
                      if all(key in r for r in rows) else "not recorded"
                      for key in ("exact_reference", "changed", "truncated")]
        else:
            cells = [f"pending ({len(rows)}/{len(ids)} records available)"] + ["pending"] * 3
        lines.append(f"| {LABELS[system]} | " + " | ".join(cells) + " |")
    lines += ["", "Exact target match is secondary imitation evidence. Alternative valid patches can differ "
              "from the target; the recorded tool target can itself fail validation. Fresh clang-tidy "
              "generates replacement edits from the original source and supplied check names, applies "
              "them in memory inside the same function-only edit scope, and validates the result. "
              "Required include/declaration edits outside the function are scope abstentions, not "
              "evidence that clang-tidy cannot repair the full file. Full-file tool repair performance "
              "is unmeasured. Engine source files are never overwritten. "
              "Byte-identical patches may reuse same-session validation only when source/tool/flags "
              "provenance matches.", ""]
    lines += ["Both neural systems use the same mechanical code extraction: one complete fenced C++ "
              "block is accepted even when surrounded by prose; multiple blocks and truncated outputs "
              "are rejected. Raw generations were preserved and re-extracted without regenerating "
              "tokens when this parser was corrected. The earlier strict-format run is archived in "
              "`results/collected_fixing/pilot_test_strict_format/`. The comparison above measures "
              "validated candidate patches rather than treating surrounding explanation as repair failure.", ""]
    if finalized["adapter"] and finalized["fresh_clang_tidy"]:
        a = sum(passed(r) for r in systems["adapter"])
        t = sum(passed(r) for r in systems["fresh_clang_tidy"])
        comparison = "exceeds" if a > t else "matches" if a == t else "trails"
        lines += [f"Within the restricted function-only edit scope, the adapter **{comparison} the fresh tool** "
                  f"({a}/{len(ids)} versus {t}/{len(ids)}). "
                  "This does not compare with unrestricted full-file tool repair. Learning the "
                  "teacher's edits alone does not demonstrate a practical advantage.", ""]
    generation = (main or {}).get("generation", {})
    timings = generation.get("generation_seconds", {})
    if timings:
        lines += ["Measured generation wall times for the same test set: " +
                  "; ".join(f"{name}: {seconds:.1f} seconds" for name,seconds in timings.items()) +
                  ". These exclude model loading and independent static validation.", ""]
    if tool:
        abstentions = [r for r in raw_tool if r.get("candidate") is None]
        generated = [r for r in raw_tool if r.get("candidate") is not None]
        abstention_status = Counter(r.get("generation_status", "unknown") for r in abstentions)
        lines += [f"Fresh tool generation: {len(generated)}/{len(ids)} candidates and "
                  f"{len(abstentions)}/{len(ids)} generation abstentions (" +
                  ", ".join(f"{k}: {v}" for k,v in sorted(abstention_status.items())) +
                  f"). Among generated candidates, {sum(passed(r) for r in generated)}/{len(generated)} "
                  f"pass static validation and {sum(not passed(r) for r in generated)}/{len(generated)} "
                  "are statically rejected or inconclusive. The main all-case denominator retains "
                  "generation abstentions as unsuccessful attempts.", ""]
        lines += [f"Fresh tool wall time including validation: {tool.get('wall_seconds', 'not recorded')} seconds. "
                  "Its generation/validation worker timings overlap and are not directly comparable "
                  "to model-only generation latency.", ""]
    lines += ["## Paired differences and engine breakdown", "",
              "| Adapter minus comparator | Difference in static pass rate | Paired bootstrap 95% interval |",
              "| --- | ---: | ---: |"]
    for other in ("base", "clang_tidy_recorded_edits", "fresh_clang_tidy"):
        if finalized["adapter"] and finalized[other]:
            left = {r["id"]: r for r in systems["adapter"]}
            right = {r["id"]: r for r in systems[other]}
            estimate, low, high = paired_bootstrap([passed(left[i]) for i in ids], [passed(right[i]) for i in ids])
            lines.append(f"| {LABELS[other]} | {100*estimate:+.1f} percentage points | [{100*low:+.1f}, {100*high:+.1f}] |")
        else:
            lines.append(f"| {LABELS[other]} | pending | pending |")
    lines += ["", "Intervals resample the same paired functions 10,000 times with seed 20261015. "
              "They quantify function-level sampling uncertainty in this frozen pilot, not variation "
              "across training seeds, files, or engines. Functions can be correlated within a file. "
              "Two engines do not support a general cross-engine population claim.", "",
              "| Engine | " + " | ".join(LABELS[s] for s in systems) + " |",
              "| --- | " + " | ".join("---:" for _ in systems) + " |"]
    for engine in sorted({r["engine"] for r in data}):
        cells = []
        for system, rows in systems.items():
            subset = [r for r in rows if r.get("engine") == engine]
            cells.append(fraction(sum(passed(r) for r in subset), len(subset)) if finalized[system] else "pending")
        lines.append(f"| {engine} | " + " | ".join(cells) + " |")
    lines += ["", "## Failure evidence", ""]
    for system in ("adapter", "clang_tidy_recorded_edits", "fresh_clang_tidy"):
        if not finalized[system]:
            lines += [f"{LABELS[system]}: pending.", ""]
            continue
        failures = sorted([r for r in systems[system] if not passed(r)], key=lambda r:r["id"])
        counts = Counter(category for r in failures for category in failure_categories(r))
        lines += [f"**{LABELS[system]}:** {len(failures)}/{len(ids)} do not pass. Categories "
                  "can overlap: " + (", ".join(f"{k}: {v}" for k,v in counts.most_common()) or "none") + ".", ""]
        for row in failures[:3]:
            source = by_id[row["id"]]
            details = row["validation"].get("reasons", [])
            lines.append(f"- `{row['id']}` — {source['engine']}, `{source['path']}`: " +
                         "; ".join(details or failure_categories(row)) + ".")
        lines.append("")
    lines += ["Example IDs are selected in ID order among failures, not chosen as representative "
              "successes. Full generated candidates, checks, compiler diagnostics, and provenance "
              "are retained in the prediction artifacts linked below.", "",
              "## Independent illustrative behavior fixtures", "",
              f"The hand-written suite contains {behavior.get('fixture_count', 'pending')} standalone C++ cases "
              f"and {behavior.get('assertions_per_system', 'pending')} explicit input assertions per system. "
              "It covers else-after-return, braces, nullptr, boolean simplification, C-style casts, "
              "and redundant void arguments, with null, signed-boundary, and loop-boundary inputs. "
              "Compiler/run checks use these fixtures' independent expected values.", "",
              "| Fixture method | Compiled cases | Cases passing all assertions |",
              "| --- | ---: | ---: |"]
    for system in ("original", "reference", "intentional_bad", "adapter"):
        values = behavior.get("systems", {}).get(system)
        lines.append(f"| {system} | " + (f"{values['compiled']}/{values['cases']} | {values['passed']}/{values['cases']} |"
                                                   if values else "pending | pending |"))
    lines += ["", "The deliberately wrong but compilable controls test that the harness catches "
              "behavior changes. Passing these finite examples supports the fixture demonstration; "
              "it is not behavioral validation of the 128 held-out engine functions. This suite "
              "does not test warning removal, and unchanged originals should pass its behavior tests.", "",
              "## Preselected native engine component", ""]
    if component:
        lines += [f"The separate component experiment targets **{component.get('functions_repaired', 1)} "
                  f"function** (`{component.get('id', 'not recorded')}`) in "
                  f"{component.get('component', 'not recorded')}. Selection: "
                  f"{component.get('selection', 'not recorded')}. It compiles the copied actual "
                  "component source, with the candidate substituted into that function and real "
                  "engine headers. Configuration: " + str(component.get("configuration", "not recorded")) + ".", "",
                  f"The harness uses {component.get('keys', 'not recorded')} keys and "
                  f"{component.get('ordered_key_pairs', 'not recorded')} ordered key pairs, with "
                  f"{component.get('assertions_per_passing_candidate', 'not recorded'):,} "
                  "identity/ordering and boundary assertions per passing candidate. These are many "
                  "inputs for **one repaired function**, not independent repair examples.", "",
                  "| Component method | Compiled | Behavior assertions passed | Assertions completed |",
                  "| --- | --- | --- | ---: |"]
        for system in ("original", "reference", "intentional_bad", "plausible_bad_ignores_version", "base", "adapter"):
            value = component.get("systems", {}).get(system)
            if value:
                completed = value.get("assertions_completed", "stopped before completion")
                lines.append(f"| {system} | {value.get('compile_ok', False)} | "
                             f"{value.get('tests_passed', False)} | {completed} |")
            else:
                lines.append(f"| {system} | pending | pending | pending |")
        lines += ["", "Collection header stubs used in this component experiment: " +
                  str(component.get("uses_collection_header_stubs", "not recorded")) + ". "
                  "This is a native component experiment, not a full engine build. Serialization "
                  "metadata and runtime integration are outside its configuration. Finite tests "
                  "on one preselected function do not establish behavioral correctness on the "
                  "remaining held-out functions or general semantic equivalence. Component behavior "
                  "and clang-tidy warning removal are separate properties.", ""]
        for name, finding in component.get("negative_control_findings", {}).items():
            static_result = component.get("negative_control_static_validation", {}).get(name, {})
            lines += [f"A constructed semantic negative control (`{name}`) intentionally "
                      f"implements this fault: {finding.get('intentional_fault', 'not recorded')}. "
                      f"Measured full-source static validation: **{finding.get('static_validated', 'not recorded')}**; "
                      f"native component tests passed: **{finding.get('native_tests_passed', 'not recorded')}**. "
                      f"The static log records {static_result.get('compiler_errors_after', 'not recorded')} "
                      f"compiler errors after repair and {len(static_result.get('new_diagnostics', []))} "
                      "new enabled warnings. Removing the listed warning while ignoring the key's "
                      "version can satisfy the static gate and break intended equality. This "
                      "measured counterexample establishes that the static gate alone is insufficient "
                      "for behavioral correctness. It is a deliberate control, **not an observed "
                      "neural-model bug rate**.", ""]
    else:
        lines += ["Optional native component results are pending. No component success is assumed.", ""]
    lines += ["## Detector to repair pilot", ""]
    if pipeline:
        p = pipeline.get("metrics", {})
        design = pipeline.get("design", {})
        matched_ids = {r["id"] for r in eligible_pipeline}
        if matched_ids != set(design.get("selected_ids", [])):
            raise ValueError("Pipeline cohort differs from the current supported frozen rows")
        matched_adapter = [r for r in systems["adapter"] if r["id"] in matched_ids]
        lines += [f"All {design.get('selected_rows', p.get('rows', 'pending'))} eligible positive functions "
                  f"out of the frozen {design.get('source_rows', len(ids))} were evaluated. Complete reference "
                  "check sets must be supported by both the detector vocabulary and the actual repair "
                  "training support. Reference checks select and score cases; the two models receive "
                  "source and predicted findings, not reference findings.", "",
                  f"Pipeline static passes: **{fraction(p.get('pipeline_passes', 0), p.get('rows', 0))}**. "
                  f"Target checks detected: {p.get('recorded_target_checks_detected', 'pending')}/"
                  f"{p.get('recorded_target_checks_total', 'pending')}. "
                  f"Unnecessary repairs on clean functions: **unmeasured**.", "",
                  "Stage outcomes: " + ", ".join(f"{k}: {v}" for k,v in p.get("pipeline_status_counts", {}).items()) + ".", ""]
        if finalized["adapter"] and len(matched_adapter) == p.get("rows"):
            lines += ["On these same selected functions, the trained fixer given reference checks passes "
                      f"**{fraction(sum(passed(r) for r in matched_adapter), len(matched_adapter))}**. "
                      "The comparison isolates the observed loss when predicted checks replace reference "
                      "checks on this small matched cohort. Neither rate measures clean-function safety "
                      "or general end-to-end performance.", ""]
    else:
        lines += [f"Pipeline evaluation is pending for {len(eligible_pipeline)}/{len(ids)} "
                  "eligible frozen positive cases. Complete target check sets must belong to both "
                  "the detector vocabulary and actual repair training support. It is restricted to supported positive cases; "
                  "it must not be described as a broad end-to-end benchmark or as clean-function safety evidence.", ""]
    lines += ["## Workshop or short-paper assessment", "",
              "This is a reproducible implementation and an exploratory empirical study. A workshop "
              "or short-paper direction is plausible if the evidence explains a useful lesson about "
              "small-model repair, tool-generated supervision, context, or failure modes. The pilot "
              "does **not yet establish publication readiness**. A model that trails the tool calls "
              "for a limitation or comparative study, not a claim of superior automated repair.", "",
              "Before making a strong effectiveness claim, add independently reviewed engine patches, "
              "broader real regression tests/native builds on a declared subset, clean and unsupported inputs, "
              "multiple seeds, near-duplicate analysis, and a controlled class/type-context experiment "
              "if context is the proposed contribution. Establish a practical benefit beyond "
              "imitating existing automatic edits. The current experiment has one training seed, "
              "two held-out engines, no native engine builds, and no human acceptance study.", "",
              "See " + link("docs/REPAIR_RESEARCH_ASSESSMENT.md", "the research assessment and primary literature") +
              " for bounded research questions and the validation requirements.", "",
              "## Reproduction and exact evidence hashes", "", "```powershell",
              ".venv-demo/Scripts/python.exe scripts/report_fixing_pilot.py", "```", ""]
    for path in ("data/fixing_pilot/train.jsonl", "data/fixing_pilot/val.jsonl",
                 "models/collected_fixing_pilot_r16/adapter_model.safetensors",
                 "models/collected_fixing_pilot_r16/trained_checks.json",
                 "results/collected_fixing/pilot_test/generation_metadata.json",
                 "results/collected_fixing/behavior/predictions.jsonl",
                 "results/collected_fixing/engine_component/predictions.jsonl",
                 "scripts/collect/build_dataset.py",
                 "results/collected_fixing/pipeline_test/manifest.json",
                 "results/collected_fixing/pipeline_test/inference.jsonl",
                 "results/collected_fixing/pipeline_test/results.jsonl",
                 "results/collected_fixing/pipeline_test/audit.json"):
        artifacts.hash(path)
    lines += ["Hashes identify the exact snapshots read for this report. Incomplete/pending "
              "artifacts are not treated as final results. Local data/model artifacts may be ignored "
              "by Git and must accompany any reproducibility package under appropriate licences.", "",
              "| Evidence | SHA256 |", "| --- | --- |"]
    lines += [f"| {link(path)} | `{digest}` |" for path,digest in sorted(artifacts.hashes.items())]
    if artifacts.notes:
        lines += ["", "Pending evidence:", ""] + ["- " + note for note in artifacts.notes]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/FIXING_RESULTS.md")
    args = parser.parse_args()
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
