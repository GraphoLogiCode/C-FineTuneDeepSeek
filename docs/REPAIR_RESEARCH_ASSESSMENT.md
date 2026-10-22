# Repair feasibility and research assessment

Assessed 14 October 2026. Corpus counts below were recomputed from the local
`data/collected/` JSONL files on this date. Literature claims were checked against
the linked primary sources. This document defines the research claims a repair
pilot can support; measured results from a particular run must be reported with
that run's saved predictions and validation logs.

## Assessment

A small, local repair adapter and an executable validation pipeline are feasible
with this repository. A working demonstration, low training loss, or close
agreement with a reference patch does not establish a publishable contribution.
The useful next result is a measured answer to a bounded question about repair
quality, context, cross-engine transfer, and compute cost.

The existing fixing data teaches the model to imitate **clang-tidy's own
automatic edits**. A paper must explain why learning those edits is useful when
clang-tidy can already apply them. The direct tool is an essential baseline.
An honest negative result, such as finding that a small model adds errors to
deterministic fixes, can still inform a carefully designed empirical study;
publication depends on the contribution, evidence, and venue review.

## What the current dataset actually represents

The implementation inspected in `scripts/collect/build_dataset.py` does this:

1. Take a function and its exported clang-tidy findings.
2. Select findings with automatic replacement edits under the fixing policy.
3. Apply noncolliding edits that fit inside the extracted function.
4. Drop unchanged targets and targets that introduce a tree-sitter parse error.
5. Train on the original function plus **reference check names**, with the edited
   function as the target.

The current fixing prompt asks for a complete function, without explanations or
JSON. It receives check names; it does not receive the exported replacement
offsets or replacement text. This avoids giving away the answer during model
inference, but the supervision remains tool-generated.

The collection documentation records 18,849 training, 2,569 validation, and 3,827
test pairs. Validation uses Irrlicht; testing uses OGRE and Spring; the other
seven engines supply training data. Evaluation functions originate from files
with no reported compiler errors under the collection flags. This does not mean
that their **fixed targets** have been compiled, re-linted, or tested.

The detection and repair tasks currently have different vocabularies:

| Quantity recomputed from JSONL | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Fixing pairs | 18,849 | 2,569 | 3,827 |
| Distinct requested fixing checks in split | 79 | 35 | 59 |
| Pairs containing any detector-vocabulary fixing check | 5,825 | 517 | 924 |
| Pairs with every requested fixing check in detector vocabulary | 2,499 | 101 | 419 |

Across splits, there are **83** fixing checks and **40** detection checks; their
intersection contains **15** checks. Braces, C-style casts, `auto`, and `nullptr`
dominate fixing data, while the detection policy deliberately excludes many
formatting and modernization checks. Counts are per function/check membership;
a function can request several fixes.

Consequences:

- Supplying `fixed_checks` from the test row evaluates **repair given reference
  findings**. Describe it as conditional repair or oracle-check repair.
- It does not evaluate the detector followed by the fixer. That experiment must
  use predicted checks, include detector misses and false positives, and define
  a shared repair vocabulary.
- To restrict evaluation to the 15-check intersection, rebuild targets using
  only the retained edits. Filtering a multi-check row's prompt while leaving
  its full edited target would reward changes that were never requested.
- A test set containing only fixable positive functions cannot measure needless
  edits to clean code. Add independently validated clean and unsupported cases.

The strongest immediate claim is about reproducing local lint edits in these
engine sources. Calling these labels validated game-specific smells requires a
separate domain definition and human annotation study.

## Prior work changes the novelty claim

| Primary source | Established contribution | Implication here |
| --- | --- | --- |
| [Berabi et al., TFix, ICML 2021](https://proceedings.mlr.press/v139/berabi21a.html) | Fine-tunes a text-to-text model on developer fixes, conditioned on static-analysis information; its evaluation covers 52 JavaScript error types. | Learning to repair analyzer findings is established. A new language and model checkpoint alone give a weak novelty argument. |
| [You, Choi and Hong, January 2026 preprint](https://arxiv.org/html/2601.14936v1) | Repairs C++ implicit-data-loss warnings using clangd/LSP context, tree-sitter extraction, and range-check decisions. In SAP HANA, developers accepted 102 of 110 patches for 135 warnings. | C++ warning repair with type context already exists. Compare the small-model setting and cross-engine evidence rather than claim this general idea is new. Its acceptance rate is not comparable to exact-match accuracy. |
| [Zhang et al., TOSEM 2026 accepted preprint](https://arxiv.org/abs/2412.13801) | Compares four parameter-efficient methods across nine models on four manually reviewed smell categories. | Parameter-efficient smell detection is established; human validation and strong baselines set a useful methodological standard. This is detection, not a repair benchmark. |
| [Xu et al., SWE-Refactor, February 2026 preprint](https://arxiv.org/abs/2602.03712) | Provides 1,099 developer-written Java refactorings across 18 projects, with compilation, test execution, and automated refactoring checks. | Realistic refactoring evaluation requires repository context and executable validation. Isolated function similarity is a weaker claim. |
| [Smith et al., FSE 2015](https://people.cs.umass.edu/~brun/pubs/pubs/Smith15fse.pdf) | Evaluates patch quality with tests independent of those used to generate repairs and studies overfitting. | Passing the tests that guide repair is insufficient evidence of general correctness. Keep independent tests and review. |

These are related examples, not an exhaustive systematic review. No claim that
this is the first C++ game-engine repair study follows from this search.

## A defensible study and research questions

A suitable working scope is **validated repair of local C++ diagnostics in
unseen game engines using small local models**. Choose an explicit warning
subset before the final experiment, and separate deterministic syntax changes
from changes requiring class/type or semantic context.

Suggested questions:

1. **RQ1 — Repair quality:** How often does a 1.3B fine-tuned model produce a
   compilable, warning-resolving, behavior-acceptable patch on engines excluded
   from training, compared with direct clang-tidy fixes and untuned models?
2. **RQ2 — Context:** How does providing relevant declarations and types affect
   validated repair success for checks that require information outside the
   function, relative to function-only input under a comparable token budget?
3. **RQ3 — Practical cost:** What are the differences in success, latency,
   memory, and unnecessary edits between the small tuned model, a reasonable
   larger local comparator, and the deterministic tool?
4. **RQ4 — Full pipeline:** How much repair performance is lost when predicted
   findings replace reference findings, including clean and unsupported cases?

RQ2 is a hypothesis. An illustrative changed answer after adding a declaration
does not establish a context benefit. Train comparable function-only and
context-enabled adapters; evaluate paired examples and measure context
extraction failures. Keep reference fixes and post-fix information out of both
prompts. A context budget matched control helps distinguish relevant
declarations from simply supplying more tokens.

If the goal is a stronger repair contribution than imitation, collect
developer-reviewed fixes for warnings that lack automatic edits, or evaluate
cases where the tool edit fails an independently defined quality criterion.
That is a different dataset and cannot be claimed from the current supervision.

## What “working” must mean

Use a ladder of evidence, reporting each level separately:

| Level | Observable criterion | What it establishes |
| --- | --- | --- |
| Output | A nonempty complete function is produced, without truncation or unrelated text. | The model follows the interface. |
| Parse | The generated function parses. | Syntax-level plausibility only. |
| Compile | Insert the patch into its original file and run the same valid translation-unit compile command. | Type and declaration compatibility for that build configuration. |
| Re-lint | Targeted findings disappear; no new relevant findings or compiler errors appear under fixed tool/check settings. | The analyzer accepts the patch in that context. |
| Independent tests | Existing regression tests and independently authored boundary/property tests pass. | Behavior is consistent with the tested specification. |
| Review | Reviewers judge the edit correct for program intent, appropriately scoped, and maintainable. | Human evidence about matters not captured by the tool and tests. |
| Performance | Relevant benchmarks show no unacceptable regression. | Evidence for efficiency claims, when such claims are made. |

Do not combine a generated, parsed function with a successful compile of the
unchanged source and call the generated patch compiled. Every outcome must link
to the exact generated text, source revision, command, and log.

Diagnostic removal is not a correctness proof. A model can delete logic, add
`NOLINT`, disable code, or insert an explicit narrowing cast that silences a
warning while retaining unintended data loss. Reject suppressions and
unrequested deletions under a published policy; inspect intended behavior.

Some targets are bug fixes, such as initializing an uninitialized variable;
they are not strictly behavior-preserving refactorings of the original buggy
program. State the intended postcondition. For actual refactorings, preserve
well-defined observable behavior. Tests and review provide evidence under
specific conditions, not a proof for all inputs.

Adding `const` to a method or changing parameter types may require a separate
header declaration edit and updating callers. A function-only model can produce
a perfectly parsable answer that breaks the project. Either support a
multi-file patch interface or explicitly exclude such repairs from the supported
pilot; count them when reporting coverage.

## Minimal rigorous evaluation protocol

```text
Algorithm: Evaluate one repair method on a frozen case
Inputs: source revision, function span, allowed checks, build command,
        prompt-only context, independent tests, repair method
Outputs: generated patch, timings, logs, success/failure at each validation stage

1. Verify source hashes and confirm the before-state compiles and triggers
   the selected findings. Mark unverified cases ineligible, with a reason.
2. Construct the input using either reference findings or detector predictions;
   record which mode is used. Never expose the expected after-state.
3. Generate once with the frozen decoding settings. Save the unmodified output.
4. Validate the output contract, then apply the patch to an isolated source copy.
5. Parse, compile, re-lint, and run independent tests, retaining failed cases.
6. Check the edit policy and obtain blinded human review for the review sample.
7. Record each outcome and the exact denominator. Restore the isolated copy.
```

Run the same cases and budgets for these baselines:

- **Identity:** return the original function; exposes metrics that reward doing
  nothing. It should fail genuine warning-removal cases.
- **Direct clang-tidy `--fix`:** use the same version and configuration, on an
  isolated source copy. The [official documentation](https://clang.llvm.org/extra/clang-tidy/)
  documents direct fixes, exported replacement edits, and compilation databases.
  Replaying cached tool edits may be a useful separate baseline but must be
  named separately from running the tool.
- **Untuned base with a fair prompt:** provide the same code, findings, allowed
  output contract, and relevant context. Prompt selection uses validation data.
- **Fine-tuned function-only model:** the immediate adapter experiment.
- **Fine-tuned context model:** same base model, data identities, training budget,
  and decoding settings, with class/type context as the controlled change.
- **Larger local model:** useful for the resource tradeoff question, with size
  chosen to fit available hardware and a declared inference budget.

Report exact or normalized reference match only as a secondary **imitation**
metric. Multiple correct patches can differ from a reference, while a matching
tool patch can still be inappropriate for program intent. BLEU/ROUGE and loss
alone cannot answer RQ1.

Report one-attempt success, generation failures, compile success, targeted
warning removal, new-warning rate, independent-test success, reviewed acceptable
patch rate, and unnecessary-edit rate on clean inputs. Include all eligible
inputs in unconditional denominators. Report validation-stage rates among
generated or compilable patches as additional conditional measures. Publish
timeouts and missing-build cases instead of silently removing difficult cases.

Use fixed source commits, pinned tools, train/validation/test identities and
cross-split near-duplicate checks. Reserve the final test split for the final
design, and do not select the best checkpoint, prompt, or warning subset by its
test score. Several training seeds are needed to assess training variability.
Use paired uncertainty estimates; functions from a file or engine are not
independent replicates. Two held-out engines support conclusions about those
engines, with limited evidence about unseen engines generally. More held-out
engines or leave-one-engine-out repetitions would strengthen that claim.

Sample review cases across check type, engine, and automatic-validation outcome;
report the selection rule and counts. Two independent reviewers and adjudicated
disagreements give stronger evidence than self-review by the model developer.
No user instruction to contact reviewers or publish artifacts is implied here.

## Adding class and type information

Use compiler-aware symbol resolution when build metadata is available. For each
referenced symbol relevant to a diagnostic, retrieve its declaration, resolved
type, enclosing class fields and method signatures, and necessary aliases or
enums. Keep a provenance location for each item. A tree-sitter text search is a
useful fallback but cannot reliably resolve overloads, templates, macros, and
aliases in C++.

Represent the material in a dedicated prompt section marked read-only. Keep the
editable function in a separate section; request only the allowed patch scope.
Retrieval should not incorporate reference fixes, held-out labels, or future
source revisions. Train on this same representation, and evaluate with context
obtained by the same retrieval procedure. Report unavailable context rather
than filling it with invented declarations.

For narrowing conversions, the useful context includes source/destination types
and, where obtainable, contracts or bounds on returned values. Types alone do
not establish a safe value range. For const/static eligibility, class and method
declarations matter, and correctness may require edits beyond the function.

## What a pilot can establish, and the paper decision

A pilot with saved generations and real parse/compile/re-lint/test logs can
establish that the implementation performs some validated repairs under its
explicit environment and case selection. A small training subset establishes
pipeline feasibility; it does not establish the final model's capability.
Standalone hand-written fixtures establish behavior on those fixtures. Held-out
engine functions reinserted into valid original translation units provide
stronger evidence for practical transfer.

Before presenting a paper as an effective game-engine repair system, obtain:

1. A fair direct-tool comparison and clear value beyond duplicating its edits.
2. Real held-out engine validation and a disclosed validation coverage rate.
3. A controlled context experiment if context is the contribution.
4. Independent correctness evidence, including clean-input behavior.
5. Repeated runs, paired uncertainty, reproducible artifacts, and source-code
   attribution/licensing appropriate to the artifacts released.
6. A literature review broad enough to substantiate the precise novelty claim.

If the tuned model only reproduces easy tool edits and loses on correctness or
cost, the evidence supports a limitation study or course project. If it provides
a measured benefit on context-sensitive, previously unsupported repairs while
preserving build/test behavior and controlling resource use, the case for a
research paper is materially stronger. The experiments determine which claim is
defensible; successful training alone does not.
