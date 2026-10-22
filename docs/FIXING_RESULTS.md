# Fixing pilot: measured results

Generated 2026-10-15 04:06 UTC by `scripts/report_fixing_pilot.py`. Regenerate this report from the saved evidence with the same script.

The trained adapter passes static validation on **69/128 (53.9%)** frozen held-out functions. This measures compile/diagnostic checks under collection flags; behavioral validation across the full test set and independent human acceptance remain unmeasured. Any separate component experiment is scoped below.

## Experiment and actual training

The target task is conditional repair: source code and reference clang-tidy check names are provided, and a complete replacement function is generated. Targets are tool edits, not human-validated fixes. The purposive pilot balances check/engine groups and selects short, strict-source functions. It does not estimate natural warning prevalence.

| Setting | Observed value |
| --- | --- |
| Base model | `deepseek-ai/deepseek-coder-1.3b-instruct` |
| Frozen sample sizes | train 4096; validation 128; test 128 |
| Rows actually used by trainer | train 3958; validation 124 |
| Training | 2 epochs; seed 42; 1357 optimizer steps |
| LoRA | rank 16; 14,991,360 trainable parameters; attention and MLP projections |
| Hardware and measured training cost | NVIDIA GeForce RTX 5090; 6.5 minutes; 15.5 GB peak allocated GPU memory |
| Best validation loss | 0.0164 at epoch 2 |

Train engines: Defold, Esoterica, Torque, cafu, cocos2d-x, godot. Validation: Irrlicht. Test: ogre (70 functions), spring (58 functions).

The tokenizer audit found stale stored token counts. The trainer excluded 138 training and 4 validation rows over its 2048-token limit rather than truncating answers. The frozen 128-function test set was retained; its inference budget is separately checked.

The collection builder's tokenizer selection has now been corrected to use the same pretrained base tokenizer as training for future builds. Frozen pilot files were not rebuilt, so their hashes and actual retained training counts above remain unchanged.

## Oracle-check repair comparison

A static pass requires the generated patch in its original translation unit to be analyzed successfully, remove requested findings, and introduce no new compiler errors or enabled main-file diagnostics. Collection flags can contain stub headers. This is neither a native engine build nor a behavioral-equivalence test. Standalone snippet parsing is not used as a substitute for full-context compilation.

| Method | Static pass / all cases | Exact recorded target | Changed input | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Trained 1.3B adapter | 69/128 (53.9%) | 57/128 (44.5%) | 124/128 (96.9%) | 0/128 (0.0%) |
| Untuned 1.3B base | 18/128 (14.1%) | 7/128 (5.5%) | 98/128 (76.6%) | 2/128 (1.6%) |
| Return input unchanged | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) |
| Recorded clang-tidy targets | 100/128 (78.1%) | 128/128 (100.0%) | 128/128 (100.0%) | 0/128 (0.0%) |
| Fresh clang-tidy (function scope) | 101/128 (78.9%) | 70/128 (54.7%) | 123/128 (96.1%) | 0/128 (0.0%) |

Exact target match is secondary imitation evidence. Alternative valid patches can differ from the target; the recorded tool target can itself fail validation. Fresh clang-tidy generates replacement edits from the original source and supplied check names, applies them in memory inside the same function-only edit scope, and validates the result. Required include/declaration edits outside the function are scope abstentions, not evidence that clang-tidy cannot repair the full file. Full-file tool repair performance is unmeasured. Engine source files are never overwritten. Byte-identical patches may reuse same-session validation only when source/tool/flags provenance matches.

Both neural systems use the same mechanical code extraction: one complete fenced C++ block is accepted even when surrounded by prose; multiple blocks and truncated outputs are rejected. Raw generations were preserved and re-extracted without regenerating tokens when this parser was corrected. The earlier strict-format run is archived in `results/collected_fixing/pilot_test_strict_format/`. The comparison above measures validated candidate patches rather than treating surrounding explanation as repair failure.

Within the restricted function-only edit scope, the adapter **trails the fresh tool** (69/128 versus 101/128). This does not compare with unrestricted full-file tool repair. Learning the teacher's edits alone does not demonstrate a practical advantage.

Measured generation wall times for the same test set: base: 394.4 seconds; adapter: 247.4 seconds. These exclude model loading and independent static validation.

Fresh tool generation: 123/128 candidates and 5/128 generation abstentions (conflicting_edits: 2, context_needed: 3). Among generated candidates, 101/123 pass static validation and 22/123 are statically rejected or inconclusive. The main all-case denominator retains generation abstentions as unsuccessful attempts.

Fresh tool wall time including validation: 144.426 seconds. Its generation/validation worker timings overlap and are not directly comparable to model-only generation latency.

## Paired differences and engine breakdown

| Adapter minus comparator | Difference in static pass rate | Paired bootstrap 95% interval |
| --- | ---: | ---: |
| Untuned 1.3B base | +39.8 percentage points | [+31.2, +49.2] |
| Recorded clang-tidy targets | -24.2 percentage points | [-32.8, -15.6] |
| Fresh clang-tidy (function scope) | -25.0 percentage points | [-33.6, -17.2] |

Intervals resample the same paired functions 10,000 times with seed 20261015. They quantify function-level sampling uncertainty in this frozen pilot, not variation across training seeds, files, or engines. Functions can be correlated within a file. Two engines do not support a general cross-engine population claim.

| Engine | Trained 1.3B adapter | Untuned 1.3B base | Return input unchanged | Recorded clang-tidy targets | Fresh clang-tidy (function scope) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ogre | 36/70 (51.4%) | 9/70 (12.9%) | 0/70 (0.0%) | 56/70 (80.0%) | 57/70 (81.4%) |
| spring | 33/58 (56.9%) | 9/58 (15.5%) | 0/58 (0.0%) | 44/58 (75.9%) | 44/58 (75.9%) |

## Failure evidence

**Trained 1.3B adapter:** 59/128 do not pass. Categories can overlap: requested findings remain: 43, new enabled warnings: 20, new compiler errors: 7.

- `fix-0175f3409ee0` — ogre, `PlugIns/ParticleFX/src/OgreHollowEllipsoidEmitter.cpp`: Requested diagnostics remain: bugprone-parent-virtual-call.
- `fix-08bb384d3e1c` — ogre, `Components/Volume/src/OgreVolumeGridSource.cpp`: Requested diagnostics remain: modernize-avoid-c-style-cast, modernize-use-auto.
- `fix-0934c6c650c8` — ogre, `Components/Overlay/src/OgrePanelOverlayElement.cpp`: Requested diagnostics remain: bugprone-parent-virtual-call.

**Recorded clang-tidy targets:** 28/128 do not pass. Categories can overlap: new enabled warnings: 21, requested findings remain: 10, new compiler errors: 2.

- `fix-0cc5f6c6512b` — spring, `rts/System/Net/LocalConnection.cpp`: Candidate introduces new enabled main-file clang-tidy diagnostics.
- `fix-0cd1bf99495e` — ogre, `OgreMain/include/OgreExternalTextureSource.h`: Candidate introduces new enabled main-file clang-tidy diagnostics.
- `fix-1abedbce82e1` — ogre, `Components/Bites/include/OgreInput.h`: Candidate introduces new enabled main-file clang-tidy diagnostics.

**Fresh clang-tidy (function scope):** 27/128 do not pass. Categories can overlap: new enabled warnings: 16, inconclusive: 5, new compiler errors: 3, requested findings remain: 3.

- `fix-0cc5f6c6512b` — spring, `rts/System/Net/LocalConnection.cpp`: Candidate introduces new enabled main-file clang-tidy diagnostics.
- `fix-0cd1bf99495e` — ogre, `OgreMain/include/OgreExternalTextureSource.h`: Candidate introduces new enabled main-file clang-tidy diagnostics.
- `fix-1abedbce82e1` — ogre, `Components/Bites/include/OgreInput.h`: Candidate introduces new enabled main-file clang-tidy diagnostics.

Example IDs are selected in ID order among failures, not chosen as representative successes. Full generated candidates, checks, compiler diagnostics, and provenance are retained in the prediction artifacts linked below.

## Independent illustrative behavior fixtures

The hand-written suite contains 6 standalone C++ cases and 36 explicit input assertions per system. It covers else-after-return, braces, nullptr, boolean simplification, C-style casts, and redundant void arguments, with null, signed-boundary, and loop-boundary inputs. Compiler/run checks use these fixtures' independent expected values.

| Fixture method | Compiled cases | Cases passing all assertions |
| --- | ---: | ---: |
| original | 6/6 | 6/6 |
| reference | 6/6 | 6/6 |
| intentional_bad | 6/6 | 0/6 |
| adapter | 6/6 | 6/6 |

The deliberately wrong but compilable controls test that the harness catches behavior changes. Passing these finite examples supports the fixture demonstration; it is not behavioral validation of the 128 held-out engine functions. This suite does not test warning removal, and unchanged originals should pass its behavior tests.

## Preselected native engine component

The separate component experiment targets **1 function** (`fix-31c269426bd1`) in Spring AIInterfaceKey. Selection: preselected GUI default equality example before model results; no outcome-based search. It compiles the copied actual component source, with the candidate substituted into that function and real engine headers. Configuration: BUILDING_AI: real engine creg_cond.h disables serialization metadata.

The harness uses 72 keys and 5184 ordered key pairs, with 31,468 identity/ordering and boundary assertions per passing candidate. These are many inputs for **one repaired function**, not independent repair examples.

| Component method | Compiled | Behavior assertions passed | Assertions completed |
| --- | --- | --- | ---: |
| original | True | True | 31468 |
| reference | True | True | 31468 |
| intentional_bad | True | False | stopped before completion |
| plausible_bad_ignores_version | True | False | stopped before completion |
| base | True | True | 31468 |
| adapter | True | True | 31468 |

Collection header stubs used in this component experiment: False. This is a native component experiment, not a full engine build. Serialization metadata and runtime integration are outside its configuration. Finite tests on one preselected function do not establish behavioral correctness on the remaining held-out functions or general semantic equivalence. Component behavior and clang-tidy warning removal are separate properties.

A constructed semantic negative control (`plausible_bad_ignores_version`) intentionally implements this fault: compare short name but ignore version. Measured full-source static validation: **True**; native component tests passed: **False**. The static log records 0 compiler errors after repair and 0 new enabled warnings. Removing the listed warning while ignoring the key's version can satisfy the static gate and break intended equality. This measured counterexample establishes that the static gate alone is insufficient for behavioral correctness. It is a deliberate control, **not an observed neural-model bug rate**.

## Detector to repair pilot

All 20 eligible positive functions out of the frozen 128 were evaluated. Complete reference check sets must be supported by both the detector vocabulary and the actual repair training support. Reference checks select and score cases; the two models receive source and predicted findings, not reference findings.

Pipeline static passes: **7/20 (35.0%)**. Target checks detected: 12/20. Unnecessary repairs on clean functions: **unmeasured**.

Stage outcomes: repair_generated: 11, detector_requested_no_repair: 7, abstained_unsupported_prediction: 2.

On these same selected functions, the trained fixer given reference checks passes **12/20 (60.0%)**. The comparison isolates the observed loss when predicted checks replace reference checks on this small matched cohort. Neither rate measures clean-function safety or general end-to-end performance.

## Workshop or short-paper assessment

This is a reproducible implementation and an exploratory empirical study. A workshop or short-paper direction is plausible if the evidence explains a useful lesson about small-model repair, tool-generated supervision, context, or failure modes. The pilot does **not yet establish publication readiness**. A model that trails the tool calls for a limitation or comparative study, not a claim of superior automated repair.

Before making a strong effectiveness claim, add independently reviewed engine patches, broader real regression tests/native builds on a declared subset, clean and unsupported inputs, multiple seeds, near-duplicate analysis, and a controlled class/type-context experiment if context is the proposed contribution. Establish a practical benefit beyond imitating existing automatic edits. The current experiment has one training seed, two held-out engines, no native engine builds, and no human acceptance study.

See [the research assessment and primary literature](../docs/REPAIR_RESEARCH_ASSESSMENT.md) for bounded research questions and the validation requirements.

## Reproduction and exact evidence hashes

```powershell
.venv-demo/Scripts/python.exe scripts/report_fixing_pilot.py
```

Hashes identify the exact snapshots read for this report. Incomplete/pending artifacts are not treated as final results. Local data/model artifacts may be ignored by Git and must accompany any reproducibility package under appropriate licences.

| Evidence | SHA256 |
| --- | --- |
| [data/collected/detection/labels.json](../data/collected/detection/labels.json) | `de5aa98cd36007102bb49a6510b7ff04c57c9fa844000f34a55aa7bd0371f86c` |
| [data/fixing_pilot/audit.json](../data/fixing_pilot/audit.json) | `816a6207f1e9a30a51ac80535f2779dc320a9c8b178cdc02b5e0bd67353a090a` |
| [data/fixing_pilot/manifest.json](../data/fixing_pilot/manifest.json) | `aab49c3d939c5cb9416a35613f7cdcaebad7298a07e307012307a46a81731479` |
| [data/fixing_pilot/reference_static_validation.jsonl](../data/fixing_pilot/reference_static_validation.jsonl) | `b679e2215ff0294545439b97f537fb5cf93730e90c86f3bb609056adfdae941e` |
| [data/fixing_pilot/runtime_tokenization.json](../data/fixing_pilot/runtime_tokenization.json) | `500843fdc84b12e7afe2a1f8118ee821e91369633c78c9cde19aa6a9999ab8f7` |
| [data/fixing_pilot/test.jsonl](../data/fixing_pilot/test.jsonl) | `51fd9bc5b2de34db4e6e9633d90efcc06e7314a535b5b0fd610ef7911a87b161` |
| [data/fixing_pilot/train.jsonl](../data/fixing_pilot/train.jsonl) | `472df4e7d0ab30bcdcf0c2459b8db65d9e2427690f4b2e2d4eb72371d79330aa` |
| [data/fixing_pilot/val.jsonl](../data/fixing_pilot/val.jsonl) | `034d663aadfaa9b657d9d8ecdc80357d43d84bdf88e3f3c5e33a5120bb4abc39` |
| [models/collected_fixing_pilot_r16/adapter_model.safetensors](../models/collected_fixing_pilot_r16/adapter_model.safetensors) | `f23bacaea014b06390d64660040bbb95d3f8b64f15014d55a3a628552ad8c89c` |
| [models/collected_fixing_pilot_r16/run_settings.json](../models/collected_fixing_pilot_r16/run_settings.json) | `74b57e72b3ca15bc0fa5f7f38e5ad2388d19589fc2d39dc8c8ee4c7bb165a4c2` |
| [models/collected_fixing_pilot_r16/trained_checks.json](../models/collected_fixing_pilot_r16/trained_checks.json) | `3aac78a0f616ade9ecedc34415500d61ddb13778373a3ac51eb5de3ece870fae` |
| [results/collected_fixing/behavior/metrics.json](../results/collected_fixing/behavior/metrics.json) | `6702e46cff35869e768bf9e6056813d4b03dd53d3555759e06fd3ba3297c6ab9` |
| [results/collected_fixing/behavior/predictions.jsonl](../results/collected_fixing/behavior/predictions.jsonl) | `6725cb1c29c72424a892ac6d7d7e4cac04a2b6d0c5015e21d1fb373286be9b04` |
| [results/collected_fixing/engine_component/metrics.json](../results/collected_fixing/engine_component/metrics.json) | `075aa50a51b69e29261b6072479a59b45d568b5c1e5217683bc40d8325855c8f` |
| [results/collected_fixing/engine_component/predictions.jsonl](../results/collected_fixing/engine_component/predictions.jsonl) | `e373fad11b22c6829bf6c9151670b0414c8e859a4297f8a8f9fe50f6f93184d3` |
| [results/collected_fixing/pilot_test/generation_metadata.json](../results/collected_fixing/pilot_test/generation_metadata.json) | `b7582231fb032802c227f8aa28269828c33106f5c254417a874bcf3ebd76ab18` |
| [results/collected_fixing/pilot_test/metrics.json](../results/collected_fixing/pilot_test/metrics.json) | `90c309fc7299a6dcc70a47100b2f3cfc410d055232b70a62f9b334b46a9be567` |
| [results/collected_fixing/pilot_test/predictions.jsonl](../results/collected_fixing/pilot_test/predictions.jsonl) | `9298155720b92386dca2669f0fc25e08ab6653ad9f3614fb4fd8f074c8a1f91d` |
| [results/collected_fixing/pipeline_test/audit.json](../results/collected_fixing/pipeline_test/audit.json) | `4ba62394b7df8346d2826a843c0e3e4785d21d7d5561accc26e88a9d31fb88f1` |
| [results/collected_fixing/pipeline_test/inference.jsonl](../results/collected_fixing/pipeline_test/inference.jsonl) | `79dd3c753b61f05ce5ecf5effd8b7b9d19ac261e7af2fc5e19f139e4669dcbe7` |
| [results/collected_fixing/pipeline_test/manifest.json](../results/collected_fixing/pipeline_test/manifest.json) | `0eed130d4f85c89728057f3b76a8ce92b7987b4bf9fdc86ee5fd670061042926` |
| [results/collected_fixing/pipeline_test/metrics.json](../results/collected_fixing/pipeline_test/metrics.json) | `a1a04e3f8ea199e91de23c27b0d701717c69031cb16b1865fa192a12cbdaabfb` |
| [results/collected_fixing/pipeline_test/results.jsonl](../results/collected_fixing/pipeline_test/results.jsonl) | `106c31bfcf264953e60f5fe458dbf85fb3dcd93b26095c00a736d348fc284829` |
| [results/collected_fixing/tool_test/metrics.json](../results/collected_fixing/tool_test/metrics.json) | `962ae4b2b9bd76c2fed7a7c19b9113238531f640050b758e070233b9d545e647` |
| [results/collected_fixing/tool_test/predictions.jsonl](../results/collected_fixing/tool_test/predictions.jsonl) | `80bd9e8ecf5051d3c7ffa5d58518ecad9e94176a45fe29bba342fd7d2634044e` |
| [scripts/collect/build_dataset.py](../scripts/collect/build_dataset.py) | `a299c1d43016bc42c3e000ff47471db26de0e73ad73a2043f014bf8f69564a04` |
