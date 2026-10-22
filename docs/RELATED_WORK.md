# Related work: detecting code smells with language models

Collected on 7 October 2026 by web search. Each entry was checked against its abstract or publisher page unless marked otherwise; none was read in full. Numbers are the authors' own.

PDFs of the open-access papers are in the local `Papers/` folder, which is not in git; `Papers/README.md` lists what was downloaded and where to get the rest.

**Summary.** Almost all of this work is on Java, uses design smells such as God Class and Feature Envy, and relies on manually reviewed labels. The search found no published study of smell detection on C++ game-engine code, so this project's setting is open; what it lacks compared with the studies below is validated labels.

## 1. Fine-tuning code models to detect smells

These are the closest to this project.

- **Zhang, Liang, Zhou, Zhou, Lo, Feng, Li and Li — [A Comprehensive Evaluation of Parameter-Efficient Fine-Tuning on Code Smell Detection](https://arxiv.org/abs/2412.13801)** (arXiv 2024; accepted at TOSEM 2026).
  - Compares prompt tuning, prefix tuning, LoRA and (IA)³ on four small models (CodeBERT, GraphCodeBERT, UniXcoder, CodeT5) and five larger ones, including **DeepSeek-Coder-1.3B** and 6.7B.
  - Data: 401 Java repositories; four smells (Complex Conditional, Complex Method, Feature Envy, Data Class); labels from DesigniteJava and PMD followed by a two-stage manual review.
  - Findings: PEFT matches or beats full fine-tuning; results are sensitive to the LoRA rank, and a higher rank is not reliably better; performance drops sharply with few training samples; tuned models beat heuristic tools and prompted LLMs (MCC gains of 0.33% to 13.69%).
  - Relevance: same base model and the same four PEFT options as this project's notebook. Its label pipeline (tools, then manual review) is the step this project skipped.
- **Alomari, Redah, Ashraf and Alshayeb — [Smelly-shot is all you need: an empirical study to compare in-context learning versus fine tuning for code smell detection](https://pure.kfupm.edu.sa/en/publications/smelly-shot-is-all-you-need-an-empirical-study-to-compare-in-cont/)** (Empirical Software Engineering, 2026).
  - Fine-tunes CodeBERT, GraphCodeBERT, UniXcoder, CodeGPT and GPT-2, and compares with in-context learning on GPT-4o Mini, Gemini Flash and CodeLlama 70B, for God Class and Data Class.
  - Finding: fine-tuning beats in-context learning by 26% mean F1; in-context learning needs far less data and generalises better.
- **Liu, Zhang, Saikrishna, Tian and Zheng — [Prompt Learning for Multi-Label Code Smell Detection: A Promising Approach](https://arxiv.org/abs/2402.10398)** (arXiv 2024). Treats multi-label smell detection as a masked-token task; reports 11.17% higher weighted precision and 7.4% higher weighted F1 than earlier approaches on Java.
- **Alomari, Alazba, Aljamaan and Alshayeb — [SmellyCode++: Multi-Label Dataset for Code Smell Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC12255726/)** (Scientific Data, 2025). 107,554 Java samples from 103 projects with source text and metrics, labelled for four smells. Argues that existing datasets are too small, single-label, or metrics-only to fine-tune language models.

## 2. Prompting general LLMs to detect smells

- **Souza, Santana, Figueiredo, Muzetti, Montandon and Briand — [Beyond Strict Rules: Assessing the Effectiveness of Large Language Models for Code Smell Detection](https://arxiv.org/abs/2601.09873)** (arXiv 2026).
  - Tests DeepSeek-R1, GPT-5 mini, Llama-3.3 and Qwen2.5-Code on nine smells in 30 Java projects; the ground truth is 268 candidates inspected by 76 developers.
  - Findings: LLMs do well on structurally simple smells (Large Class, Long Method); combining an LLM with a static-analysis tool gives the best F1 for five of the nine smells.
- **Sadik and Govind — [Benchmarking LLM for Code Smells Detection: OpenAI GPT-4.0 vs DeepSeek-V3](https://arxiv.org/abs/2504.16027)** (arXiv 2025; EASE 2025). The only detection study found that includes **C++** (with Java, Python and JavaScript). GPT-4.0 reaches a precision of 0.79 against 0.42 for DeepSeek-V3; both have low recall.
- **Silva, Silva, Montandon, Andrade and Valente — [Detecting Code Smells using ChatGPT: Initial Insights](https://conf.researchr.org/details/esem-2024/esem-2024-emerging-results-vision-and-reflection-papers/2/Detecting-Code-Smells-using-ChatGPT-Initial-Insights)** (ESEM 2024). Four Java smells at three severity levels. Naming the smells in the prompt makes a correct answer 2.54 times more likely than a generic prompt; F1 is 0.52 for critical smells and 0.43 for minor ones.
- **Gomes, Sousa, Maia and Paixao — [Attentionsmelling: Using Large Language Models to Identify Code Smells](https://sol.sbc.org.br/index.php/sbes/article/view/37005)** (SBES 2025). GPT-4 on Long Method, God Class and Feature Envy. Adding code metrics and context to the prompt improves F1 by 56% over the starting setup of code plus smell definitions.
- **Fahad, Asif and Tawhid — [Mitigating LLM Sycophancy in Code Smell Detection Using Evidence-Guided Reasoning Prompts](https://arxiv.org/abs/2607.10411)** (arXiv 2026). On the MLCQ dataset, a prompt that hints at the answer flips up to 72% of decisions. Relevant to prompt design: the original system prompt here told the model which smells to prioritise and how severe they are.

## 3. Combining LLMs with tools, and fixing

- **Wu, Mu, Shi, Guo, Liu, Zhuang, Zhong and Zhang — [iSMELL: Assembling LLMs with Expert Toolsets for Code Smell Detection and Refactoring](https://conf.researchr.org/details/ase-2024/ase-2024-research/108/iSMELL-Assembling-LLMs-with-Expert-Toolsets-for-Code-Smell-Detection-and-Refactoring)** (ASE 2024). A learned selector picks the best detection tool per smell, and the tool's findings are put in the LLM's prompt for refactoring. Average detection F1 of 75.17%, 35.05% above LLM-only baselines.
- **You, Choi and Hong — [LLM-Based Repair of C++ Implicit Data Loss Compiler Warnings: An Industrial Case Study](https://arxiv.org/abs/2601.14936)** (arXiv 2026). **C++**, with tree-sitter to extract the function and a language server for type context. 92.73% of fixes were accepted by reviewers. The closest match to this project's fixing idea.
- **Wadhwa et al. — [Frustrated with Code Quality Issues? LLMs can Help!](https://arxiv.org/abs/2309.12938)** (arXiv 2023). One LLM proposes a fix for a static-analysis finding and a second ranks it; 59.2% of Python files and 76.8% of Java files pass both the tool and human review.
- **Berabi, He, Raychev and Vechev — [TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer](https://proceedings.mlr.press/v139/berabi21a.html)** (ICML 2021). Fine-tunes T5 on before/after pairs mined from GitHub commits for 52 static-analyser error types; fixes 67% of cases. A model for how to obtain real fix pairs.
- **Cordeiro, Noei and Zou — [An Empirical Study on the Code Refactoring Capability of Large Language Models](https://arxiv.org/abs/2411.02320)** (arXiv 2024). StarCoder2 on 30 Java projects removes more simple smells than developers do, but developers do better on design-level ones.

## 4. Game-specific smells

These define what a "game smell" is. All are about game projects (mostly Unity), not engine code, and none comes with labelled C++.

- **Nardone, Muse, Abidi, Khomh and Di Penta — [Video Game Bad Smells: What They Are and How Developers Perceive Them](https://publications.polymtl.ca/54833/)** (TOSEM 2023, [doi:10.1145/3563214](https://doi.org/10.1145/3563214)). A catalogue of 28 smells in five categories, mined from 572 forum discussions and rated by 76 professionals.
- **Borrelli, Nardone, Di Lucca, Canfora and Di Penta — [Detecting Video Game-Specific Bad Smells in Unity Projects](https://2020.msrconf.org/details/msr-2020-papers/15/Detecting-Video-Game-Specific-Bad-Smells-in-Unity-Projects)** (MSR 2020). UnityLinter detects seven smells by static analysis, with 86–100% precision and 50–100% recall on 100 projects.
- **Agrahari and Chimalakonda — [A Catalogue of Game-Specific Anti-Patterns](https://arxiv.org/abs/2006.13129)** (ISEC 2022). 11 anti-patterns mined from 229 GitHub game repositories.

The 13 smells in this project's original prompt appear to come from the first two papers: the nine with percentages resemble the survey-rated smells of Nardone et al., and the four Unity ones resemble those of Borrelli et al. I could not open either full text to confirm the names.

## 5. A warning from earlier machine-learning work

- **Di Nucci, Palomba, Tamburri, Serebrenik and De Lucia — [Detecting Code Smells using Machine Learning Techniques: Are We There Yet?](https://figshare.com/articles/Detecting_Code_Smells_using_Machine_Learning_Techniques_Are_We_There_Yet_/5786631)** (SANER 2018). Shows that earlier near-perfect results came from datasets in which smelly and clean samples had very different metric distributions, so classifiers could separate them without learning the smell. This is the same problem as the length shortcut documented in [DATASET.md](DATASET.md).

## Found but not opened

- Mesbah, El Madhoun, Al Agha and Chalouati, "Leveraging Prompt-based Large Language Models for Code Smell Detection: A Comparative Study on the MLCQ Dataset" (EIDWT 2025, [HAL](https://hal.science/hal-04881949)) — GPT-4 and LLaMA. PDF in `Papers/`.
- Ghadimi, Şahin, Demirci and Yıldız, "Benchmarking Large Language Models for Java Code Smell Detection" (2026, [doi:10.1109/isades69945.2026.11608239](https://doi.org/10.1109/isades69945.2026.11608239)) — seven LLMs on 1,000 MLCQ snippets.
- Agrahari and Chimalakonda, "What's Inside Unreal Engine? – A Curious Gaze!" (ISEC 2021, [doi:10.1145/3452383.3452404](https://doi.org/10.1145/3452383.3452404)) — an analysis of a C++ engine's code; worth reading for the engine angle.
- Bosco, Cavoto, Ungolo, Muse, Khomh, Nardone and Di Penta, "UnityLint: A Bad Smell Detector for Unity" (ICPC 2023, [doi:10.1109/icpc58990.2023.00033](https://doi.org/10.1109/icpc58990.2023.00033)).
- Öztürk, "Analyzing maintainability factors in open-source game engines: implications for game developers" (Multimedia Tools and Applications, 2025, [doi:10.1007/s11042-025-20899-8](https://doi.org/10.1007/s11042-025-20899-8)) — static analysis of open-source engines including Godot. PDF in `Papers/`.

## What this means for the project

1. **Labels are the standard everyone else meets.** The fine-tuning studies start from tool output but review it by hand before training. Tool output alone, as used here, is not treated as ground truth anywhere in this list.
2. **Small models may be enough.** The first version of Zhang et al. reported that the larger models did not outperform the small ones on this task, so the 1.3B model is a defensible choice once the data is sound.
3. **A prompted LLM plus a static tool is a strong baseline.** Two studies find the combination beats either alone. A fine-tuned model should be compared against it.
4. **Report a shortcut baseline.** Di Nucci et al. is the precedent for checking whether a trivial feature explains the result.
5. **Fixing needs real pairs.** TFix mined them from commits; the C++ case study validated fixes with reviewers. Neither generated targets synthetically.
