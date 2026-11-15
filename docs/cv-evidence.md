# Evidence for a research CV

Suggested project description:

**Scientific information extraction with LoRA and task decomposition** — compared zero-shot/few-shot and parameter-efficient adaptation of Qwen2.5, Llama 3.2, and Gemma 2 on SciER; combined Gemma entity extraction with typed-pair SciBERT relation classification, evaluated structured outputs and out-of-domain generalization, and analyzed entity boundaries, relation direction, and parsing failures.

This is sentence-level scientific entity/relation extraction, not extraction of arbitrary objects from whole PDFs. Agent memory, scientific assistants, and RAG are downstream motivations; an autonomous agent was not implemented in the supplied work.

The presentation identifies Fatemeh Alizadeh as author, June 2026. The source does not establish an external publication or research employment. Avoid claiming sole contribution beyond what the candidate confirms.

## Evidence locations

- Report pp. 1-4: objective, SciER splits, schema, entity/relation types, metrics.
- Report pp. 5-8 and notebook cells 159-314: prompting and LoRA model experiments.
- Report pp. 6-7 and notebook cells 315-366: SciBERT and split-task architecture.
- Report pp. 9-12: recorded test/OOD results and error analysis.
- Notebook cells 362/364: threshold tuning on test, which qualifies the reported 0.634 composite score.
- Notebook cells 370-377: OOD inference using that selected threshold; recorded composite 0.5187.
- Notebook cells 339/343: gold-entity experiment, a diagnostic and not end-to-end performance.

Prefer describing the methods and evaluation work over an unqualified “achieved 0.634 held-out F1” claim. The score is both test-selected and a mean of two F1s. The independently re-scored Qwen LoRA saved predictions give entity micro-F1 0.708072, relation micro-F1 0.371919, and mean 0.539996; this re-scoring does not rerun the model.
