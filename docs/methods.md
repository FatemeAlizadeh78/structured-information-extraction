# Methods and evaluation

## Task

Input: one scientific sentence. Output: JSON with entity objects (`text`, `type`) and directed relation objects (`head`, `relation`, `tail`). Entity types are Method, Task, and Dataset. Relation labels are Part-Of, Used-For, Evaluated-With, Synonym-Of, Compare-With, SubClass-Of, Benchmark-For, SubTask-Of, and Trained-With.

The split system generates entities with the existing Gemma LoRA adapter prompted for entities only; it does not train a separate entity-only adapter. It forms all ordered pairs of distinct entities, constructs typed head/tail markers in context, and classifies each pair with a SciBERT sequence classifier plus No-Relation. Nonempty predictions above a confidence threshold are reconstructed into JSON.

## Experimental settings

| Experiment | Historical setting | Evidence in original notebook |
| --- | --- | --- |
| Qwen/Llama LoRA | 3,000 train examples, 300 dev, seed 42, rank 16, alpha 32, length 1024, 2 epochs | cells 191-199 and 237-249 |
| Gemma LoRA | 5,000 train, all 713 dev prepared, seed 42, rank 8, alpha 16, length 768, 2 epochs | cells 291-302 |
| SciBERT relation training | all train/dev sentences expanded to typed directed pairs; up to 3 negative pairs per positive, or 3 if no positives | cells 320-333 |
| SciBERT optimization | 4 epochs, learning rate 2e-5, batches 16/32, max token length 256; best model selected by dev relation macro-F1 excluding No-Relation | cells 330-333 |
| Entity-only inference | Gemma adapter epoch 2, max input length 512, max new tokens 96, repetition penalty 1.15; regex repair filters spans to sentence | cells 348-354 |
| Split threshold selection | thresholds 0.20-0.70, selected by mean F1 on normal test | cells 362-364 |
| OOD evaluation | 580 OOD sentences with normal-test-selected threshold | cells 370-377 |

Gemma's historical manual training loop prepares dev data but does not evaluate it during the loop. Its checkpoint is epoch 2, not a demonstrated dev-selected best checkpoint. The loop also steps only on full accumulation boundaries; leftover gradients are not stepped at epoch end. The portable Trainer-based runner avoids that issue and adds dev selection, so new scores may differ.

## Metric definitions

The evaluator preserves the notebook's case-folding and whitespace normalization, set-based deduplication, entity `(text,type)` matching, and relation `(head,label,tail)` matching. It does not evaluate character offsets, and relation matching does not require endpoint types. Thus the report's word “exact” means normalized tuple/set equality, not case-sensitive character-exact document spans. Relation direction matters.

It computes micro precision/recall/F1 globally and example-level averages separately. Empty gold and predicted sets receive per-example F1 of 1.0, explaining nonzero relation example-F1 for an empty baseline. The main composite is the mean of entity and relation micro-F1. Partial recall uses substring overlap of normalized spans with matching entity type; it is a permissive diagnostic, not exact extraction accuracy.

## Evaluation caveats

- **Test-set threshold tuning:** cells 362/364 choose a threshold on `gemma_entity_test_df`, then report its test score. Future experiments must tune on dev and freeze the threshold before test/OOD evaluation.
- **Saved metrics:** several notebook cells explicitly assign recorded metric dictionaries. The report and CSVs agree, but executing the consolidated notebook is not independent evidence that all underlying runs took place in that runtime.
- **Incomplete exported artifacts:** most final Gemma/SciBERT predictions and the SciBERT trained checkpoint are absent. Qwen full-test LoRA predictions exist in a nested archive and can be evaluated separately. No claim of independently reproduced final split-task metrics is made.
- **JSON validity after repair:** many predictions are repaired before scoring. A 1.0 valid-JSON rate after reconstruction does not mean the raw generative model always returned valid JSON.
- **Unequal fine-tuning budgets:** Gemma sees 5,000 training examples while Qwen/Llama see 3,000; comparisons confound architecture and training data size.
- **Entity-span ambiguity:** pair markers use the first textual occurrence. Repeated mentions and overlapping spans can fall back to a plain pair description.
- **Sentence-level scope:** document-level and cross-sentence relations are outside the experiment.
- No multi-seed confidence intervals or claim of statistical significance is provided.

The original report is retained unchanged as research provenance. This code audit adds caveats rather than silently rewriting its conclusions.
