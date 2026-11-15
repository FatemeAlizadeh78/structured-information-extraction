# Validation performed

Verified locally on 2026-11-15 using a fresh virtual environment with Python 3.12. The recommended historical model environment is Python 3.11; GPU/model extras were not installed or exercised.

## Completed

- Editable installation of the base package succeeded in a clean virtual environment.
- A distributable Python wheel built successfully with `pip wheel --no-deps`; the build output is retained under ignored `runs/wheels/`.
- `pip check` reported no broken requirements.
- Four CPU tests passed: exact/invalid-JSON scoring, directed relations, typed pair generation and negative examples, and filtering hallucinated entities/dangling relations.
- Synthetic baseline and evaluation CLI completed, writing JSONL predictions and JSON metrics.
- Recorded-results CLI completed and reports saved result semantics explicitly.
- Qwen LoRA's supplied 854 full-test predictions were re-scored with the extracted evaluator. Entity micro-F1 0.7080720210, relation micro-F1 0.3719193428, and equal-weighted composite 0.5399956819 match the corresponding original CSV to floating-point precision. The re-scored metrics are retained in `results/recomputed/qwen-lora-metrics.json`; the dataset-containing prediction file remains private.
- Package source compiled successfully.
- Original notebook has 398 cells; public copy has zero outputs and zero execution counts. Cloud metadata was removed.
- No email-pattern strings or live Hugging Face token strings were found in the public notebook. Report metadata contains only document-generation software and a creation timestamp, not personal contact details or student identifiers.
- Large checkpoints, datasets, predictions containing dataset text, environment files, generated runs, and virtual environments are ignored by Git.
- Final Git candidate audit found 46 files, each below 10 MB, with build artifacts, private data, and virtual environments excluded. Local Git was initialized with no remote and no commit.

## Not performed

- No model training, GPU inference, full historical notebook execution, or final split-task metric reproduction.
- No claim that the supplied final SciBERT classifier can be recovered; it is absent from the archive.
- No public GitHub repository, commit, remote, push, or hosted service was created.

GitHub Actions configuration is included for CPU installation, tests, baseline/evaluation smoke checks, and recorded-results display. It has not yet run remotely.
