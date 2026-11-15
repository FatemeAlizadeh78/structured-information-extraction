# Reproduction guide

## Three distinct paths

1. **Offline functionality:** install base dependencies, run tests, baseline prediction, and metric calculation on the synthetic fixture. No model downloads or GPU required.
2. **Historical evidence review:** inspect recorded CSVs, report, and notebook. These are saved evidence, not newly reproduced runs. The Qwen LoRA full-test predictions in the private nested ZIP can be re-scored locally with `scier evaluate`.
3. **New model experiments:** acquire the authorized SciER version and base models, train models or supply compatible checkpoints, and use the portable inference runner. These experiments can require hours and a CUDA GPU. They were not run as part of repository preparation.

## Environment

Use Python 3.11 and the optional model versions in `pyproject.toml`, derived from the notebook's Transformers 4.51.3, Accelerate 1.6.0, and Datasets 3.5.1 installs. PyTorch must match the target CUDA driver. The CPU path is tested independently of model extras. The model stack is not a fully reproduced hardware lockfile; the original runs span Kaggle and Colab sessions with differing bitsandbytes installs. PEFT 0.15.2 is a compatibility choice for this package, not a claim that every historical run used it.

## Data and checkpoints

The original SciER processed split files and available predictions are kept privately. No dataset text, private home paths, multi-gigabyte model files, or optimizer state are required in the public repository. Do not commit downloaded models or access tokens.

The source export includes partial FLAN-T5 weights, Gemma epoch adapters, and a Qwen adapter archive. It does **not** include the final trained SciBERT classifier or every final system's predictions. Full historical reproduction is therefore incomplete. The training commands provide a way to create new compatible checkpoints; a new run does not establish the provenance of old scores.

## Evaluation protocol for new runs

Keep original document-level splits, log the upstream dataset commit and base-model revisions, and record versions and seeds. Use development data to choose threshold and model configuration; freeze those choices before test. Save raw and repaired outputs and distinguish raw parsing rate from postprocessed JSON validity. Use several seeds and report uncertainty before making superiority claims.

The historical notebook is an exploration record with Colab/Drive setup, repeated definitions, cloud paths, and some metrics entered as dictionaries. It is intentionally preserved as an archive rather than advertised as a clean top-to-bottom local pipeline. The package commands are the portable path.

## Publication

This directory is its own Git repository. Before a first public push, review `git status --short`, inspect the staged diff, choose a license you control, and set the intended remote. No commits or remotes have been created automatically. Publishing the directory does not deploy a web service.
