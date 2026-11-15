# Scientific entity and relation extraction

**From scientific sentences to structured JSON, using instruction-tuned language models and a split-task Gemma–SciBERT pipeline.**

This research project compares prompting, parameter-efficient fine-tuning, and task decomposition on SciER. It extracts **Method**, **Task**, and **Dataset** entities, then predicts nine directed relation types. The central experiment separates entity generation with Gemma LoRA from pairwise relation classification with SciBERT.

The repository includes a runnable CPU evaluation path, optional model inference and training, recorded experimental results, and the original research notebook and report. The CPU utilities run locally; GPU experiments have separate model and hardware requirements.

```mermaid
flowchart LR
  A[Scientific sentence] --> B[Gemma LoRA]
  B --> C[Entity spans and types]
  C --> D[Directed entity pairs]
  D --> E[SciBERT relation classifier]
  E --> F[Structured JSON]
```

## Research question

Can decomposing entity extraction and relation classification improve structured scientific extraction compared with direct JSON generation by a single language model?

Experiments cover FLAN-T5-small/base, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, and Gemma-2-2B-it; zero-shot/few-shot prompting; Qwen, Llama, and Gemma LoRA adaptation; a gold-entity diagnostic; and out-of-domain evaluation. See [methods and evaluation](docs/methods.md) for details and experimental caveats.

## Quick start without a GPU

Use Python 3.11 for the documented model stack. CPU utilities support Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
scier baseline --input data/examples/synthetic.jsonl --output runs/example.jsonl
scier evaluate --input runs/example.jsonl --output runs/example-metrics.json
scier results
```

The example is a tiny **synthetic** fixture, not a SciER sample or a model demonstration. The historical rule baseline extracts candidate entities and deliberately predicts no relations. `scier results` displays saved research metrics; it does not run inference.

## Data preparation

The upstream dataset was obtained from [the SciER authors' repository](https://github.com/edzq/SciER). Acquire it separately and follow its license and usage terms. The dataset is not redistributed here. The original project used `SciER/LLM/{train,dev,test,test_ood}.jsonl`; upstream updates may differ because the original notebook did not pin a commit.

```bash
git clone https://github.com/edzq/SciER.git data/raw/SciER
scier prepare --input data/raw/SciER/SciER/LLM/train.jsonl --output data/processed/train.jsonl
scier prepare --input data/raw/SciER/SciER/LLM/dev.jsonl --output data/processed/dev.jsonl
scier prepare --input data/raw/SciER/SciER/LLM/test.jsonl --output data/processed/test.jsonl
scier prepare --input data/raw/SciER/SciER/LLM/test_ood.jsonl --output data/processed/test_ood.jsonl
```

Historical split sizes: train **5,575**, development **713**, test **854**, OOD test **580** sentences. Preserve document IDs and splits; do not mix training examples into evaluation prompts.

## Model inference

Install optional dependencies on a machine suitable for the selected model. Some models require prior acceptance of access terms. Supply credentials through `HF_TOKEN` or your local Hugging Face login; never put tokens in source files.

```bash
python -m pip install -e '.[models]'
scier extract --input data/processed/test.jsonl --output runs/qwen-zero.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct --limit 5
scier extract --input data/processed/test.jsonl --output runs/qwen-few.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct --few-shot-train data/processed/train.jsonl --shots 3 --limit 5
scier evaluate --input runs/qwen-zero.jsonl
```

Run the split-task pipeline after supplying a Gemma LoRA adapter and a **trained** SciBERT relation checkpoint:

```bash
scier extract --input data/processed/test_ood.jsonl --output runs/split-ood.jsonl \
  --model google/gemma-2-2b-it --adapter models/gemma-lora \
  --relation-model models/scibert-relations --threshold 0.5 --max-new-tokens 96
scier evaluate --input runs/split-ood.jsonl
```

**The supplied archive does not contain the trained SciBERT classifier.** The original full pipeline cannot be rerun from the archive alone. Gemma and Qwen adapters exist in the private source archives but are excluded from Git, along with multi-gigabyte checkpoints. Supply authorized local weights or train new models. A plain SciBERT base checkpoint cannot replace the trained relation classifier.

The portable inference runner makes inputs explicit and retains raw and repaired predictions. Its consolidated prompts and cleanup differ from some historical experimental variants, so it is a reproduction starting point, not a bit-identical replay of every saved score. Check [reproduction notes](docs/reproduction.md).

## Training new experiments

GPU training is opt-in. The LoRA configuration requires CUDA and bitsandbytes; it is not a CPU quick-start command.

```bash
python -m pip install -e '.[models,cuda]'
python -m scier_extraction.training --config configs/gemma-lora.json
python -m scier_extraction.training --config configs/scibert-relations.json
```

The Gemma configuration uses rank 8, alpha 16, 5,000 training sentences, 2 epochs, maximum length 768, and gradient accumulation 8. SciBERT uses directed typed entity pairs, negative sampling, and dev-based model selection. Adjust output paths to keep runs separate. The training entry points reject existing run directories. For Qwen/Llama configurations see `configs/`; the complete historical experiment variants remain in the notebook.

## Recorded results

| System | Split | Entity micro-F1 | Relation micro-F1 | Mean of the two F1s |
| --- | --- | ---: | ---: | ---: |
| Gemma few-shot | test | 0.519 | 0.066 | 0.292 |
| Qwen LoRA | test | 0.708 | 0.372 | 0.540 |
| Llama LoRA | test | 0.722 | 0.377 | 0.549 |
| Gemma LoRA direct generation | test | 0.733 | 0.379 | 0.556 |
| Gemma entities + SciBERT relations | test, threshold selected on test | 0.781 | 0.487 | 0.634 |
| Gemma entities + SciBERT relations | OOD test | 0.674 | 0.363 | 0.519 |
| Gold entities + SciBERT | diagnostic only | 1.000 | 0.747 | 0.873 |

These are **recorded results**, transcribed from the supplied CSVs and report. They have not been reproduced by retraining in this repository. The final column is `0.5 × entity micro-F1 + 0.5 × relation micro-F1`; it is not a pooled micro-F1 over all entity and relation decisions.

The source notebook selects the split-task relation threshold on the test set. Its 0.634 score therefore has test-set selection bias and should not be presented as an untouched held-out benchmark or a definitive improvement estimate. The same selected threshold is reused on OOD data. The gold-entity result is an upper-bound diagnostic, never an end-to-end result. See [the audit](docs/methods.md#evaluation-caveats).

## Repository map

| Path | Purpose |
| --- | --- |
| `src/scier_extraction/` | Conversion, historical metrics/baselines, typed-pair construction, portable inference and training |
| `configs/` | Explicit training configurations |
| `tests/` | CPU tests for metrics, relation direction, pair markers, and output filtering |
| `data/examples/` | Synthetic offline fixture |
| `results/recorded/` | Original saved metric CSVs; never mixed with new run outputs |
| `notebooks/original_experiments.ipynb` | Historical notebook, outputs and cloud metadata removed |
| `docs/project-report.pdf` | Original report; later code audit caveats are documented separately |
| `docs/` | Methods, provenance, reproduction limits, and validation |
| `.github/workflows/check.yml` | CPU tests and CLI smoke checks for GitHub Actions |

## Scope and reuse

This is a sentence-level research experiment. It does not parse arbitrary PDFs, build an autonomous agent, deploy a medical system, or provide a hosted API. Knowledge graphs, scientific assistants, and agent memory are potential applications described in the report.

The work is attributed to Fatemeh Alizadeh in the supplied presentation. Dataset, model, and third-party dependencies retain their own terms. No new blanket open-source license has been assigned to the author's research materials; see [NOTICE](NOTICE.md). Original archives remain outside this repository and are listed in the private review workspace.
