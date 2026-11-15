# Source review and provenance

The supplied Google Drive export consists of three archives named `NLP_25_26-20261115T210339Z-1-001.zip`, `...-002.zip`, and `...-003.zip`. They are separate ZIP files with complementary members, not split ZIP volumes. Paths were checked before selective extraction. All 76 file members were inventoried, including model weights, optimizer states, tokenizers, predictions, metrics, two PDFs, and the consolidated 398-cell notebook. Nested Qwen adapter/output ZIPs were inventoried too. No pickle-based checkpoint was executed.

The original archives remain in the user's Downloads directory. Review copies of the readable files, an exact member inventory, extracted report text, and slide contact sheets remain in the main thesis workspace's ignored `sources/project-review/nlp/` directory. Large weight tensors and optimizer state were inventoried, not deserialized or republished.

## Included research artifacts

- `docs/project-report.pdf`: exact original 16-page report, copied without content changes.
- `notebooks/original_experiments.ipynb`: original cell source preserved, except authentication token assignment uses `HF_TOKEN`; all outputs, execution counts, per-cell metadata, and cloud notebook metadata removed. Historical paths and ordering remain, and are documented as limitations.
- `results/recorded/*.csv`: original recorded metrics and comparison tables copied without re-estimation. The cleaned comparison table and per-model files include the final metrics; the earlier comparison table is incomplete.
- The presentation was reviewed visually across all 26 pages. It repeats the methods/results, identifies Fatemeh Alizadeh and June 2026, and includes artwork and a presentation-service QR code. It is retained privately rather than republished because artwork redistribution terms are not supplied.

## Code extraction

| Module | Original zero-based notebook cells |
| --- | --- |
| `metrics.py` | 31, 32, 41, 45, 171, 48, 49, 50 |
| `baselines.py` | 64 |
| `pairs.py` | 320, 322, 324, 326 |
| `io.py` | Portable adaptation of cell 12 |
| `inference.py` | Portable adaptation of generation, repair, typed-pair inference, and split-task cells |
| `training.py` | New explicit entry points adapted from LoRA and relation training cells; training-loop differences documented |

AST extraction keeps research functions and constants without executing notebook setup, model downloads, training, shell commands, or saved output. Notebook outputs were reviewed privately and are not published. No live Hugging Face token string was found in the source notebook. The original notebook contains placeholder credentials; public authentication uses the environment.

See `source-manifest.json` for source hashes and counts, and `validation.md` for checks actually run.
