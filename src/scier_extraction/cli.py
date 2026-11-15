"""Explicit commands for preparation, extraction, evaluation, and saved results."""
import argparse
import csv
import json
from pathlib import Path
import pandas as pd
from .io import read_jsonl, write_jsonl, convert
from .metrics import evaluate_structured_predictions, add_equal_weighted_final_scores, add_relation_weighted_final_scores, add_partial_entity_recall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert an upstream SciER/LLM JSONL split")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    baseline = commands.add_parser("baseline", help="Run empty or historical rule baseline")
    baseline.add_argument("--input", required=True, type=Path)
    baseline.add_argument("--output", required=True, type=Path)
    baseline.add_argument("--method", choices=["empty", "rules"], default="rules")
    evaluate = commands.add_parser("evaluate", help="Evaluate JSONL containing target_json and prediction_text")
    evaluate.add_argument("--input", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)
    predict = commands.add_parser("extract", help="Explicit model inference; downloads model weights")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--model", default="google/gemma-2-2b-it")
    predict.add_argument("--adapter", type=Path)
    predict.add_argument("--relation-model", type=Path)
    predict.add_argument("--few-shot-train", type=Path)
    predict.add_argument("--shots", type=int, default=3)
    predict.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--max-new-tokens", type=int, default=256)
    predict.add_argument("--device", default="auto")
    summary = commands.add_parser("results", help="Show recorded metrics; does not rerun models")
    summary.add_argument("--directory", type=Path, default=Path("results/recorded"))
    args = parser.parse_args()
    if args.command == "prepare":
        write_jsonl(args.output, [convert(row) for row in read_jsonl(args.input)])
    elif args.command == "baseline":
        from .baselines import rule_based_extract
        rows = read_jsonl(args.input)
        for row in rows:
            result = rule_based_extract(row["sentence"]) if args.method == "rules" else {"entities": [], "relations": []}
            row["prediction_text"] = json.dumps(result)
        write_jsonl(args.output, rows)
    elif args.command == "evaluate":
        rows = read_jsonl(args.input)
        if not rows:
            parser.error("Input is empty")
        if any("target_json" not in row or "prediction_text" not in row for row in rows):
            parser.error("Every row needs target_json and prediction_text")
        frame = pd.DataFrame(rows)
        result = add_partial_entity_recall(add_relation_weighted_final_scores(
            add_equal_weighted_final_scores(evaluate_structured_predictions(frame))), frame)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text)
    elif args.command == "extract":
        if not 0 <= args.threshold <= 1 or args.shots < 0 or (args.limit is not None and args.limit < 1):
            parser.error("Threshold must be in [0,1], shots nonnegative, limit positive")
        from .inference import extract
        rows = read_jsonl(args.input)
        examples = read_jsonl(args.few_shot_train)[:args.shots] if args.few_shot_train else []
        extract(rows[:args.limit] if args.limit else rows, args.output, args.model, args.adapter,
                args.relation_model, examples, args.threshold, args.max_new_tokens, args.device)
    else:
        paths = sorted(args.directory.glob("*metrics.csv"))
        if not paths:
            parser.error("No recorded metric CSVs found")
        print("RECORDED RESULTS - not a new inference run")
        print("Caveat: split-task test threshold was selected on test; gold-entity scores are diagnostic only.")
        for path in paths:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    print(f"{row['model']} | {row['split']} | entity={float(row['entity_micro_f1']):.3f} relation={float(row['relation_micro_f1']):.3f} average={float(row['final_micro_f1_equal_weighted']):.3f}")


if __name__ == "__main__":
    main()
