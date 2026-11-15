"""Portable JSONL input/output and SciER conversion."""
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(record):
    if "sentence" not in record:
        raise ValueError("SciER input needs a sentence field")
    entities = [{"text": e[0], "type": e[1]} for e in record.get("ner", [])]
    relations = [{"head": r[0], "relation": r[1], "tail": r[2]} for r in record.get("rel", [])]
    target = {"entities": entities, "relations": relations}
    return {"doc_id": record.get("doc_id", ""), "sentence": record["sentence"],
            "input_text": "Extract scientific entities and relations as JSON from the sentence: " + record["sentence"],
            "target": target, "target_json": json.dumps(target, ensure_ascii=False, sort_keys=True)}
