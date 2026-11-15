"""Portable inference runner. GPU/model execution is optional and explicit.

The notebook remains the historical experiment reference. This runner makes
paths and dependencies explicit; it is not a claim of bit-identical reproduction.
"""
import json
import os
from pathlib import Path
from .io import write_jsonl
from .pairs import ENTITY_TYPES, RELATION_TYPES, make_pair_text


def user_prompt(sentence, entity_only=False):
    task = "entities" if entity_only else "entities and directed relations"
    schema = {"entities": [{"text": "exact span", "type": "Method, Task, or Dataset"}]}
    if not entity_only:
        schema["relations"] = [{"head": "entity text", "relation": "allowed label", "tail": "entity text"}]
    return (f"Extract scientific {task}. Return only JSON.\n"
            f"Entity types: {', '.join(ENTITY_TYPES)}.\n"
            f"Relation types: {', '.join(RELATION_TYPES[1:])}.\n"
            f"Schema: {json.dumps(schema)}\n"
            "Copy exact spans from the sentence. Do not invent entities. "
            "Relation endpoints must appear in entities.\nSentence:\n" + sentence)


def repair(text, sentence, entity_only=False):
    # Keep raw output beside repaired output; JSON validity after repair is not
    # raw model schema compliance. Like the source, recover entity objects.
    import re
    start, end = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[start:end + 1])
        if not isinstance(obj, dict):
            obj = {}
    except (ValueError, TypeError):
        pairs = re.findall(r'"text"\s*:\s*"([^"\n]+)"\s*,\s*"type"\s*:\s*"([^"\n]+)"', text)
        pairs += [(t, k) for k, t in re.findall(r'"type"\s*:\s*"([^"\n]+)"\s*,\s*"text"\s*:\s*"([^"\n]+)"', text)]
        obj = {"entities": [{"text": t, "type": k} for t, k in pairs]}
    entities, seen = [], set()
    source_entities = obj.get("entities", [])
    for e in source_entities if isinstance(source_entities, list) else []:
        if not isinstance(e, dict):
            continue
        key = (str(e.get("text", "")).strip(), str(e.get("type", "")).strip())
        if key[0] and key[0] in sentence and key[1] in ENTITY_TYPES and key not in seen:
            entities.append({"text": key[0], "type": key[1]})
            seen.add(key)
    endpoints = {e["text"] for e in entities}
    relations = []
    source_relations = obj.get("relations", [])
    for r in source_relations if isinstance(source_relations, list) else []:
        if isinstance(r, dict) and r.get("relation") in RELATION_TYPES[1:] and r.get("head") in endpoints and r.get("tail") in endpoints:
            if r not in relations:
                relations.append({k: r[k] for k in ("head", "relation", "tail")})
    return {"entities": entities, "relations": [] if entity_only else relations}


def extract(rows, output, model_name, adapter=None, relation_model=None,
            examples=None, threshold=0.5, max_new_tokens=256, device="auto"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
    token = os.environ.get("HF_TOKEN")
    config = AutoConfig.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    kwargs = {"token": token, "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if device == "auto":
        kwargs["device_map"] = "auto"
    if "gemma" in model_name.lower():
        kwargs["attn_implementation"] = "eager"
    model = cls.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model.to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    relation = RelationClassifier(relation_model) if relation_model else None
    predictions = []
    for index, row in enumerate(rows):
        sentence = row["sentence"]
        prompt = user_prompt(sentence, entity_only=bool(relation))
        messages = []
        for example in examples or []:
            messages.extend([{"role": "user", "content": user_prompt(example["sentence"])},
                             {"role": "assistant", "content": example["target_json"]}])
        messages.append({"role": "user", "content": prompt})
        if tokenizer.chat_template and not config.is_encoder_decoder:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif examples:
            prompt = "\n".join(m["content"] for m in messages)
        batch = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512 if relation or config.is_encoder_decoder else 2048).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=max_new_tokens,
                                       do_sample=False, repetition_penalty=1.15 if relation else 1.1,
                                       pad_token_id=tokenizer.pad_token_id)
        ids = generated[0] if config.is_encoder_decoder else generated[0][batch["input_ids"].shape[-1]:]
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        result = repair(raw, sentence, entity_only=bool(relation))
        if relation:
            result["relations"] = relation.predict(sentence, result["entities"], threshold)
        predictions.append({**row, "prediction_text_raw": raw, "prediction_text": json.dumps(result)})
        if (index + 1) % 20 == 0:
            write_jsonl(str(output) + ".partial.jsonl", predictions)
    write_jsonl(output, predictions)


class RelationClassifier:
    def __init__(self, checkpoint):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        if set(self.model.config.id2label.values()) != set(RELATION_TYPES):
            raise ValueError("Relation checkpoint must contain SciER labels; a plain SciBERT base is not sufficient")

    def predict(self, sentence, entities, threshold):
        relations = []
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j:
                    continue
                text = make_pair_text(sentence, head["text"], head["type"], tail["text"], tail["type"])
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.model.device)
                with self.torch.inference_mode():
                    probs = self.torch.softmax(self.model(**inputs).logits, dim=-1)[0]
                confidence, label_id = probs.max(dim=-1)
                label = self.model.config.id2label[int(label_id)]
                if label != "No-Relation" and float(confidence) >= threshold:
                    result = {"head": head["text"], "relation": label, "tail": tail["text"]}
                    if result not in relations:
                        relations.append(result)
        return relations
