"""Configurable GPU training entry points adapted from notebook experiments.

The source notebook preserves historical loops. This runner uses Trainer for
both tasks, fixes leftover-gradient handling, and does not recreate exact saved
scores. Tune only on dev data and evaluate once on test.
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from .io import read_jsonl


def train_relations(config):
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    from .pairs import RELATION_TYPES, ENTITY_TYPES, build_relation_examples_from_df
    set_seed(config["seed"])
    train = build_relation_examples_from_df(pd.DataFrame(read_jsonl(config["train"])), random_state=config["seed"])
    dev = build_relation_examples_from_df(pd.DataFrame(read_jsonl(config["dev"])), random_state=config["seed"] + 1)
    names = {i: label for i, label in enumerate(RELATION_TYPES)}
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model"], num_labels=len(names), id2label=names, label2id={v:k for k,v in names.items()})
    tokenizer.add_special_tokens({"additional_special_tokens": [f"[{end}{role}_{kind}]" for kind in ENTITY_TYPES for role in ("HEAD", "TAIL") for end in ("", "/")]})
    model.resize_token_embeddings(len(tokenizer))
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    def dataset(frame):
        return Dataset.from_pandas(frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True, remove_columns=["text"])
    def metrics(pair):
        logits, gold = pair
        return {"relation_macro_f1_excl_no_relation": f1_score(gold, np.argmax(logits, axis=-1), labels=list(range(1,len(names))), average="macro", zero_division=0)}
    args = TrainingArguments(output_dir=config["output"], num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
                             per_device_train_batch_size=16, per_device_eval_batch_size=32,
                             eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="relation_macro_f1_excl_no_relation", greater_is_better=True,
                             fp16=torch.cuda.is_available(), save_total_limit=2, report_to="none", seed=config["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=dataset(train), eval_dataset=dataset(dev),
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(config["output"])
    tokenizer.save_pretrained(config["output"])


def train_lora(config):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from .inference import user_prompt
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training configuration requires a CUDA GPU and the cuda extra")
    set_seed(config["seed"])
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(config["model"], token=token, quantization_config=quantization,
                                              device_map={"":0}, attn_implementation="eager")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    def make_dataset(path, size):
        frame = pd.DataFrame(read_jsonl(path)).sample(n=min(size,len(read_jsonl(path))),random_state=config["seed"])
        texts = []
        for row in frame.to_dict("records"):
            messages = [{"role":"user","content":user_prompt(row["sentence"])},
                        {"role":"assistant","content":row["target_json"]}]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        ds = Dataset.from_dict({"text":texts})
        return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=config["max_length"]), batched=True, remove_columns=["text"])
    def collate(features):
        batch=tokenizer.pad(features,padding=True,return_tensors="pt")
        labels=batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0]=-100
        batch["labels"]=labels
        return batch
    args = TrainingArguments(output_dir=config["output"], num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
                             per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                             eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
                             fp16=True, report_to="none", seed=config["seed"], optim="adamw_torch")
    trainer=Trainer(model=model,args=args,train_dataset=make_dataset(config["train"],config["train_size"]),
                    eval_dataset=make_dataset(config["dev"],config["dev_size"]),data_collator=collate)
    trainer.train()
    trainer.save_model(config["output"])
    tokenizer.save_pretrained(config["output"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if Path(config["output"]).exists():
        parser.error("Output directory already exists; choose a fresh run path")
    if config["task"] not in ("lora","relations"):
        parser.error("Task must be lora or relations")
    if Path(config["train"]).resolve() == Path(config["dev"]).resolve():
        parser.error("Train and dev paths must differ")
    (train_lora if config["task"]=="lora" else train_relations)(config)
    Path(config["output"],"run-config.json").write_text(json.dumps(config,indent=2)+"\n")


if __name__=="__main__":
    main()
