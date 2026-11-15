"""Research functions extracted from the supplied consolidated notebook.

Cell numbers are zero-based; see docs/provenance.md. Semantics are preserved.
"""
import json
import re
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# Original notebook cell 64
KNOWN_TASK_KEYWORDS = ['classification', 'detection', 'object detection', 'segmentation', 'translation', 'summarization', 'question answering', 'named entity recognition', 'relation extraction', 'image classification', 'machine translation', 'speech recognition']

# Original notebook cell 64
KNOWN_DATASETS = ['ImageNet', 'COCO', 'SQuAD', 'MNIST', 'CIFAR', 'GLUE', 'MultiNLI', 'WikiText']

# Original notebook cell 64
def rule_based_extract(sentence):
    entities = []
    relations = []
    sentence_str = str(sentence)
    for dataset in KNOWN_DATASETS:
        if re.search(f'\\b{re.escape(dataset)}\\b', sentence_str, re.IGNORECASE):
            entities.append({'text': dataset, 'type': 'Dataset'})
    for task in KNOWN_TASK_KEYWORDS:
        if re.search(f'\\b{re.escape(task)}\\b', sentence_str, re.IGNORECASE):
            entities.append({'text': task, 'type': 'Task'})
    method_candidates = re.findall('\\b[A-Z][A-Za-z0-9\\-]*(?:Net|BERT|GPT|CNN|RNN|LSTM|Transformer)?\\b', sentence_str)
    for method in method_candidates:
        if len(method) > 2 and method.lower() not in ['the', 'this', 'that', 'using']:
            entities.append({'text': method, 'type': 'Method'})
    unique_entities = []
    seen = set()
    for ent in entities:
        key = (ent['text'].lower(), ent['type'].lower())
        if key not in seen:
            unique_entities.append(ent)
            seen.add(key)
    return {'entities': unique_entities, 'relations': relations}
