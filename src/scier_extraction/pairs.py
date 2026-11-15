"""Research functions extracted from the supplied consolidated notebook.

Cell numbers are zero-based; see docs/provenance.md. Semantics are preserved.
"""
import json
import re
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .metrics import parse_json_safe
# Original notebook cell 320
RELATION_TYPES = ['No-Relation', 'Part-Of', 'Used-For', 'Evaluated-With', 'Synonym-Of', 'Compare-With', 'SubClass-Of', 'Benchmark-For', 'SubTask-Of', 'Trained-With']

# Original notebook cell 320
label2id = {label: i for i, label in enumerate(RELATION_TYPES)}

# Original notebook cell 320
id2label = {i: label for label, i in label2id.items()}

# Original notebook cell 320
ENTITY_TYPES = ['Method', 'Task', 'Dataset']

# Original notebook cell 322
def get_entities_and_relations(row):
    obj, _ = parse_json_safe(row['target_json'])
    entities = obj.get('entities', [])
    relations = obj.get('relations', [])
    clean_entities = []
    seen_entities = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get('text', '')).strip()
        ent_type = str(ent.get('type', '')).strip()
        if not text or ent_type not in ENTITY_TYPES:
            continue
        key = (text, ent_type)
        if key not in seen_entities:
            clean_entities.append({'text': text, 'type': ent_type})
            seen_entities.add(key)
    clean_relations = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        head = str(rel.get('head', '')).strip()
        relation = str(rel.get('relation', '')).strip()
        tail = str(rel.get('tail', '')).strip()
        if head and tail and (relation in RELATION_TYPES):
            clean_relations.append({'head': head, 'relation': relation, 'tail': tail})
    return (clean_entities, clean_relations)

# Original notebook cell 324
def mark_entity_pair_in_sentence(sentence, head_text, head_type, tail_text, tail_type):
    sentence = str(sentence)
    head_start = sentence.find(head_text)
    tail_start = sentence.find(tail_text)
    if head_start == -1 or tail_start == -1:
        return None
    head_end = head_start + len(head_text)
    tail_end = tail_start + len(tail_text)
    if not (head_end <= tail_start or tail_end <= head_start):
        return None
    spans = [{'start': head_start, 'end': head_end, 'start_marker': f'[HEAD_{head_type}]', 'end_marker': f'[/HEAD_{head_type}]'}, {'start': tail_start, 'end': tail_end, 'start_marker': f'[TAIL_{tail_type}]', 'end_marker': f'[/TAIL_{tail_type}]'}]
    spans = sorted(spans, key=lambda x: x['start'], reverse=True)
    marked_sentence = sentence
    for span in spans:
        marked_sentence = marked_sentence[:span['start']] + span['start_marker'] + ' ' + marked_sentence[span['start']:span['end']] + ' ' + span['end_marker'] + marked_sentence[span['end']:]
    return marked_sentence

# Original notebook cell 324
def make_pair_text(sentence, head, head_type, tail, tail_type):
    marked_sentence = mark_entity_pair_in_sentence(sentence, head, head_type, tail, tail_type)
    if marked_sentence is not None:
        return f'Sentence: {marked_sentence}\nHead entity type: {head_type}\nTail entity type: {tail_type}\nClassify the directed relation from HEAD to TAIL.'
    return f'Sentence: {sentence}\nHead entity: {head} [type: {head_type}]\nTail entity: {tail} [type: {tail_type}]\nClassify the directed relation from head entity to tail entity.'

# Original notebook cell 326
def build_relation_examples_from_df(df, max_negative_per_positive=3, max_negative_if_no_positive=3, random_state=42):
    rng = random.Random(random_state)
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row['sentence']
        entities, relations = get_entities_and_relations(row)
        rel_lookup = {}
        for rel in relations:
            rel_lookup[rel['head'], rel['tail']] = rel['relation']
        positive_examples = []
        negative_examples = []
        for i, head_ent in enumerate(entities):
            for j, tail_ent in enumerate(entities):
                if i == j:
                    continue
                head = head_ent['text']
                tail = tail_ent['text']
                label_name = rel_lookup.get((head, tail), 'No-Relation')
                example = {'text': make_pair_text(sentence, head, head_ent['type'], tail, tail_ent['type']), 'label': label2id[label_name], 'label_name': label_name}
                if label_name == 'No-Relation':
                    negative_examples.append(example)
                else:
                    positive_examples.append(example)
        examples.extend(positive_examples)
        if len(positive_examples) > 0:
            max_negatives = max_negative_per_positive * len(positive_examples)
            if len(negative_examples) > max_negatives:
                negative_examples = rng.sample(negative_examples, max_negatives)
            examples.extend(negative_examples)
        else:
            if len(negative_examples) > max_negative_if_no_positive:
                negative_examples = rng.sample(negative_examples, max_negative_if_no_positive)
            examples.extend(negative_examples)
    return pd.DataFrame(examples)
