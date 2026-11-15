"""Research functions extracted from the supplied consolidated notebook.

Cell numbers are zero-based; see docs/provenance.md. Semantics are preserved.
"""
import json
import re
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# Original notebook cell 31
def parse_json_safe(text):
    if isinstance(text, dict):
        return (text, True)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj.setdefault('entities', [])
            obj.setdefault('relations', [])
            return (obj, True)
        return ({'entities': [], 'relations': []}, False)
    except Exception:
        return ({'entities': [], 'relations': []}, False)

# Original notebook cell 31
def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub('\\s+', ' ', text)
    return text

# Original notebook cell 31
def entity_set(obj):
    entities = obj.get('entities', [])
    output = set()
    if not isinstance(entities, list):
        return output
    for ent in entities:
        if isinstance(ent, dict):
            text = normalize_text(ent.get('text', ''))
            ent_type = normalize_text(ent.get('type', ''))
            if text and ent_type:
                output.add((text, ent_type))
    return output

# Original notebook cell 31
def relation_set(obj):
    relations = obj.get('relations', [])
    output = set()
    if not isinstance(relations, list):
        return output
    for rel in relations:
        if isinstance(rel, dict):
            head = normalize_text(rel.get('head', ''))
            relation = normalize_text(rel.get('relation', ''))
            tail = normalize_text(rel.get('tail', ''))
            if head and relation and tail:
                output.add((head, relation, tail))
    return output

# Original notebook cell 31
def prf_from_counts(correct, predicted, gold):
    if gold == 0 and predicted == 0:
        return (1.0, 1.0, 1.0)
    precision = correct / predicted if predicted > 0 else 0.0
    recall = correct / gold if gold > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)

# Original notebook cell 32
def evaluate_structured_predictions(df, gold_col='target_json', pred_col='prediction_text'):
    valid_json_scores = []
    exact_match_scores = []
    entity_correct_total = 0
    entity_pred_total = 0
    entity_gold_total = 0
    relation_correct_total = 0
    relation_pred_total = 0
    relation_gold_total = 0
    entity_example_precision = []
    entity_example_recall = []
    entity_example_f1 = []
    relation_example_precision = []
    relation_example_recall = []
    relation_example_f1 = []
    for _, row in df.iterrows():
        gold_obj, _ = parse_json_safe(row[gold_col])
        pred_obj, is_valid = parse_json_safe(row[pred_col])
        valid_json_scores.append(1 if is_valid else 0)
        gold_entities = entity_set(gold_obj)
        pred_entities = entity_set(pred_obj)
        gold_relations = relation_set(gold_obj)
        pred_relations = relation_set(pred_obj)
        exact_match = gold_entities == pred_entities and gold_relations == pred_relations
        exact_match_scores.append(1 if exact_match else 0)
        entity_correct = len(gold_entities & pred_entities)
        entity_correct_total += entity_correct
        entity_pred_total += len(pred_entities)
        entity_gold_total += len(gold_entities)
        relation_correct = len(gold_relations & pred_relations)
        relation_correct_total += relation_correct
        relation_pred_total += len(pred_relations)
        relation_gold_total += len(gold_relations)
        ep, er, ef1 = prf_from_counts(correct=entity_correct, predicted=len(pred_entities), gold=len(gold_entities))
        entity_example_precision.append(ep)
        entity_example_recall.append(er)
        entity_example_f1.append(ef1)
        rp, rr, rf1 = prf_from_counts(correct=relation_correct, predicted=len(pred_relations), gold=len(gold_relations))
        relation_example_precision.append(rp)
        relation_example_recall.append(rr)
        relation_example_f1.append(rf1)
    entity_micro_p, entity_micro_r, entity_micro_f1 = prf_from_counts(correct=entity_correct_total, predicted=entity_pred_total, gold=entity_gold_total)
    relation_micro_p, relation_micro_r, relation_micro_f1 = prf_from_counts(correct=relation_correct_total, predicted=relation_pred_total, gold=relation_gold_total)
    return {'valid_json_rate': float(np.mean(valid_json_scores)), 'exact_match_rate': float(np.mean(exact_match_scores)), 'entity_micro_precision': entity_micro_p, 'entity_micro_recall': entity_micro_r, 'entity_micro_f1': entity_micro_f1, 'entity_example_precision': float(np.mean(entity_example_precision)), 'entity_example_recall': float(np.mean(entity_example_recall)), 'entity_example_f1': float(np.mean(entity_example_f1)), 'relation_micro_precision': relation_micro_p, 'relation_micro_recall': relation_micro_r, 'relation_micro_f1': relation_micro_f1, 'relation_example_precision': float(np.mean(relation_example_precision)), 'relation_example_recall': float(np.mean(relation_example_recall)), 'relation_example_f1': float(np.mean(relation_example_f1))}

# Original notebook cell 41
def add_equal_weighted_final_scores(metrics):
    metrics = dict(metrics)
    metrics['final_micro_f1_equal_weighted'] = 0.5 * metrics['entity_micro_f1'] + 0.5 * metrics['relation_micro_f1']
    metrics['final_example_f1_equal_weighted'] = 0.5 * metrics['entity_example_f1'] + 0.5 * metrics['relation_example_f1']
    return metrics

# Original notebook cell 45
def add_relation_weighted_final_scores(metrics):
    metrics = dict(metrics)
    metrics['final_micro_f1_relation_weighted'] = 0.4 * metrics['entity_micro_f1'] + 0.6 * metrics['relation_micro_f1']
    metrics['final_example_f1_relation_weighted'] = 0.4 * metrics['entity_example_f1'] + 0.6 * metrics['relation_example_f1']
    return metrics

# Original notebook cell 171
def partial_entity_match(gold_entity, pred_entity):
    """
    Partial match between gold and predicted entities.
    The entity type must be the same.
    The entity text can be partially overlapping.
    """
    gold_text, gold_type = gold_entity
    pred_text, pred_type = pred_entity
    gold_text = str(gold_text).lower().strip()
    pred_text = str(pred_text).lower().strip()
    gold_type = str(gold_type).lower().strip()
    pred_type = str(pred_type).lower().strip()
    if gold_type != pred_type:
        return False
    return gold_text in pred_text or pred_text in gold_text

# Original notebook cell 48
def partial_entity_recall(gold_entities, pred_entities):
    if len(gold_entities) == 0:
        return 1.0 if len(pred_entities) == 0 else 0.0
    matched = 0
    for gold_ent in gold_entities:
        for pred_ent in pred_entities:
            if partial_entity_match(gold_ent, pred_ent):
                matched += 1
                break
    return matched / len(gold_entities)

# Original notebook cell 49
def compute_partial_entity_recall_for_df(df, gold_col='target_json', pred_col='prediction_text'):
    recalls = []
    for _, row in df.iterrows():
        gold_obj, _ = parse_json_safe(row[gold_col])
        pred_obj, _ = parse_json_safe(row[pred_col])
        gold_entities = entity_set(gold_obj)
        pred_entities = entity_set(pred_obj)
        recall = partial_entity_recall(gold_entities, pred_entities)
        recalls.append(recall)
    return sum(recalls) / len(recalls)

# Original notebook cell 50
def add_partial_entity_recall(metrics, df):
    metrics = dict(metrics)
    metrics['partial_entity_recall'] = compute_partial_entity_recall_for_df(df)
    return metrics
