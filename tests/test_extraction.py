import json
import unittest
import pandas as pd
from scier_extraction.io import convert
from scier_extraction.metrics import evaluate_structured_predictions, add_equal_weighted_final_scores
from scier_extraction.pairs import mark_entity_pair_in_sentence, build_relation_examples_from_df
from scier_extraction.inference import repair


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.row=convert({"doc_id":"synthetic-1","sentence":"AtlasNet is used for classification.",
                          "ner":[["AtlasNet","Method"],["classification","Task"]],
                          "rel":[["AtlasNet","Used-For","classification"]]})

    def test_exact_prediction_and_invalid_json(self):
        frame=pd.DataFrame([{**self.row,"prediction_text":self.row["target_json"]}])
        metrics=add_equal_weighted_final_scores(evaluate_structured_predictions(frame))
        self.assertEqual(metrics["final_micro_f1_equal_weighted"],1)
        frame["prediction_text"]="broken"
        metrics=evaluate_structured_predictions(frame)
        self.assertEqual(metrics["valid_json_rate"],0)
        self.assertEqual(metrics["entity_micro_f1"],0)

    def test_relations_are_directed(self):
        obj=json.loads(self.row["target_json"])
        obj["relations"][0].update(head="classification",tail="AtlasNet")
        metrics=evaluate_structured_predictions(pd.DataFrame([{**self.row,"prediction_text":json.dumps(obj)}]))
        self.assertEqual(metrics["entity_micro_f1"],1)
        self.assertEqual(metrics["relation_micro_f1"],0)

    def test_pair_markers_and_negative_pairs(self):
        text=mark_entity_pair_in_sentence(self.row["sentence"],"AtlasNet","Method","classification","Task")
        self.assertIn("[HEAD_Method] AtlasNet [/HEAD_Method]",text)
        self.assertIn("[TAIL_Task] classification [/TAIL_Task]",text)
        pairs=build_relation_examples_from_df(pd.DataFrame([self.row]))
        self.assertEqual(set(pairs["label_name"]),{"Used-For","No-Relation"})

    def test_repair_rejects_hallucinations_and_dangling_relations(self):
        output='{"entities":[{"text":"invented","type":"Method"},{"text":"AtlasNet","type":"Method"}],"relations":[{"head":"AtlasNet","relation":"Used-For","tail":"invented"}]}'
        obj=repair(output,self.row["sentence"])
        self.assertEqual(obj["entities"],[{"text":"AtlasNet","type":"Method"}])
        self.assertEqual(obj["relations"],[])


if __name__=="__main__":
    unittest.main()
