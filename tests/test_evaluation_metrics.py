"""All valid matched labels contribute, while teacher NLL always uses all rows."""
import unittest

import numpy as np
import pandas as pd

from opensim.evaluation import metrics, semantic


def fixture():
    rows, teacher = [], []
    for event, counts, losses in (("a", (2, 1), (1, 2, 100)), ("b", (1, 3), (1, 2, 3, 4))):
        position = 0
        for step, count in enumerate(counts, 1):
            for index in range(count):
                row = {"event": event, "time_step": f"t{step}"}
                for dimension, labels in semantic.LABELS.items():
                    row[f"real_{dimension}"] = labels[0]
                    row[f"gen_{dimension}"] = labels[index % min(2, len(labels))]
                rows.append(row)
                teacher.append({"event": event, "content_nll": losses[position]})
                position += 1
    return pd.DataFrame(rows), pd.DataFrame(teacher)


class EvaluationMetricsTests(unittest.TestCase):
    def test_closed_form_binary_metrics_keep_bits_and_class_order(self):
        frame = pd.DataFrame({"time_step": ["t1", "t1"], "real_rumor": ["spread", "spread"], "gen_rumor": ["spread", "counter"]})
        actual = metrics.compute_stepwise_micro_for_dim(frame, "rumor")
        self.assertEqual(actual["Support"], 2)
        self.assertAlmostEqual(actual["KL Div"], 1, places=6)
        self.assertAlmostEqual(actual["Wass Dist"], .5, places=6)
        self.assertAlmostEqual(actual["Macro F1"], 1 / 3)
        self.assertEqual(actual["Micro F1"], .5)
        self.assertEqual(actual["DTW"], 0)

    def test_all_valid_rows_and_singleton_steps_are_retained(self):
        frame, teacher = fixture()
        events, overall = metrics.micro_tables(frame, teacher)
        self.assertTrue(overall["Support"].eq(7).all())
        self.assertTrue(events[events["Event"].eq("a")]["Support"].eq(3).all())
        self.assertTrue(events[events["Event"].eq("b")]["Support"].eq(4).all())
        self.assertTrue(np.allclose(events[events["Event"].eq("a")]["NLL Loss"], 103 / 3))
        self.assertTrue(np.allclose(overall["NLL Loss"], 113 / 7))

    def test_overall_pools_relative_step_distributions_not_event_means(self):
        frame = pd.DataFrame({
            "time_step": ["t1", "t1", "t1", "t2"],
            "real_rumor": ["spread"] * 4,
            "gen_rumor": ["spread", "spread", "counter", "counter"],
        })
        actual = metrics.compute_stepwise_micro_for_dim(frame, "rumor")
        first = metrics.compute_stepwise_micro_for_dim(frame.iloc[:3], "rumor")
        singleton = metrics.compute_stepwise_micro_for_dim(frame.iloc[3:], "rumor")
        self.assertEqual(actual["Support"], 4)
        self.assertEqual(singleton["Support"], 1)
        self.assertAlmostEqual(actual["KL Div"], (first["KL Div"] + singleton["KL Div"]) / 2)
        self.assertEqual(actual["Micro F1"], .5)

    def test_zero_support_is_na_with_full_teacher_nll(self):
        frame, teacher = fixture()
        for dimension in semantic.DIMENSIONS:
            frame[f"gen_{dimension}"] = ""
        events, overall = metrics.micro_tables(frame, teacher)
        for table in (events, overall):
            self.assertTrue(table["Support"].eq(0).all())
            self.assertTrue(table[["KL Div", "Wass Dist", "DTW", "Macro F1", "Micro F1"]].isna().all().all())
            self.assertTrue(table["NLL Loss"].notna().all())
        self.assertTrue(np.allclose(overall["NLL Loss"], 113 / 7))

    def test_invalid_labels_excluded_per_dimension_not_teacher_nll(self):
        frame, teacher = fixture()
        frame.loc[0, "gen_rumor"] = "illegal"
        _, overall = metrics.micro_tables(frame, teacher)
        self.assertEqual(overall.set_index("Dimension").loc["rumor", "Support"], 6)
        self.assertEqual(overall.set_index("Dimension").loc["stance", "Support"], 7)
        self.assertTrue(np.allclose(overall["NLL Loss"], 113 / 7))

    def test_empty_semantic_rows_return_na_and_full_teacher_nll(self):
        frame, teacher = fixture()
        events, overall = metrics.micro_tables(frame.iloc[:0], teacher)
        self.assertTrue(events.empty)
        self.assertTrue(overall["Support"].eq(0).all())
        self.assertTrue(overall["KL Div"].isna().all())
        self.assertTrue(np.allclose(overall["NLL Loss"], 113 / 7))


if __name__ == "__main__":
    unittest.main()
