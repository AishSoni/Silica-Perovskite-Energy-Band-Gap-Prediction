import unittest

import pandas as pd

from download_data import build_search_kwargs
from src.data_io import select_feature_columns
from src.pipeline_config import load_config
from src.preprocess import split_preprocess_data
from src.validate_dataset import validate_record


class PipelineSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("experiments/query_config.yaml")

    def test_config_builds_expected_search_kwargs(self):
        kwargs = build_search_kwargs(self.config)
        self.assertEqual(kwargs["num_elements"], (3, 5))
        self.assertEqual(kwargs["num_sites"], (8, 40))
        self.assertEqual(kwargs["energy_above_hull"], (None, 0.2))
        self.assertIn("formula_anonymous", kwargs["fields"])
        self.assertNotIn("elements", kwargs)

    def test_validation_rejects_formula_only_false_positive(self):
        record = {
            "material_id": "mp-test",
            "formula_pretty": "AgBi(PS3)2",
            "formula_anonymous": "ABC2D6",
            "elements": "Ag,Bi,P,S",
            "band_gap": 1.2,
            "energy_above_hull": 0.01,
            "nsites": 10,
            "deprecated": False,
            "structure_json": '{"lattice": {}}',
        }
        accepted, reasons = validate_record(record, self.config)
        self.assertFalse(accepted)
        self.assertIn("missing_x_site_element", reasons)

    def test_task_feature_selection_excludes_leakage_columns(self):
        df = pd.DataFrame(
            {
                "material_id": ["m1", "m2"],
                "formula_pretty": ["A", "B"],
                "band_gap": [1.0, 2.0],
                "is_gap_direct": [True, False],
                "density": [4.0, 5.0],
            }
        )
        self.assertNotIn("band_gap", select_feature_columns(df, task="classification"))
        self.assertNotIn("is_gap_direct", select_feature_columns(df, task="regression"))

    def test_preprocess_splits_before_fitting_artifacts(self):
        X = pd.DataFrame(
            {
                "density": [1.0, 2.0, 3.0, None, 5.0, 6.0],
                "volume": [6.0, 5.0, None, 3.0, 2.0, 1.0],
            }
        )
        y = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], name="band_gap")
        result = split_preprocess_data(
            X=X,
            y=y,
            feature_names=["density", "volume"],
            task="regression",
            test_size=0.33,
            random_state=42,
            apply_smote=False,
        )
        self.assertEqual(result["split_manifest"]["n_features"], 2)
        self.assertEqual(result["X_test"].shape[1], 2)
        self.assertIsNotNone(result["imputer"])


if __name__ == "__main__":
    unittest.main()

