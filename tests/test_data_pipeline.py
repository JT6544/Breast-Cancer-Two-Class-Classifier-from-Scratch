from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from breast_cancer_classifier import (
    BENIGN_LABEL,
    DEFAULT_DATA_PATH,
    FEATURE_NAMES,
    MALIGNANT_LABEL,
    StandardScaler,
    grouped_stratified_split,
    load_dataset,
)
from scripts.prepare_data import write_prepared_csv


class DataPipelineTests(unittest.TestCase):
    def test_official_dataset_is_parsed_with_correct_class_semantics(self) -> None:
        dataset = load_dataset(DEFAULT_DATA_PATH)

        self.assertEqual(dataset.features.shape, (699, 8))
        self.assertEqual(tuple(dataset.feature_names), FEATURE_NAMES)
        self.assertTrue(np.isfinite(dataset.features).all())
        self.assertEqual(dataset.features.min(), 1)
        self.assertEqual(dataset.features.max(), 10)
        self.assertEqual(np.sum(dataset.labels == BENIGN_LABEL), 458)
        self.assertEqual(np.sum(dataset.labels == MALIGNANT_LABEL), 241)

        # UCI row 1 has class 2 (benign); Bare Nuclei is explicitly omitted.
        np.testing.assert_array_equal(dataset.features[0], [5, 1, 1, 1, 2, 3, 1, 1])
        self.assertEqual(dataset.labels[0], BENIGN_LABEL)

        # The first class-4 observation is malignant after preprocessing.
        np.testing.assert_array_equal(dataset.features[5], [8, 10, 10, 8, 7, 9, 7, 1])
        self.assertEqual(dataset.labels[5], MALIGNANT_LABEL)

    def test_grouped_split_is_deterministic_balanced_and_leakage_resistant(self) -> None:
        dataset = load_dataset(DEFAULT_DATA_PATH)
        first = grouped_stratified_split(dataset.sample_ids, dataset.labels, seed=42)
        second = grouped_stratified_split(dataset.sample_ids, dataset.labels, seed=42)

        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(sum(indices.size for indices in first), dataset.labels.size)
        self.assertEqual(len(set(np.concatenate(first).tolist())), dataset.labels.size)

        id_sets = [set(dataset.sample_ids[indices]) for indices in first]
        self.assertTrue(id_sets[0].isdisjoint(id_sets[1]))
        self.assertTrue(id_sets[0].isdisjoint(id_sets[2]))
        self.assertTrue(id_sets[1].isdisjoint(id_sets[2]))

        overall_rate = float(dataset.labels.mean())
        for indices in first:
            self.assertLess(abs(float(dataset.labels[indices].mean()) - overall_rate), 0.04)

    def test_standardisation_uses_only_fitted_values(self) -> None:
        features = np.asarray([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(features)

        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(scaler.transform([[3.0, 7.0]]), [[0.0, 0.0]])

    def test_prepared_csv_is_row_oriented_and_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "prepared.csv"
            write_prepared_csv(DEFAULT_DATA_PATH, destination)

            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["sample_id", *FEATURE_NAMES, "diagnosis"])
            self.assertEqual(len(rows), 700)
            self.assertEqual(rows[1][-1], "benign")

            with self.assertRaises(FileExistsError):
                write_prepared_csv(DEFAULT_DATA_PATH, destination)


if __name__ == "__main__":
    unittest.main()
