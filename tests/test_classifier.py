from __future__ import annotations

import unittest

import numpy as np

from breast_cancer_classifier import (
    BinaryLogisticRegression,
    StandardScaler,
    classification_metrics,
    logistic_loss,
    stable_sigmoid,
)


class ClassifierTests(unittest.TestCase):
    def test_stable_sigmoid_handles_extreme_values(self) -> None:
        probabilities = stable_sigmoid(np.asarray([-1_000.0, 0.0, 1_000.0]))
        np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0], atol=1e-15)

    def test_logistic_loss_is_finite_for_extreme_logits(self) -> None:
        design = np.asarray([[1.0, -1_000.0], [1.0, 1_000.0]])
        labels = np.asarray([0, 1])
        weights = np.asarray([0.0, 1.0])
        self.assertTrue(np.isfinite(logistic_loss(design, labels, weights)))

    def test_model_learns_a_separable_toy_problem(self) -> None:
        features = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        model = BinaryLogisticRegression(
            learning_rate=0.2,
            max_iterations=1_000,
            l2_strength=0.0,
            patience=100,
        ).fit(scaled, labels, scaled, labels)

        np.testing.assert_array_equal(model.predict(scaled), labels)
        self.assertLess(
            model.history_["training_loss"][-1], model.history_["training_loss"][0]
        )

    def test_metrics_use_malignant_as_positive_class(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        predictions = np.asarray([1, 0, 1, 0])
        metrics = classification_metrics(labels, predictions)

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["recall_malignant"], 0.5)
        self.assertEqual(metrics["specificity_benign"], 0.5)


if __name__ == "__main__":
    unittest.main()
