"""Educational two-class logistic classifier implemented with NumPy.

This project uses the Breast Cancer Wisconsin (Original) dataset. It is an
educational demonstration and must not be used for diagnosis or treatment.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "raw" / "breast-cancer-wisconsin.data"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs"

ALL_FEATURE_NAMES = (
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
)
OMITTED_FEATURE = "bare_nuclei"
FEATURE_NAMES = tuple(name for name in ALL_FEATURE_NAMES if name != OMITTED_FEATURE)
OMITTED_FEATURE_INDEX = ALL_FEATURE_NAMES.index(OMITTED_FEATURE)

BENIGN_LABEL = 0
MALIGNANT_LABEL = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible training and evaluation settings."""

    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    learning_rate: float = 0.10
    max_iterations: int = 5_000
    l2_strength: float = 0.0
    patience: int = 250
    min_delta: float = 1e-6
    threshold: float = 0.50

    def validate(self) -> None:
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if not np.isclose(fractions, 1.0):
            raise ValueError("Train, validation, and test fractions must sum to 1.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All split fractions must be positive.")
        if self.learning_rate <= 0 or self.max_iterations <= 0:
            raise ValueError("Learning rate and maximum iterations must be positive.")
        if self.l2_strength < 0 or self.patience <= 0:
            raise ValueError("L2 strength cannot be negative and patience must be positive.")
        if not 0 < self.threshold < 1:
            raise ValueError("The classification threshold must be between 0 and 1.")


@dataclass(frozen=True)
class Dataset:
    """Parsed features, labels, and grouping identifiers."""

    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


@dataclass
class StandardScaler:
    """Feature standardisation fitted on training observations only."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "StandardScaler":
        features = _as_feature_matrix(features)
        self.mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler must be fitted before transform is called.")
        features = _as_feature_matrix(features)
        return (features - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


class BinaryLogisticRegression:
    """Binary logistic regression trained using an analytical gradient."""

    def __init__(
        self,
        learning_rate: float = 0.10,
        max_iterations: int = 5_000,
        l2_strength: float = 0.0,
        patience: int = 250,
        min_delta: float = 1e-6,
    ) -> None:
        if learning_rate <= 0 or max_iterations <= 0:
            raise ValueError("Learning rate and maximum iterations must be positive.")
        if l2_strength < 0 or patience <= 0:
            raise ValueError("L2 strength cannot be negative and patience must be positive.")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l2_strength = l2_strength
        self.patience = patience
        self.min_delta = min_delta
        self.weights_: np.ndarray | None = None
        self.best_iteration_: int | None = None
        self.history_: dict[str, list[float]] = {
            "iteration": [],
            "training_loss": [],
            "validation_loss": [],
            "training_error_rate": [],
            "validation_error_rate": [],
        }

    def fit(
        self,
        training_features: np.ndarray,
        training_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "BinaryLogisticRegression":
        training_features = _as_feature_matrix(training_features)
        validation_features = _as_feature_matrix(validation_features)
        training_labels = _as_binary_labels(training_labels)
        validation_labels = _as_binary_labels(validation_labels)

        if training_features.shape[0] != training_labels.size:
            raise ValueError("Training features and labels have different lengths.")
        if validation_features.shape[0] != validation_labels.size:
            raise ValueError("Validation features and labels have different lengths.")
        if training_features.shape[1] != validation_features.shape[1]:
            raise ValueError("Training and validation feature counts differ.")

        training_design = _add_bias_column(training_features)
        validation_design = _add_bias_column(validation_features)
        weights = np.zeros(training_design.shape[1], dtype=float)
        best_weights = weights.copy()
        best_validation_loss = np.inf
        iterations_without_improvement = 0

        for iteration in range(1, self.max_iterations + 1):
            probabilities = stable_sigmoid(training_design @ weights)
            gradient = training_design.T @ (probabilities - training_labels)
            gradient /= training_labels.size
            gradient[1:] += self.l2_strength * weights[1:]
            weights -= self.learning_rate * gradient

            training_loss = logistic_loss(
                training_design, training_labels, weights, self.l2_strength
            )
            validation_loss = logistic_loss(
                validation_design, validation_labels, weights, self.l2_strength
            )
            training_predictions = (stable_sigmoid(training_design @ weights) >= 0.5).astype(int)
            validation_predictions = (
                stable_sigmoid(validation_design @ weights) >= 0.5
            ).astype(int)

            self.history_["iteration"].append(float(iteration))
            self.history_["training_loss"].append(float(training_loss))
            self.history_["validation_loss"].append(float(validation_loss))
            self.history_["training_error_rate"].append(
                float(np.mean(training_predictions != training_labels))
            )
            self.history_["validation_error_rate"].append(
                float(np.mean(validation_predictions != validation_labels))
            )

            if validation_loss < best_validation_loss - self.min_delta:
                best_validation_loss = validation_loss
                best_weights = weights.copy()
                self.best_iteration_ = iteration
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement >= self.patience:
                    break

        self.weights_ = best_weights
        if self.best_iteration_ is None:
            raise RuntimeError("Training did not produce a valid model.")
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("The model must be fitted before prediction.")
        design = _add_bias_column(_as_feature_matrix(features))
        return stable_sigmoid(design @ self.weights_)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("The classification threshold must be between 0 and 1.")
        return (self.predict_proba(features) >= threshold).astype(int)


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> Dataset:
    """Load the official raw UCI data and reproduce the approved 8-feature scope.

    The sample code number is retained only for grouped splitting. Bare Nuclei
    is omitted explicitly because it is the sole feature containing missing
    values in the raw dataset and was absent from the uploaded coursework CSV.
    """

    path = Path(path)
    rows: list[list[float]] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 11:
                raise ValueError(f"Expected 11 columns at line {line_number}; found {len(row)}.")

            sample_id, *feature_text, class_text = row
            original_class = int(class_text)
            if original_class not in (2, 4):
                raise ValueError(f"Unexpected class {original_class} at line {line_number}.")

            retained: list[float] = []
            for index, value in enumerate(feature_text):
                if index == OMITTED_FEATURE_INDEX:
                    continue
                if value == "?":
                    raise ValueError(
                        f"Missing value outside the omitted feature at line {line_number}."
                    )
                retained.append(float(value))

            rows.append(retained)
            labels.append(MALIGNANT_LABEL if original_class == 4 else BENIGN_LABEL)
            sample_ids.append(sample_id)

    dataset = Dataset(
        features=np.asarray(rows, dtype=float),
        labels=np.asarray(labels, dtype=int),
        sample_ids=np.asarray(sample_ids, dtype=str),
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    """Validate structural and clinical-label invariants before modelling."""

    features = _as_feature_matrix(dataset.features)
    labels = _as_binary_labels(dataset.labels)
    sample_ids = np.asarray(dataset.sample_ids)
    if features.shape[0] != labels.size or labels.size != sample_ids.size:
        raise ValueError("Features, labels, and sample IDs must have equal row counts.")
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features; found {features.shape[1]}.")
    if not np.isfinite(features).all():
        raise ValueError("The retained feature matrix contains non-finite values.")
    if np.any((features < 1) | (features > 10)):
        raise ValueError("All retained UCI feature values must be in the range 1 to 10.")
    if labels.size == 0 or set(np.unique(labels)) != {BENIGN_LABEL, MALIGNANT_LABEL}:
        raise ValueError("Both benign and malignant observations are required.")
    if np.any(sample_ids == ""):
        raise ValueError("Sample IDs cannot be empty.")


def grouped_stratified_split(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split observations without placing one sample ID in multiple subsets.

    A deterministic greedy assignment balances class counts and total sizes
    while treating every repeated sample code as one indivisible group.
    """

    sample_ids = np.asarray(sample_ids, dtype=str)
    labels = _as_binary_labels(np.asarray(labels))
    fractions_array = np.asarray(fractions, dtype=float)
    if fractions_array.shape != (3,) or np.any(fractions_array <= 0):
        raise ValueError("Exactly three positive split fractions are required.")
    if not np.isclose(fractions_array.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1.")
    if sample_ids.size != labels.size:
        raise ValueError("Sample IDs and labels must have equal lengths.")

    groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(str(sample_id), []).append(index)

    rng = np.random.default_rng(seed)
    group_records = []
    for sample_id, indices in groups.items():
        class_counts = np.bincount(labels[indices], minlength=2).astype(float)
        group_records.append((sample_id, np.asarray(indices, dtype=int), class_counts, rng.random()))
    group_records.sort(key=lambda item: (-len(item[1]), item[3]))

    total_class_counts = np.bincount(labels, minlength=2).astype(float)
    target_class_counts = fractions_array[:, None] * total_class_counts[None, :]
    target_sizes = fractions_array * labels.size
    assigned_class_counts = np.zeros((3, 2), dtype=float)
    assigned_sizes = np.zeros(3, dtype=float)
    split_indices: list[list[int]] = [[], [], []]

    for _sample_id, indices, class_counts, _tie_breaker in group_records:
        candidate_costs = []
        for split_number in range(3):
            next_class_counts = assigned_class_counts.copy()
            next_sizes = assigned_sizes.copy()
            next_class_counts[split_number] += class_counts
            next_sizes[split_number] += indices.size

            class_error = np.sum(
                ((next_class_counts - target_class_counts) / np.maximum(target_class_counts, 1.0))
                ** 2
            )
            size_error = np.sum(
                ((next_sizes - target_sizes) / np.maximum(target_sizes, 1.0)) ** 2
            )
            overflow = np.sum(
                np.maximum(next_class_counts - target_class_counts, 0.0)
                / np.maximum(target_class_counts, 1.0)
            )
            candidate_costs.append(class_error + 0.25 * size_error + 0.5 * overflow)

        chosen_split = int(np.argmin(candidate_costs))
        split_indices[chosen_split].extend(indices.tolist())
        assigned_class_counts[chosen_split] += class_counts
        assigned_sizes[chosen_split] += indices.size

    outputs = tuple(np.sort(np.asarray(indices, dtype=int)) for indices in split_indices)
    if any(indices.size == 0 for indices in outputs):
        raise RuntimeError("Grouped splitting produced an empty subset.")
    return outputs  # type: ignore[return-value]


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute the sigmoid without overflow for large positive or negative values."""

    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def logistic_loss(
    design_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_strength: float = 0.0,
) -> float:
    """Return stable mean binary cross-entropy plus L2 weight regularisation."""

    logits = design_matrix @ weights
    data_loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    penalty = 0.5 * l2_strength * float(np.dot(weights[1:], weights[1:]))
    return float(data_loss + penalty)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    """Calculate binary metrics with malignant disease as the positive class."""

    labels = _as_binary_labels(labels)
    predictions = _as_binary_labels(predictions)
    if labels.size != predictions.size:
        raise ValueError("Labels and predictions must have equal lengths.")

    true_positive = int(np.sum((labels == MALIGNANT_LABEL) & (predictions == MALIGNANT_LABEL)))
    false_negative = int(np.sum((labels == MALIGNANT_LABEL) & (predictions == BENIGN_LABEL)))
    false_positive = int(np.sum((labels == BENIGN_LABEL) & (predictions == MALIGNANT_LABEL)))
    true_negative = int(np.sum((labels == BENIGN_LABEL) & (predictions == BENIGN_LABEL)))

    def safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    accuracy = safe_divide(true_positive + true_negative, labels.size)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        "observations": int(labels.size),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "accuracy": accuracy,
        "precision_malignant": precision,
        "recall_malignant": recall,
        "specificity_benign": specificity,
        "f1_malignant": f1_score,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    create_plots: bool = True,
) -> dict[str, object]:
    """Run the complete reproducible train/validation/test workflow."""

    config.validate()
    dataset = load_dataset(data_path)
    train_indices, validation_indices, test_indices = grouped_stratified_split(
        dataset.sample_ids,
        dataset.labels,
        fractions=(config.train_fraction, config.validation_fraction, config.test_fraction),
        seed=config.seed,
    )

    scaler = StandardScaler()
    training_features = scaler.fit_transform(dataset.features[train_indices])
    validation_features = scaler.transform(dataset.features[validation_indices])
    test_features = scaler.transform(dataset.features[test_indices])

    model = BinaryLogisticRegression(
        learning_rate=config.learning_rate,
        max_iterations=config.max_iterations,
        l2_strength=config.l2_strength,
        patience=config.patience,
        min_delta=config.min_delta,
    ).fit(
        training_features,
        dataset.labels[train_indices],
        validation_features,
        dataset.labels[validation_indices],
    )

    split_metrics: dict[str, dict[str, float | int]] = {}
    for name, indices, features in (
        ("training", train_indices, training_features),
        ("validation", validation_indices, validation_features),
        ("test", test_indices, test_features),
    ):
        predictions = model.predict(features, threshold=config.threshold)
        split_metrics[name] = classification_metrics(dataset.labels[indices], predictions)

    synthetic_examples = {
        "A": [3, 5, 2, 7, 6, 1, 5, 1],
        "B": [1, 5, 4, 7, 1, 6, 5, 2],
        "C": [1, 9, 1, 1, 1, 9, 1, 1],
    }
    synthetic_matrix = scaler.transform(np.asarray(list(synthetic_examples.values()), dtype=float))
    synthetic_probabilities = model.predict_proba(synthetic_matrix)
    synthetic_predictions = model.predict(synthetic_matrix, threshold=config.threshold)
    synthetic_results = {
        name: {
            "feature_values": values,
            "predicted_class": "malignant" if prediction == 1 else "benign",
            "model_probability_malignant": float(probability),
            "note": "Synthetic educational example; not a clinical prediction.",
        }
        for name, values, probability, prediction in zip(
            synthetic_examples,
            synthetic_examples.values(),
            synthetic_probabilities,
            synthetic_predictions,
        )
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "disclaimer": "Educational demonstration only; not for diagnosis or treatment.",
        "dataset": {
            "source": "UCI Breast Cancer Wisconsin (Original), dataset id 15",
            "observations": int(dataset.labels.size),
            "features": list(dataset.feature_names),
            "omitted_feature": OMITTED_FEATURE,
            "class_counts": {
                "benign": int(np.sum(dataset.labels == BENIGN_LABEL)),
                "malignant": int(np.sum(dataset.labels == MALIGNANT_LABEL)),
            },
        },
        "config": asdict(config),
        "split_sizes": {
            "training": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
        },
        "best_iteration": model.best_iteration_,
        "metrics": split_metrics,
        "synthetic_examples": synthetic_results,
    }

    _write_json(output_dir / "metrics.json", results)
    _write_history(output_dir / "training_history.csv", model.history_)
    np.savez(
        output_dir / "model.npz",
        weights=model.weights_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_names=np.asarray(FEATURE_NAMES),
        threshold=config.threshold,
    )

    if create_plots:
        _plot_training_history(model.history_, model.best_iteration_, output_dir)
        test_predictions = model.predict(test_features, threshold=config.threshold)
        _plot_confusion_matrix(dataset.labels[test_indices], test_predictions, output_dir)

    return results


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _write_history(path: Path, history: dict[str, list[float]]) -> None:
    columns = tuple(history)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(history[column] for column in columns)))


def _plot_training_history(
    history: dict[str, list[float]], best_iteration: int | None, output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    iterations = np.asarray(history["iteration"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(iterations, history["training_loss"], label="Training")
    axes[0].plot(iterations, history["validation_loss"], label="Validation")
    axes[0].set(title="Stable logistic loss", xlabel="Iteration", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(iterations, history["training_error_rate"], label="Training")
    axes[1].plot(iterations, history["validation_error_rate"], label="Validation")
    axes[1].set(title="Classification error", xlabel="Iteration", ylabel="Error rate")
    axes[1].legend()

    if best_iteration is not None:
        for axis in axes:
            axis.axvline(best_iteration, color="#B91C1C", linestyle="--", linewidth=1, label=None)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Training diagnostics (model selected by validation loss)")
    figure.tight_layout()
    figure.savefig(output_dir / "training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = classification_metrics(labels, predictions)
    matrix = np.asarray(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Benign", "Malignant"])
    axis.set_yticks([0, 1], labels=["Benign", "Malignant"])
    axis.set(xlabel="Predicted class", ylabel="Actual class", title="Held-out test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("Features must be a two-dimensional matrix.")
    return features


def _as_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if not set(np.unique(labels)).issubset({BENIGN_LABEL, MALIGNANT_LABEL}):
        raise ValueError("Labels must be encoded as 0 (benign) and 1 (malignant).")
    return labels


def _add_bias_column(features: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(features.shape[0]), features))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an educational NumPy logistic classifier on the UCI dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--max-iterations", type=int, default=5_000)
    parser.add_argument("--l2-strength", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(arguments: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(arguments)
    config = TrainingConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        l2_strength=args.l2_strength,
        patience=args.patience,
        threshold=args.threshold,
    )
    results = run_experiment(args.data, args.output_dir, config, create_plots=not args.no_plots)
    test_metrics = results["metrics"]["test"]  # type: ignore[index]

    print("Educational demonstration only; not for diagnosis or treatment.")
    print(f"Best validation-loss iteration: {results['best_iteration']}")
    print(f"Held-out test observations: {test_metrics['observations']}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Malignant precision: {test_metrics['precision_malignant']:.4f}")
    print(f"Malignant recall: {test_metrics['recall_malignant']:.4f}")
    print(f"Benign specificity: {test_metrics['specificity_benign']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Results written to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
