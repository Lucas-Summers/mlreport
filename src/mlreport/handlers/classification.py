import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .base import ModelHandler


class ClassificationHandler(ModelHandler):
    def metric_accuracy(self, splits: dict) -> dict:
        """Accuracy"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(accuracy_score(y, y_pred))
        return results

    def metric_precision_macro(self, splits: dict) -> dict:
        """Precision (Macro)"""
        results = {}
        per_class = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                precision_score(y, y_pred, average="macro")
            )
            per_class[name] = self._per_class_scores(y, y_pred, precision_score)
        results["per_class"] = per_class
        return results

    def metric_recall_macro(self, splits: dict) -> dict:
        """Recall (Macro)"""
        results = {}
        per_class = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                recall_score(y, y_pred, average="macro")
            )
            per_class[name] = self._per_class_scores(y, y_pred, recall_score)
        results["per_class"] = per_class
        return results

    def metric_f1_macro(self, splits: dict) -> dict:
        """F1 Score (Macro)"""
        results = {}
        per_class = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                f1_score(y, y_pred, average="macro")
            )
            per_class[name] = self._per_class_scores(y, y_pred, f1_score)
        results["per_class"] = per_class
        return results

    def metric_precision_weighted(self, splits: dict) -> dict:
        """Precision (Weighted)"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                precision_score(y, y_pred, average="weighted")
            )
        return results

    def metric_recall_weighted(self, splits: dict) -> dict:
        """Recall (Weighted)"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                recall_score(y, y_pred, average="weighted")
            )
        return results

    def metric_f1_weighted(self, splits: dict) -> dict:
        """F1 Score (Weighted)"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                f1_score(y, y_pred, average="weighted")
            )
        return results

    def _per_class_scores(self, y, y_pred, scorer) -> dict[str, float]:
        y_arr = np.asarray(y)
        y_pred_arr = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_arr, y_pred_arr]))
        split_class_scores = {}
        for label in labels:
            y_true_bin = (y_arr == label).astype(int)
            y_pred_bin = (y_pred_arr == label).astype(int)
            split_class_scores[str(label)] = float(scorer(y_true_bin, y_pred_bin))
        return split_class_scores

    def plot_confusion_matrix(self, ax, splits: dict):
        """Confusion Matrix"""
        if "test" in splits:
            split_items = [("test", splits["test"])]
        else:
            split_items = list(splits.items())
        fig = ax.figure
        fig.clear()
        axes = fig.subplots(1, len(split_items), squeeze=False)[0]

        for idx, (split_name, (_, y, y_pred)) in enumerate(split_items):
            labels = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))
            cm = confusion_matrix(y, y_pred, labels=labels)
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
                ax=axes[idx],
                colorbar=False,
                cmap=getattr(self, "_plot_cmap", "viridis"),
            )

    def plot_class_distribution(self, ax, splits: dict):
        """Class Distribution"""
        split_names = list(splits.keys())
        class_labels = sorted(
            {
                label
                for _, (_, y, _) in splits.items()
                for label in np.unique(np.asarray(y))
            }
        )

        counts_by_split = {}
        for split_name, (_, y, _) in splits.items():
            y_arr = np.asarray(y)
            counts = {
                label: int(np.sum(y_arr == label))
                for label in class_labels
            }
            counts_by_split[split_name] = counts

        x = np.arange(len(class_labels))
        width = 0.8 / max(len(split_names), 1)

        for i, split_name in enumerate(split_names):
            offsets = x - 0.4 + width / 2 + i * width
            values = [counts_by_split[split_name][label] for label in class_labels]
            ax.bar(offsets, values, width=width, label=split_name.capitalize())

        ax.set_xticks(x)
        ax.set_xticklabels([str(label) for label in class_labels])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.legend()

    def plot_predicted_class_distribution(self, ax, splits: dict):
        """Predicted Class Distribution"""
        split_name = "test" if "test" in splits else "cv"
        _, y, y_pred = splits[split_name]
        y_arr = np.asarray(y)
        y_pred_arr = np.asarray(y_pred)
        class_labels = np.unique(np.concatenate([y_arr, y_pred_arr]))

        actual_counts = [int(np.sum(y_arr == label)) for label in class_labels]
        predicted_counts = [int(np.sum(y_pred_arr == label)) for label in class_labels]

        x = np.arange(len(class_labels))
        width = 0.38

        ax.bar(x - width / 2, actual_counts, width=width, label="Actual")
        ax.bar(x + width / 2, predicted_counts, width=width, label="Predicted")

        ax.set_xticks(x)
        ax.set_xticklabels([str(label) for label in class_labels])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(split_name.capitalize())
        ax.legend()

    def plot_per_class_metrics(self, ax, splits: dict):
        """Per-Class Metrics"""
        split_name = "test" if "test" in splits else "cv"
        _, y, y_pred = splits[split_name]
        precision_scores = self._per_class_scores(y, y_pred, precision_score)
        recall_scores = self._per_class_scores(y, y_pred, recall_score)
        f1_scores = self._per_class_scores(y, y_pred, f1_score)
        class_labels = list(precision_scores.keys())

        x = np.arange(len(class_labels))
        width = 0.24

        ax.bar(
            x - width,
            [precision_scores[label] for label in class_labels],
            width=width,
            label="Precision",
        )
        ax.bar(
            x,
            [recall_scores[label] for label in class_labels],
            width=width,
            label="Recall",
        )
        ax.bar(
            x + width,
            [f1_scores[label] for label in class_labels],
            width=width,
            label="F1",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_title(split_name.capitalize())
        ax.legend()
