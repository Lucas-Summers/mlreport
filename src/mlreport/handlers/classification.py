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
    def attach_extra_metrics(self, metrics: dict, splits: dict) -> None:
        metric_ids = ["precision_macro", "recall_macro", "f1_macro"]

        for metric_id in metric_ids:
            if metric_id not in metrics:
                continue

            per_class_values = {}
            for split_name, (_, y, y_pred) in splits.items():
                y_arr = np.asarray(y)
                y_pred_arr = np.asarray(y_pred)
                labels = np.unique(np.concatenate([y_arr, y_pred_arr]))
                split_class_scores = {}
                for label in labels:
                    y_true_bin = (y_arr == label).astype(int)
                    y_pred_bin = (y_pred_arr == label).astype(int)

                    if metric_id == "precision_macro":
                        score = precision_score(y_true_bin, y_pred_bin)
                    elif metric_id == "recall_macro":
                        score = recall_score(y_true_bin, y_pred_bin)
                    else:
                        score = f1_score(y_true_bin, y_pred_bin)

                    split_class_scores[str(label)] = float(score)

                per_class_values[split_name] = split_class_scores

            metrics[metric_id]["values"]["per_class"] = per_class_values

    def metric_accuracy(self, splits: dict) -> dict:
        """Accuracy"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(accuracy_score(y, y_pred))
        return results

    def metric_precision_macro(self, splits: dict) -> dict:
        """Precision (Macro)"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                precision_score(y, y_pred, average="macro")
            )
        return results

    def metric_recall_macro(self, splits: dict) -> dict:
        """Recall (Macro)"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                recall_score(y, y_pred, average="macro")
            )
        return results

    def metric_f1_macro(self, splits: dict) -> dict:
        """F1 Score (Macro)"""
        results = {}
        for name, (_, y, y_pred) in splits.items():
            results[name] = float(
                f1_score(y, y_pred, average="macro")
            )
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
            axes[idx].set_title(split_name.capitalize())

        fig.suptitle("Confusion Matrix")
        fig.tight_layout()

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
