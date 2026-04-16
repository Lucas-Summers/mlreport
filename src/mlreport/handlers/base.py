from __future__ import annotations

import inspect
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from ..theme import get_palette, get_plot_colors


class ModelHandler(ABC):
    @classmethod
    def _discover(cls, prefix: str) -> dict[str, str]:
        """
        Discover methods with the given prefix.

        Args:
            prefix: Function name prefix to match.

        Returns:
            Mapping of discovered IDs to display names from docstrings.
        """
        result = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith(prefix):
                id = name[len(prefix) :]
                result[id] = method.__doc__ or id
        return result

    @classmethod
    def _metrics(cls) -> dict[str, str]:
        """
        Return all available metrics for the model type.

        Returns:
            Mapping of metric IDs to display names.
        """
        return cls._discover("metric_")

    @classmethod
    def _plots(cls) -> dict[str, str]:
        """
        Return all available plots for the model type.

        Returns:
            Mapping of plot IDs to display names.
        """
        return cls._discover("plot_")

    def build_metrics(self, splits: dict, exclude: list[str], cv=None) -> dict:
        """
        Compute all discovered metrics for the given data splits.

        Args:
            splits: Mapping of split names to their data.
            exclude: Metric IDs to skip.
            cv: Optional cross-validation splitter. When provided, metrics are
                summarized fold-by-fold under the ``cv`` split.

        Returns:
            Dict keyed by metric ID with ``name``, ``values``, and any
            additional metadata returned by the metric function.
        """
        if not splits:
            return {}

        if cv is not None:
            first_split = next(iter(splits.values()))
            X, y, y_pred = first_split
        else:
            X, y, y_pred = None, None, None

        metrics = {}
        for metric_id, display_name in self._metrics().items():
            if metric_id in exclude:
                continue

            if cv is None:
                metric_result = getattr(self, f"metric_{metric_id}")(splits)
            else:
                metric_result = self._build_cv_metric_values(
                    metric_id=metric_id,
                    X=X,
                    y=y,
                    y_pred=y_pred,
                    cv=cv,
                )

            if isinstance(metric_result, dict) and "values" in metric_result:
                metric_payload = dict(metric_result)
            else:
                metric_payload = {"values": metric_result}

            metric_payload["name"] = display_name
            metrics[metric_id] = metric_payload

        return metrics

    def build_plots(
        self,
        splits: dict,
        theme: str,
        exclude: list[str],
        cmap: str = "viridis",
    ) -> dict:
        """
        Generate matplotlib figures for all discovered plot methods.

        Args:
            splits: Mapping of split names to their data.
            theme: Theme name used to derive plot foreground/background colors.
            exclude: Plot IDs to skip.
            cmap: Colormap name used for plot color styling.

        Returns:
            Dict keyed by plot ID with ``name`` and ``fig`` entries.
        """
        results = {}
        plot_fg, plot_bg = get_plot_colors(theme)
        self._plot_cmap = cmap
        palette = get_palette(cmap, max(len(splits), 3))
        for plot_id, display_name in self._plots().items():
            if plot_id in exclude:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))

            fig.patch.set_facecolor(plot_bg)
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=plot_fg)
            ax.xaxis.label.set_color(plot_fg)
            ax.yaxis.label.set_color(plot_fg)
            ax.title.set_color(plot_fg)
            for spine in ax.spines.values():
                spine.set_edgecolor(plot_fg)
            ax.set_prop_cycle(color=palette)

            method = getattr(self, f"plot_{plot_id}")
            method(ax, splits)
            for plot_ax in fig.axes:
                legend = plot_ax.get_legend()
                if legend is not None:
                    handles, labels = plot_ax.get_legend_handles_labels()
                    plot_ax.legend(handles, labels, loc="upper right")
            target_ax = fig.axes[0] if fig.axes else ax
            target_ax.set_title(display_name)
            fig.tight_layout()
            results[plot_id] = {"name": display_name, "fig": fig}

        return results

    def _build_cv_metric_values(self, metric_id: str, X, y, y_pred, cv) -> dict:
        """
        Build the standard ``values`` payload for a CV metric summary.

        Args:
            metric_id: Metric identifier to compute.
            X: Full feature matrix.
            y: Full target array.
            y_pred: Out-of-fold predictions aligned with ``y``.
            cv: Cross-validation splitter.

        Returns:
            Dict containing the normalized metric payload with a ``values`` key.
        """
        method = getattr(self, f"metric_{metric_id}")
        fold_scores = []
        metric_meta = {}

        for _, test_idx in cv.split(X, y):
            fold_split = {
                "cv": (
                    np.asarray(X)[test_idx],
                    np.asarray(y)[test_idx],
                    np.asarray(y_pred)[test_idx],
                )
            }
            fold_metric = method(fold_split)
            fold_values = (
                fold_metric["values"]
                if isinstance(fold_metric, dict) and "values" in fold_metric
                else fold_metric
            )
            if not metric_meta and isinstance(fold_metric, dict) and "values" in fold_metric:
                metric_meta.update(
                    {
                        key: value
                        for key, value in fold_metric.items()
                        if key != "values"
                    }
                )
            fold_value = fold_values["cv"]
            fold_scores.append(float(fold_value))

        summary = {
            "scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores, ddof=0)),
            "min": float(np.min(fold_scores)),
            "max": float(np.max(fold_scores)),
        }
        metric_payload = dict(metric_meta)
        metric_payload["values"] = {"cv": summary}
        return metric_payload
