from __future__ import annotations

import inspect
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np


class ModelHandler(ABC):
    def _palette_from_cmap(self, n_colors: int) -> list:
        """
        Build a subtle sampled color palette from the active colormap.

        Args:
            n_colors: Number of colors to generate.

        Returns:
            List of RGBA tuples sampled from the colormap.
        """
        cmap_name = getattr(self, "_plot_cmap", "viridis")
        cmap = plt.get_cmap(cmap_name)
        if n_colors <= 1:
            stops = np.asarray([0.6], dtype=float)
        else:
            stops = np.linspace(0.15, 0.85, n_colors)
        return [cmap(float(stop)) for stop in stops]

    def attach_extra_metrics(self, metrics: dict, splits: dict) -> None:
        """
        Hook for subclasses to append non-scalar metric structures.

        Args:
            metrics: Metric result dictionary to enrich.
            splits: Data splits used for metric computation.
        """
        return None

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

    def compute_metrics(self, splits: dict, exclude: list[str]) -> dict:
        """
        Compute all discovered metrics for the given data splits.

        Args:
            splits: Mapping of split names to their data.
            exclude: Metric IDs to skip.

        Returns:
            Dict keyed by metric ID with ``name`` and ``values`` entries.
        """
        results = {}
        for metric_id, display_name in self._metrics().items():
            if metric_id in exclude:
                continue
            method = getattr(self, f"metric_{metric_id}")
            results[metric_id] = {
                "name": display_name,
                "values": method(splits),
            }
        self.attach_extra_metrics(results, splits)
        return results

    def generate_plots(
        self,
        splits: dict,
        colors: tuple[str, str],
        exclude: list[str],
        cmap: str = "viridis",
    ) -> dict:
        """
        Generate matplotlib figures for all discovered plot methods.

        Args:
            splits: Mapping of split names to their data.
            colors: foreground and background colors applied to each figure.
            exclude: Plot IDs to skip.
            cmap: Colormap name used for plot color styling.

        Returns:
            Dict keyed by plot ID with ``name`` and ``fig`` entries.
        """
        results = {}
        plot_fg, plot_bg = colors
        self._plot_cmap = cmap
        palette = self._palette_from_cmap(max(len(splits), 1))
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
            ax.set_title(display_name)
            results[plot_id] = {"name": display_name, "fig": fig}

        return results

    def compute_cv_metrics(self, splits: dict, cv, exclude: list[str]) -> dict:
        """
        Compute fold-level metric summaries for cross-validation.

        Args:
            splits: Mapping containing the cross-validation data split.
            cv: Cross-validation splitter with ``split(X, y)``.
            exclude: Metric IDs to skip.

        Returns:
            Metric dict in standard report format with CV summary under
            values['cv'] as scores/mean/std/min/max.
        """
        if not splits:
            return {}

        first_split = next(iter(splits.values()))
        X, y, y_pred = first_split

        metrics = {}
        for metric_id, display_name in self._metrics().items():
            if metric_id in exclude:
                continue

            method = getattr(self, f"metric_{metric_id}")
            fold_scores = []

            for _, test_idx in cv.split(X, y):
                fold_split = {
                    "cv": (
                        np.asarray(X)[test_idx],
                        np.asarray(y)[test_idx],
                        np.asarray(y_pred)[test_idx],
                    )
                }
                fold_value = method(fold_split)["cv"]
                fold_scores.append(float(fold_value))

            summary = {
                "scores": fold_scores,
                "mean": float(np.mean(fold_scores)),
                "std": float(np.std(fold_scores, ddof=0)),
                "min": float(np.min(fold_scores)),
                "max": float(np.max(fold_scores)),
            }

            metrics[metric_id] = {
                "name": display_name,
                "values": {"cv": summary},
            }

        self.attach_extra_metrics(metrics, splits)
        return metrics
