from __future__ import annotations

import inspect
from abc import ABC

import matplotlib.pyplot as plt


class ModelHandler(ABC):
    @classmethod
    def _discover(cls, prefix: str) -> dict[str, str]:
        """Find methods with the given prefix and return {id: display_name} from docstrings."""
        result = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith(prefix):
                id = name[len(prefix) :]
                result[id] = method.__doc__ or id
        return result

    @classmethod
    def _metrics(cls) -> dict[str, str]:
        """Return all available metrics for the model type as {id: display_name}."""
        return cls._discover("metric_")

    @classmethod
    def _plots(cls) -> dict[str, str]:
        """Return all available plots for the model type as {id: display_name}."""
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
        return results

    def generate_plots(
        self, splits: dict, colors: tuple[str, str], exclude: list[str]
    ) -> dict:
        """
        Generate matplotlib figures for all discovered plot methods.

        Args:
            splits: Mapping of split names to their data.
            colors: foreground and background colors applied to each figure.
            exclude: Plot IDs to skip.

        Returns:
            Dict keyed by plot ID with ``name`` and ``fig`` entries.
        """
        results = {}
        plot_fg, plot_bg = colors
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

            method = getattr(self, f"plot_{plot_id}")
            method(ax, splits)
            ax.set_title(display_name)
            results[plot_id] = {"name": display_name, "fig": fig}

        return results
