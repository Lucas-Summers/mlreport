from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import is_classifier, is_clusterer, is_regressor
from sklearn.model_selection import cross_val_predict

from .handlers.base import ModelHandler
from .handlers.classification import ClassificationHandler
from .handlers.clustering import ClusteringHandler
from .handlers.regression import RegressionHandler
from .render import (
    fig_to_base64,
    fig_to_file,
    render_html,
    render_json,
    render_md,
    render_pdf,
)
from .theme import get_plot_colors


@dataclass
class ReportState:
    handler: ModelHandler
    splits: dict[str, tuple] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    tuning: dict = field(
        default_factory=lambda: {
            "summary": None,
            "plots": {},
        }
    )
    cv: object | None = None
    built: bool = False


class Report:
    def __init__(
        self,
        model,
        title: str | None = None,
        author: str | None = None,
        description: str | None = None,
        theme: str = "light",
        cmap: str = "viridis",
    ):
        self.model = model
        self.title = title
        self.author = author
        self.description = description
        self.theme = theme
        self.cmap = cmap

        self._state = ReportState(handler=self._get_handler(model))

    def available_metrics(self) -> None:
        """
        Print available metrics for this report's model type.
        """
        for i, (id, name) in enumerate(self._state.handler._metrics().items(), 1):
            print(f"  {i}. {id} — {name}")

    def available_plots(self) -> None:
        """
        Print available plots for this report's model type.
        """
        for i, (id, name) in enumerate(self._state.handler._plots().items(), 1):
            print(f"  {i}. {id} — {name}")

    def add_split(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray | None = None,
    ) -> Report:
        """
        Add a data split (e.g., 'train', 'val', 'test').

        Args:
            name: Split name identifier.
            X: Feature matrix for the split.
            y: Ground-truth target values for the split.
            y_pred: Optional model predictions for the split.

        Returns:
            Self for method chaining.
        """
        if self._state.cv is not None:
            raise ValueError("Cannot add train/test splits after add_crossval().")

        if y_pred is None:
            y_pred = self.model.predict(X)

        self._state.splits[name] = (
            np.asarray(X),
            np.asarray(y),
            np.asarray(y_pred),
        )
        return self

    def add_crossval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray | None = None,
        cv=None,
    ) -> Report:
        """
        Add a cross-validation prediction set.

        Args:
            X: Full feature matrix used for cross-validation.
            y: Full true target values.
            y_pred: Optional out-of-fold predictions aligned with y.
            cv: Cross-validation splitter with split(X, y).

        Returns:
            Self for method chaining.
        """
        if self._state.splits:
            raise ValueError("Cannot call add_crossval() after add_split().")

        if cv is None:
            raise ValueError("cv must be provided for add_crossval().")

        if y_pred is None:
            y_pred = cast(
                np.ndarray,
                np.asarray(cross_val_predict(self.model, X, y, cv=cv)),
            )

        self._state.cv = cv
        self._state.splits = {
            "cv": (
                np.asarray(X),
                np.asarray(y),
                np.asarray(y_pred),
            )
        }
        return self

    def add_search(self, search_cv) -> Report:
        """
        Add a fitted sklearn search CV object for hyperparameter tuning output.

        Args:
            search_cv: Fitted search object (e.g., GridSearchCV, RandomizedSearchCV).

        Returns:
            Self for method chaining.
        """
        if not hasattr(search_cv, "cv_results_"):
            raise ValueError("search_cv must be fitted and expose cv_results_.")

        cv_results = getattr(search_cv, "cv_results_", {})
        if not cv_results:
            raise ValueError("search_cv.cv_results_ is empty.")

        score_column = self._get_search_score_column(cv_results, search_cv)
        param_names = self._get_search_param_names(cv_results)
        if len(param_names) != 2:
            raise ValueError(
                "mlreport currently supports exactly 2 tuned hyperparameters. "
                f"Found {len(param_names)}."
            )

        best_params = dict(getattr(search_cv, "best_params_", {}) or {})
        best_score = getattr(search_cv, "best_score_", None)
        if best_score is not None:
            best_score = float(best_score)

        params_list = list(cv_results.get("params", []))
        n_candidates = len(params_list)
        if n_candidates == 0 and score_column in cv_results:
            n_candidates = int(len(cv_results[score_column]))

        if best_score is None and score_column in cv_results:
            score_values = np.asarray(cv_results[score_column], dtype=float)
            finite_scores = score_values[np.isfinite(score_values)]
            if finite_scores.size > 0:
                best_score = float(np.max(finite_scores))

        metric_name = self._get_search_metric_name(search_cv, score_column)

        self._state.tuning["summary"] = {
            "method": search_cv.__class__.__name__,
            "metric": metric_name,
            "cv_folds": getattr(search_cv, "n_splits_", None),
            "best_score": best_score,
            "best_params": best_params,
            "n_candidates": n_candidates,
        }
        self._state.tuning["plots"] = self._build_search_param_plots(
            cv_results=cv_results,
            score_column=score_column,
            metric_name=metric_name,
            colors=self._get_plot_colors(),
            param_names=param_names,
            cmap=self.cmap,
        )
        return self

    def build(
        self,
        exclude_metrics: list[str] | None = None,
        exclude_plots: list[str] | None = None,
    ) -> Report:
        """
        Compute metrics and generate plots.

        Args:
            exclude_metrics: List of metric IDs to exclude from the report.
            exclude_plots: List of plot IDs to exclude from the report.

        Returns:
            Self for method chaining.
        """
        if not self._state.splits:
            raise ValueError(
                "No data splits added. Use add_split() or add_crossval() first."
            )

        self._state.metrics = self._state.handler.build_metrics(
            self._state.splits,
            exclude_metrics or [],
            cv=self._state.cv,
        )

        self._state.plots = self._state.handler.build_plots(
            self._state.splits,
            self.theme,
            exclude_plots or [],
            cmap=self.cmap,
        )
        self._state.built = True
        return self

    def summary(self) -> Report:
        """
        Print a text summary of the report.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        report_data = self._to_dict()
        meta = report_data.get("meta", {})
        model_data = report_data.get("model", {})
        data = report_data.get("data", {})
        metrics = report_data.get("metrics", {})
        tuning_summary = report_data.get("tuning", {}).get("summary")

        print("Report")
        if meta.get("title"):
            print(f"  title: {meta['title']}")
        if meta.get("author"):
            print(f"  author: {meta['author']}")
        if meta.get("description"):
            print(f"  description: {meta['description']}")
        if meta.get("generated_at"):
            print(f"  generated_at: {meta['generated_at']}")
        print()

        model_params = model_data.get("params", {})
        param_count = len(model_params) if isinstance(model_params, dict) else 0

        print("Model")
        if model_data.get("name"):
            print(f"  name: {model_data['name']}")
        if model_data.get("type"):
            print(f"  type: {model_data['type']}")
        if model_data.get("version"):
            print(f"  sklearn_version: {model_data['version']}")
        print(f"  param_count: {param_count}")

        if tuning_summary:
            print(f"  tuning_method: {tuning_summary.get('method')}")
            print(f"  tuning_metric: {tuning_summary.get('metric')}")

            best_score = tuning_summary.get("best_score")
            if isinstance(best_score, Number):
                print(f"  tuning_best_score: {best_score:.4f}")
            elif best_score is not None:
                print(f"  tuning_best_score: {best_score}")

            n_candidates = tuning_summary.get("n_candidates")
            if n_candidates is not None:
                print(f"  tuning_candidates: {n_candidates}")

            cv_folds = tuning_summary.get("cv_folds")
            if cv_folds is not None:
                print(f"  tuning_cv_folds: {cv_folds}")

            best_params = tuning_summary.get("best_params") or {}
            if isinstance(best_params, dict) and best_params:
                best_params_str = ", ".join(
                    f"{key}={value}" for key, value in sorted(best_params.items())
                )
                print(f"  tuning_best_params: {best_params_str}")
        print()

        splits = data.get("splits", {})
        splits_str = (
            ", ".join(f"{name}={count}" for name, count in splits.items())
            if isinstance(splits, dict)
            else ""
        )

        print("Data")
        if data.get("features") is not None:
            print(f"  features: {data['features']}")
        if splits_str:
            print(f"  splits: {splits_str}")
        if data.get("total") is not None:
            print(f"  total: {data['total']}")
        cv_folds = data.get("cv_folds")
        if cv_folds is not None:
            print(f"  cv_folds: {cv_folds}")
        print()

        print("Metrics")
        for metric_data in metrics.values():
            metric_name = metric_data.get("name", "metric")
            values = metric_data.get("values", {})

            parts = []
            if isinstance(values, dict):
                for split_name, value in values.items():
                    if split_name == "per_class":
                        continue
                    if isinstance(value, dict) and "mean" in value:
                        mean = value.get("mean")
                        std = value.get("std")
                        if isinstance(mean, Number):
                            if isinstance(std, Number):
                                parts.append(f"{split_name}={mean:.4f} (std={std:.4f})")
                            else:
                                parts.append(f"{split_name}={mean:.4f}")
                        else:
                            parts.append(f"{split_name}={mean}")
                    elif isinstance(value, Number):
                        parts.append(f"{split_name}={value:.4f}")
                    else:
                        parts.append(f"{split_name}={value}")

            if parts:
                print(f"  {metric_name}: {', '.join(parts)}")
            else:
                print(f"  {metric_name}")

        return self

    def to_html(self, path: str) -> Report:
        """
        Render report to HTML.

        Args:
            path: Output file path for the HTML report.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        plots_with_images = {
            plot_id: {
                "name": plot_data["name"],
                "image": fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.plots.items()
        }

        context = self._to_dict()
        context["plots"] = plots_with_images
        context.setdefault("tuning", {})
        context["tuning"]["plots"] = {
            plot_id: {
                "name": plot_data["name"],
                "image": fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.tuning["plots"].items()
        }

        render_html("report", self.theme, context, path=path)

        return self

    def to_pdf(self, path: str) -> Report:
        """
        Render report to PDF.

        Args:
            path: Output file path for the PDF report.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        plots_with_images = {
            plot_id: {
                "name": plot_data["name"],
                "image": fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.plots.items()
        }

        context = self._to_dict()
        context["plots"] = plots_with_images
        context.setdefault("tuning", {})
        context["tuning"]["plots"] = {
            plot_id: {
                "name": plot_data["name"],
                "image": fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.tuning["plots"].items()
        }

        render_pdf("report", self.theme, context, path=path)
        return self

    def to_json(self, path: str) -> Report:
        """
        Render report to JSON.

        Args:
            path: Output file path for the JSON report.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        data = self._to_dict()
        data.pop("plots", None)
        if isinstance(data.get("tuning"), dict):
            data["tuning"].pop("plots", None)

        render_json("report", self.theme, data, path=path)

        return self

    def to_md(self, path: str, image_dir: str | None = None) -> Report:
        """
        Render report to Markdown.

        Args:
            path: Output file path for the Markdown report.
            image_dir: Directory to store exported plot images.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        if image_dir is None:
            image_dir = str(Path(path).parent / "images")

        Path(image_dir).mkdir(exist_ok=True)

        plots_with_paths = {}
        for plot_id, plot_data in self._state.plots.items():
            img_path = f"{image_dir}/{plot_id}.png"
            fig_to_file(plot_data["fig"], img_path)
            plots_with_paths[plot_id] = {
                "name": plot_data["name"],
                "path": img_path,
            }

        context = self._to_dict()
        context["plots"] = plots_with_paths
        context.setdefault("tuning", {})
        tuning_plot_paths = {}
        for plot_id, plot_data in self._state.tuning["plots"].items():
            img_path = f"{image_dir}/tuning_{plot_id}.png"
            fig_to_file(plot_data["fig"], img_path)
            tuning_plot_paths[plot_id] = {
                "name": plot_data["name"],
                "path": img_path,
            }
        context["tuning"]["plots"] = tuning_plot_paths

        render_md("report", self.theme, context, path=path)

        return self

    def _require_built(self) -> None:
        """
        Validate that ``build()`` has been called.
        """
        if not self._state.built:
            raise ValueError("Call build() first.")

    def _get_plot_colors(self) -> tuple[str, str]:
        """Return theme-derived plot foreground/background colors.

        Returns:
            Foreground and background hex colors.
        """
        return get_plot_colors(self.theme)

    def _get_search_score_column(self, cv_results: dict, search_cv) -> str:
        refit_metric = getattr(search_cv, "refit", None)
        if isinstance(refit_metric, str):
            refit_column = f"mean_test_{refit_metric}"
            if refit_column in cv_results:
                return refit_column

        if "mean_test_score" in cv_results:
            return "mean_test_score"

        metric_columns = [
            key for key in cv_results.keys() if key.startswith("mean_test_")
        ]
        if metric_columns:
            return sorted(metric_columns)[0]

        raise ValueError("search_cv.cv_results_ does not contain mean_test_* scores.")

    def _get_search_metric_name(self, search_cv, score_column: str) -> str:
        if score_column != "mean_test_score":
            return score_column.replace("mean_test_", "")

        scoring = getattr(search_cv, "scoring", None)
        if isinstance(scoring, str):
            return scoring
        if callable(scoring):
            return getattr(scoring, "__name__", "custom_score")

        return "score"

    def _get_search_param_names(self, cv_results: dict) -> list[str]:
        params_list = list(cv_results.get("params", []))
        return sorted({name for params in params_list for name in params.keys()})

    def _build_search_param_plots(
        self,
        cv_results: dict,
        score_column: str,
        metric_name: str,
        colors: tuple[str, str],
        param_names: list[str],
        cmap: str,
    ) -> dict:
        params_list = list(cv_results.get("params", []))
        if not params_list or score_column not in cv_results:
            return {}

        if len(param_names) != 2:
            raise ValueError(
                "mlreport currently supports exactly 2 tuned hyperparameters. "
                f"Found {len(param_names)}."
            )

        score_values = np.asarray(cv_results[score_column], dtype=float)
        plot_fg, plot_bg = colors
        param_a, param_b = param_names
        values_a = [params.get(param_a) for params in params_list]
        values_b = [params.get(param_b) for params in params_list]
        is_a_numeric = values_a and all(
            self._is_numeric_param_value(value) for value in values_a
        )
        is_b_numeric = values_b and all(
            self._is_numeric_param_value(value) for value in values_b
        )

        grouped = {}
        for params, score in zip(params_list, score_values):
            if not np.isfinite(score):
                continue
            value_a = params.get(param_a)
            value_b = params.get(param_b)
            key = (repr(value_a), repr(value_b))
            if key not in grouped:
                grouped[key] = {
                    "values": {param_a: value_a, param_b: value_b},
                    "scores": [],
                }
            grouped[key]["scores"].append(float(score))

        combos = [
            {
                "values": item["values"],
                "mean_score": float(np.mean(item["scores"])),
            }
            for item in grouped.values()
            if item["scores"]
        ]

        if not combos:
            return {}

        interaction_fig = None
        interaction_name = ""
        if is_a_numeric and is_b_numeric:
            interaction_fig = self._build_numeric_numeric_tuning_plot(
                combos,
                param_a,
                param_b,
                metric_name,
                plot_fg,
                plot_bg,
                cmap,
            )
            interaction_name = f"Tuning Interaction: {param_a} vs {param_b}"
        elif is_a_numeric or is_b_numeric:
            numeric_param = param_a if is_a_numeric else param_b
            categorical_param = param_b if is_a_numeric else param_a
            interaction_fig = self._build_numeric_categorical_tuning_plot(
                combos,
                numeric_param,
                categorical_param,
                metric_name,
                plot_fg,
                plot_bg,
                cmap,
            )
            interaction_name = f"Score vs {numeric_param} by {categorical_param}"
        else:
            interaction_fig = self._build_categorical_categorical_tuning_plot(
                combos,
                param_a,
                param_b,
                metric_name,
                plot_fg,
                plot_bg,
                cmap,
            )
            interaction_name = f"Heatmap: {param_a} x {param_b}"

        tuning_plots = {}
        if interaction_fig is not None:
            interaction_id = f"interaction_{param_a}_{param_b}".replace(" ", "_")
            tuning_plots[interaction_id] = {
                "name": interaction_name,
                "fig": interaction_fig,
            }

        best_candidates_fig = self._build_best_candidates_plot(
            combos,
            param_a,
            param_b,
            metric_name,
            plot_fg,
            plot_bg,
            cmap,
            top_n=10,
        )
        if best_candidates_fig is not None:
            tuning_plots["best_candidates"] = {
                "name": "Top 10 Tuning Candidates",
                "fig": best_candidates_fig,
            }

        return tuning_plots

    def _build_numeric_numeric_tuning_plot(
        self,
        combos: list[dict],
        param_a: str,
        param_b: str,
        metric_name: str,
        plot_fg: str,
        plot_bg: str,
        cmap: str,
    ):
        x_vals = []
        y_vals = []
        z_vals = []
        for item in combos:
            value_a = item["values"].get(param_a)
            value_b = item["values"].get(param_b)
            if not self._is_numeric_param_value(
                value_a
            ) or not self._is_numeric_param_value(value_b):
                continue
            x_vals.append(float(value_a))
            y_vals.append(float(value_b))
            z_vals.append(float(item["mean_score"]))

        if not x_vals:
            return None

        fig = plt.figure(figsize=(7, 5))
        fig.patch.set_facecolor(plot_bg)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(plot_bg)
        ax.tick_params(colors=plot_fg)
        ax.xaxis.label.set_color(plot_fg)
        ax.yaxis.label.set_color(plot_fg)
        ax.zaxis.label.set_color(plot_fg)
        ax.title.set_color(plot_fg)

        scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap=cmap, s=35)  # type: ignore
        if len(x_vals) >= 3 and len(set(x_vals)) >= 2 and len(set(y_vals)) >= 2:
            ax.plot_trisurf(x_vals, y_vals, z_vals, cmap=cmap, alpha=0.35)

        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.12)
        colorbar.set_label(f"Mean CV {metric_name}", color=plot_fg)
        colorbar.ax.tick_params(colors=plot_fg)

        ax.set_xlabel(param_a)
        ax.set_ylabel(param_b)
        ax.set_zlabel(f"Mean CV {metric_name}")
        ax.set_title(f"Tuning Interaction: {param_a} vs {param_b}")
        ax.view_init(elev=25, azim=135)
        fig.tight_layout()
        return fig

    def _build_numeric_categorical_tuning_plot(
        self,
        combos: list[dict],
        numeric_param: str,
        categorical_param: str,
        metric_name: str,
        plot_fg: str,
        plot_bg: str,
        cmap: str,
    ):
        series = {}
        x_values_set = set()
        for item in combos:
            numeric_value = item["values"].get(numeric_param)
            category_value = item["values"].get(categorical_param)
            if not self._is_numeric_param_value(numeric_value):
                continue

            x_value = float(numeric_value)
            x_values_set.add(x_value)
            category_key = repr(category_value)
            if category_key not in series:
                series[category_key] = {"label": str(category_value), "scores": {}}
            series[category_key]["scores"][x_value] = float(item["mean_score"])

        if not series or not x_values_set:
            return None

        x_values = sorted(x_values_set)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        fig.patch.set_facecolor(plot_bg)
        ax.set_facecolor(plot_bg)
        ax.tick_params(colors=plot_fg)
        ax.xaxis.label.set_color(plot_fg)
        ax.yaxis.label.set_color(plot_fg)
        ax.title.set_color(plot_fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(plot_fg)

        series_keys = sorted(series.keys(), key=lambda key: series[key]["label"])
        color_points = np.linspace(0.15, 0.85, max(len(series_keys), 1))
        line_colors = [plt.get_cmap(cmap)(float(point)) for point in color_points]

        for i, key in enumerate(series_keys):
            series_data = series[key]
            y_values = [
                series_data["scores"].get(x_value, np.nan) for x_value in x_values
            ]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                label=series_data["label"],
                color=line_colors[i],
            )

        if len(series) > 1:
            legend = ax.legend(title=categorical_param)
            if legend is not None:
                legend.get_title().set_color(plot_fg)
                for text in legend.get_texts():
                    text.set_color(plot_fg)

        ax.set_xlabel(numeric_param)
        ax.set_ylabel(f"Mean CV {metric_name}")
        ax.set_title(f"Score vs {numeric_param} by {categorical_param}")
        fig.tight_layout()
        return fig

    def _build_categorical_categorical_tuning_plot(
        self,
        combos: list[dict],
        param_a: str,
        param_b: str,
        metric_name: str,
        plot_fg: str,
        plot_bg: str,
        cmap: str,
    ):
        x_lookup = {}
        y_lookup = {}
        for item in combos:
            value_a = item["values"].get(param_a)
            value_b = item["values"].get(param_b)
            x_lookup[repr(value_a)] = value_a
            y_lookup[repr(value_b)] = value_b

        if not x_lookup or not y_lookup:
            return None

        x_keys = sorted(x_lookup.keys(), key=lambda key: str(x_lookup[key]))
        y_keys = sorted(y_lookup.keys(), key=lambda key: str(y_lookup[key]))
        matrix = np.full((len(y_keys), len(x_keys)), np.nan, dtype=float)
        x_index = {key: i for i, key in enumerate(x_keys)}
        y_index = {key: i for i, key in enumerate(y_keys)}

        for item in combos:
            value_a = item["values"].get(param_a)
            value_b = item["values"].get(param_b)
            row = y_index.get(repr(value_b))
            col = x_index.get(repr(value_a))
            if row is None or col is None:
                continue
            matrix[row, col] = float(item["mean_score"])

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        fig.patch.set_facecolor(plot_bg)
        ax.set_facecolor(plot_bg)
        ax.tick_params(colors=plot_fg)
        ax.xaxis.label.set_color(plot_fg)
        ax.yaxis.label.set_color(plot_fg)
        ax.title.set_color(plot_fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(plot_fg)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Mean CV {metric_name}", color=plot_fg)
        cbar.ax.tick_params(colors=plot_fg)

        ax.set_axisbelow(False)
        ax.set_xticks(np.arange(len(x_keys) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(y_keys) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=plot_fg, linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size:
            thresh = (float(np.max(finite_values)) + float(np.min(finite_values))) / 2.0
        else:
            thresh = 0.0
        cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if not np.isfinite(value):
                    continue

                text_color = cmap_max if value < thresh else cmap_min
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

        ax.set_xticks(np.arange(len(x_keys)))
        ax.set_xticklabels(
            [str(x_lookup[key]) for key in x_keys], rotation=30, ha="right"
        )
        ax.set_yticks(np.arange(len(y_keys)))
        ax.set_yticklabels([str(y_lookup[key]) for key in y_keys])
        ax.set_xlabel(param_a)
        ax.set_ylabel(param_b)
        ax.set_title(f"Heatmap: {param_a} x {param_b}")
        fig.tight_layout()
        return fig

    def _build_best_candidates_plot(
        self,
        combos: list[dict],
        param_a: str,
        param_b: str,
        metric_name: str,
        plot_fg: str,
        plot_bg: str,
        cmap: str,
        top_n: int = 10,
    ):
        if not combos:
            return None

        ranked = sorted(combos, key=lambda item: item["mean_score"], reverse=True)[
            :top_n
        ]
        labels = [
            f"{param_a}={item['values'].get(param_a)}, {param_b}={item['values'].get(param_b)}"
            for item in ranked
        ]
        scores = [float(item["mean_score"]) for item in ranked]

        fig, ax = plt.subplots(figsize=(7, 4.8))
        fig.patch.set_facecolor(plot_bg)
        ax.set_facecolor(plot_bg)
        ax.tick_params(colors=plot_fg)
        ax.xaxis.label.set_color(plot_fg)
        ax.yaxis.label.set_color(plot_fg)
        ax.title.set_color(plot_fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(plot_fg)

        y_pos = np.arange(len(labels))
        bar_points = np.linspace(0.15, 0.85, max(len(labels), 1))
        bar_colors = [plt.get_cmap(cmap)(float(point)) for point in bar_points]
        ax.barh(y_pos, scores, color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean CV {metric_name}")
        ax.set_title(f"Top {len(ranked)} Tuning Candidates")
        fig.tight_layout()
        return fig

    def _is_numeric_param_value(self, value) -> bool:
        if isinstance(value, bool) or value is None:
            return False
        return isinstance(value, (Number, np.integer, np.floating))

    def _get_handler(self, model) -> ModelHandler:
        """
        Return the appropriate handler for the model type.

        Args:
            model: Fitted scikit-learn compatible estimator.

        Returns:
            ModelHandler implementation matching the estimator type.
        """
        if is_clusterer(model):
            return ClusteringHandler()
        elif is_classifier(model):
            return ClassificationHandler()
        elif is_regressor(model):
            return RegressionHandler()
        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    def _get_model_display_name(self) -> str:
        final_step = getattr(self.model, "steps", None)
        if isinstance(final_step, list) and final_step:
            last_estimator = final_step[-1][1]
            return last_estimator.__class__.__name__
        return self.model.__class__.__name__

    def _serialize_param_value(self, value):
        if hasattr(value, "get_params") and not isinstance(value, type):
            return value.__class__.__name__

        if isinstance(value, tuple):
            return tuple(self._serialize_param_value(item) for item in value)

        if isinstance(value, list):
            return [self._serialize_param_value(item) for item in value]

        if isinstance(value, dict):
            return {
                key: self._serialize_param_value(item) for key, item in value.items()
            }

        rendered = str(value)
        if len(rendered) > 80:
            return f"{rendered[:77]}..."
        return value

    def _get_model_params(self) -> dict:
        return {
            key: self._serialize_param_value(value)
            for key, value in self.model.get_params().items()
            if key != "steps"
        }

    def _to_dict(self) -> dict:
        """
        Convert report state to a renderer-friendly dictionary.

        Returns:
            Report payload with metadata, model info, data summary, metrics,
            and tuning details.
        """
        first_split = next(iter(self._state.splits.values()))
        X, y, _ = first_split
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        splits_counts = {
            name: len(split[0]) for name, split in self._state.splits.items()
        }

        class_distribution = {}
        class_percentages = {}

        if isinstance(self._state.handler, ClassificationHandler):
            all_y = np.concatenate(
                [np.asarray(split[1]) for split in self._state.splits.values()]
            )
            labels = np.unique(all_y)
            total_count = len(all_y)

            for split_name, (_, split_y, _) in self._state.splits.items():
                split_arr = np.asarray(split_y)
                class_distribution[split_name] = {
                    str(label): int(np.sum(split_arr == label)) for label in labels
                }

            class_percentages = {
                str(label): float(np.sum(all_y == label) / total_count * 100.0)
                for label in labels
            }

        cv_folds = None
        if self._state.cv is not None:
            get_n_splits = getattr(self._state.cv, "get_n_splits", None)
            if callable(get_n_splits):
                cv_folds = get_n_splits(X, y)

        return {
            "meta": {
                "title": self.title,
                "author": self.author,
                "description": self.description,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "model": {
                "name": self._get_model_display_name(),
                "type": self._state.handler.__class__.__name__.replace("Handler", ""),
                "version": sklearn.__version__,
                "params": self._get_model_params(),
            },
            "data": {
                "features": n_features,
                "splits": splits_counts,
                "total": sum(splits_counts.values()),
                "cv_folds": cv_folds,
                "class_distribution": class_distribution,
                "class_percentages": class_percentages,
            },
            "metrics": self._state.metrics,
            "plots": self._state.plots,
            "tuning": {
                "summary": self._state.tuning["summary"],
                "plots": self._state.tuning["plots"],
            },
        }

    def to_dict(self) -> dict:
        """
        Convert the built report into a public renderer-friendly dictionary.

        Returns:
            Report payload with metadata, model info, data summary, metrics,
            and tuning details.
        """
        self._require_built()
        return self._to_dict()
