from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import is_classifier, is_clusterer, is_regressor
from sklearn.model_selection import check_cv, cross_val_predict

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
    render_txt,
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
    is_crossval: bool = False
    fold_ids: np.ndarray | None = None
    built: bool = False


class Report:
    """Build and render evaluation reports for fitted models or predictions."""

    def __init__(
        self,
        model,
        title: str | None = None,
        author: str | None = None,
        description: str | None = None,
        theme: str = "light",
        cmap: str = "viridis",
        model_type: str | None = None,
        model_params: dict | None = None,
    ):
        """
        Initialize a report around a model and optional display metadata.

        Args:
            model: Fitted sklearn-compatible estimator or custom model object.
            title: Optional report title.
            author: Optional report author.
            description: Optional report description.
            theme: Render theme name.
            cmap: Matplotlib colormap name for generated plots.
            model_type: Explicit model type for custom models.
            model_params: Optional parameter mapping for display.
        """
        self.model = model
        self.model_type = model_type
        self.model_params = model_params
        self.title = title
        self.author = author
        self.description = description
        self.theme = theme
        self.cmap = cmap

        self._state = ReportState(
            handler=self._get_handler(model, model_type=model_type)
        )

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
        if self._state.is_crossval:
            raise ValueError("Cannot add train/test splits after add_crossval().")

        if y_pred is None:
            predict = getattr(self.model, "predict", None)
            if not callable(predict):
                raise ValueError(
                    f"Model {type(self.model).__name__} does not expose predict(). "
                    "Pass y_pred explicitly to add_split(..., y_pred=...)."
                )

            y_pred = np.asarray(predict(X))

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
        cv: Any | None = None,
        fold_ids: np.ndarray | None = None,
    ) -> Report:
        """
        Add a cross-validation prediction set.

        Args:
            X: Full feature matrix used for cross-validation.
            y: Full true target values.
            y_pred: Optional out-of-fold predictions aligned with y.
            cv: Optional cross-validation splitter with split(X, y), fold
                count, or iterable of (train_idx, test_idx) pairs.
            fold_ids: Optional fold identifier for each row. Cannot be used
                together with cv.

        Returns:
            Self for method chaining.
        """
        if self._state.splits:
            raise ValueError("Cannot call add_crossval() after add_split().")

        if cv is not None and fold_ids is not None:
            raise ValueError("Pass either cv or fold_ids, not both.")

        splits = None
        fold_ids_arr = None
        n_samples = len(y)

        if cv is not None:
            splits = self._materialize_cv_splits(cv, X, y)
            fold_ids_arr = self._fold_ids_from_splits(n_samples, splits)
        elif fold_ids is not None:
            fold_ids_arr = self._validate_fold_ids(fold_ids, n_samples)
            splits = self._splits_from_fold_ids(fold_ids_arr)

        if y_pred is None:
            if splits is None:
                raise ValueError(
                    "cv or fold_ids must be provided when y_pred is not provided."
                )
            y_pred = np.asarray(cross_val_predict(self.model, X, y, cv=splits))
        else:
            y_pred = np.asarray(y_pred)
            if len(y_pred) != n_samples:
                raise ValueError(
                    "y_pred must contain one prediction for each target value."
                )

        self._state.is_crossval = True
        self._state.fold_ids = fold_ids_arr
        self._state.splits = {
            "cv": (
                np.asarray(X),
                np.asarray(y),
                np.asarray(y_pred),
            )
        }
        return self

    def _materialize_cv_splits(
        self, cv: Any, X: np.ndarray, y: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """
        Normalize a sklearn CV input into concrete train/test index arrays.

        Args:
            cv: Fold count, splitter object, or iterable of split pairs.
            X: Feature matrix passed to the splitter.
            y: Target values passed to the splitter.

        Returns:
            Tuple of ``(train_idx, test_idx)`` arrays.
        """
        splitter = check_cv(
            cv,
            y,
            classifier=isinstance(self._state.handler, ClassificationHandler),
        )
        raw_splits = list(splitter.split(X, y))  # type: ignore

        if not raw_splits:
            raise ValueError("cv must produce at least one split.")

        splits = []
        for split_pair in raw_splits:
            if len(split_pair) != 2:
                raise ValueError("Each cv split must be a (train_idx, test_idx) pair.")
            train_idx, test_idx = split_pair
            splits.append(
                (
                    np.asarray(train_idx, dtype=int),
                    np.asarray(test_idx, dtype=int),
                )
            )
        return tuple(splits)

    def _fold_ids_from_splits(
        self,
        n_samples: int,
        splits: tuple[tuple[np.ndarray, np.ndarray], ...],
    ) -> np.ndarray:
        """
        Convert train/test splits into one held-out fold id per sample.

        Args:
            n_samples: Number of rows in the full evaluation set.
            splits: Concrete CV splits.

        Returns:
            Array where each element is the held-out fold id for that row.
        """
        fold_ids = np.full(n_samples, -1, dtype=int)
        seen = np.zeros(n_samples, dtype=bool)

        for fold_idx, (_, test_idx) in enumerate(splits):
            if test_idx.size == 0:
                raise ValueError("cv test folds must not be empty.")
            if np.any((test_idx < 0) | (test_idx >= n_samples)):
                raise ValueError("cv test indices must be valid row positions.")
            if np.any(seen[test_idx]):
                raise ValueError("cv test folds must not overlap.")

            seen[test_idx] = True
            fold_ids[test_idx] = fold_idx

        if not np.all(seen):
            raise ValueError("cv must assign every sample to exactly one test fold.")

        return self._validate_fold_ids(fold_ids, n_samples)

    def _validate_fold_ids(self, fold_ids: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Validate user-provided fold assignments for out-of-fold predictions.

        Args:
            fold_ids: Fold id per target row.
            n_samples: Expected number of fold ids.

        Returns:
            The validated fold id array.
        """
        if fold_ids.ndim != 1:
            raise ValueError("fold_ids must be a one-dimensional array.")
        if len(fold_ids) != n_samples:
            raise ValueError("fold_ids must contain one fold id per target value.")
        if len(np.unique(fold_ids)) < 2:
            raise ValueError("fold_ids must contain at least two folds.")
        return fold_ids

    def _splits_from_fold_ids(
        self,
        fold_ids: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """
        Reconstruct train/test split pairs from row-level fold assignments.

        Args:
            fold_ids: Fold id per row.

        Returns:
            Tuple of ``(train_idx, test_idx)`` arrays.
        """
        row_indices = np.arange(len(fold_ids))
        splits = []
        for fold_id in np.unique(fold_ids):
            test_mask = fold_ids == fold_id
            splits.append((row_indices[~test_mask], row_indices[test_mask]))
        return tuple(splits)

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
            fold_ids=self._state.fold_ids,
        )

        self._state.plots = self._state.handler.build_plots(
            self._state.splits,
            self.theme,
            exclude_plots or [],
            cmap=self.cmap,
        )
        self._state.built = True
        return self

    def to_txt(self, path: str | None = None) -> Report:
        """
        Render report to TXT.

        Args:
            path: Optional output file path for the TXT report.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        context = self._to_dict()
        context.pop("plots", None)
        if isinstance(context.get("tuning"), dict):
            context["tuning"].pop("plots", None)

        content = render_txt("report", self.theme, context, path=path)
        if path is None:
            print(content)
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
        if path is None:
            raise TypeError("path is required. Pass an output path like 'report.html'.")

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
        if path is None:
            raise TypeError("path is required. Pass an output path like 'report.pdf'.")

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
        if path is None:
            raise TypeError("path is required. Pass an output path like 'report.json'.")

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
        if path is None:
            raise TypeError("path is required. Pass an output path like 'report.md'.")

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
        """
        Select the score column to use from a fitted search object's results.

        Args:
            cv_results: ``cv_results_`` mapping from a fitted search object.
            search_cv: Fitted sklearn search object.

        Returns:
            Name of the selected mean test score column.
        """
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
        """
        Infer a display name for the search scoring metric.

        Args:
            search_cv: Fitted sklearn search object.
            score_column: Selected score column from ``cv_results_``.

        Returns:
            Human-readable scoring metric name.
        """
        if score_column != "mean_test_score":
            return score_column.replace("mean_test_", "")

        scoring = getattr(search_cv, "scoring", None)
        if isinstance(scoring, str):
            return scoring
        if callable(scoring):
            return getattr(scoring, "__name__", "custom_score")

        return "score"

    def _get_search_param_names(self, cv_results: dict) -> list[str]:
        """
        Return sorted hyperparameter names present in search candidates.

        Args:
            cv_results: ``cv_results_`` mapping from a fitted search object.

        Returns:
            Sorted list of parameter names found in ``params`` entries.
        """
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
        """
        Build tuning visualizations for two searched hyperparameters.

        Args:
            cv_results: ``cv_results_`` mapping from a fitted search object.
            score_column: Selected mean test score column.
            metric_name: Display name for the search metric.
            colors: Plot foreground and background colors.
            param_names: Two tuned parameter names.
            cmap: Matplotlib colormap name.

        Returns:
            Mapping of tuning plot ids to plot metadata.
        """
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
        """
        Build a 3D tuning plot for two numeric hyperparameters.

        Args:
            combos: Aggregated parameter combinations and mean scores.
            param_a: First numeric parameter name.
            param_b: Second numeric parameter name.
            metric_name: Display name for the search metric.
            plot_fg: Plot foreground color.
            plot_bg: Plot background color.
            cmap: Matplotlib colormap name.

        Returns:
            Matplotlib figure, or ``None`` when no numeric points are available.
        """
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
        """
        Build a line plot for one numeric and one categorical hyperparameter.

        Args:
            combos: Aggregated parameter combinations and mean scores.
            numeric_param: Numeric parameter name for the x-axis.
            categorical_param: Categorical parameter name for series groups.
            metric_name: Display name for the search metric.
            plot_fg: Plot foreground color.
            plot_bg: Plot background color.
            cmap: Matplotlib colormap name.

        Returns:
            Matplotlib figure, or ``None`` when no compatible data exists.
        """
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
        """
        Build a heatmap for two categorical hyperparameters.

        Args:
            combos: Aggregated parameter combinations and mean scores.
            param_a: First categorical parameter name.
            param_b: Second categorical parameter name.
            metric_name: Display name for the search metric.
            plot_fg: Plot foreground color.
            plot_bg: Plot background color.
            cmap: Matplotlib colormap name.

        Returns:
            Matplotlib figure, or ``None`` when no compatible data exists.
        """
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

        masked_matrix = np.ma.masked_invalid(matrix)
        x_edges = np.arange(len(x_keys) + 1, dtype=float)
        y_edges = np.arange(len(y_keys) + 1, dtype=float)
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            masked_matrix,
            cmap=cmap,
            shading="flat",
            edgecolors=plot_fg,
            linewidth=1.0,
            antialiased=False,
        )
        ax.set_aspect("auto")
        ax.invert_yaxis()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Mean CV {metric_name}", color=plot_fg)
        cbar.ax.tick_params(colors=plot_fg)

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
                    col + 0.5,
                    row + 0.5,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

        ax.set_xticks(np.arange(len(x_keys), dtype=float) + 0.5)
        ax.set_xticklabels(
            [str(x_lookup[key]) for key in x_keys], rotation=30, ha="right"
        )
        ax.set_yticks(np.arange(len(y_keys), dtype=float) + 0.5)
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
        """
        Build a horizontal bar chart of the best tuning candidates.

        Args:
            combos: Aggregated parameter combinations and mean scores.
            param_a: First tuned parameter name.
            param_b: Second tuned parameter name.
            metric_name: Display name for the search metric.
            plot_fg: Plot foreground color.
            plot_bg: Plot background color.
            cmap: Matplotlib colormap name.
            top_n: Maximum number of candidates to show.

        Returns:
            Matplotlib figure, or ``None`` when no candidates exist.
        """
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
        """
        Return whether a search parameter value should be plotted as numeric.

        Args:
            value: Parameter value from a search candidate.

        Returns:
            ``True`` for numeric non-boolean values.
        """
        if isinstance(value, bool) or value is None:
            return False
        return isinstance(value, (Number, np.integer, np.floating))

    def _get_handler(self, model, model_type: str | None = None) -> ModelHandler:
        """
        Return the appropriate handler for the model type.

        Args:
            model: Fitted model or scikit-learn compatible estimator.
            model_type: Explicit model type for custom models. Sklearn models
                are detected automatically.

        Returns:
            ModelHandler implementation matching the estimator type.
        """
        if model_type is not None:
            return self._get_handler_from_model_type(model_type)

        try:
            if is_clusterer(model):
                return ClusteringHandler()
            elif is_classifier(model):
                return ClassificationHandler()
            elif is_regressor(model):
                return RegressionHandler()
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported model type: {type(model).__name__}. "
                "Pass model_type='classification' or model_type='regression' "
                "for custom models."
            ) from exc

        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    def _get_handler_from_model_type(self, model_type: str) -> ModelHandler:
        """
        Resolve an explicit model type string to a handler instance.

        Args:
            model_type: User-provided model type label.

        Returns:
            Handler for the requested model type.
        """
        normalized_type = model_type.lower().replace("-", "_").replace(" ", "_")
        if normalized_type in {"classification", "classifier"}:
            return ClassificationHandler()
        if normalized_type in {"regression", "regressor"}:
            return RegressionHandler()

        raise ValueError(
            f"Unsupported model_type: {model_type!r}. "
            "Expected 'classification' or 'regression'."
        )

    def _get_model_display_name(self) -> str:
        """
        Return the model name shown in the report.

        Returns:
            Final pipeline step class name, or the model class name.
        """
        final_step = getattr(self.model, "steps", None)
        if isinstance(final_step, list) and final_step:
            last_estimator = final_step[-1][1]
            return last_estimator.__class__.__name__
        return self.model.__class__.__name__

    def _serialize_param_value(self, value):
        """
        Convert model parameter values into renderer-friendly values.

        Args:
            value: Raw parameter value from ``get_params()`` or ``model_params``.

        Returns:
            Serialized parameter value suitable for text and template output.
        """
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
        """
        Return serialized parameters for the report model.

        Returns:
            Mapping of parameter names to display values.
        """
        if self.model_params is not None:
            return {
                key: self._serialize_param_value(value)
                for key, value in self.model_params.items()
            }

        if hasattr(self.model, "get_params"):
            return {
                key: self._serialize_param_value(value)
                for key, value in self.model.get_params().items()
                if key != "steps"
            }

        raise ValueError(
            f"Model {type(self.model).__name__} does not expose get_params(). "
            "Pass model_params={...} for custom models."
        )

    def _is_sklearn_model(self) -> bool:
        """
        Return whether the model is recognized by sklearn type checks.

        Returns:
            ``True`` when sklearn identifies the model as supported.
        """
        try:
            return (
                is_clusterer(self.model)
                or is_classifier(self.model)
                or is_regressor(self.model)
            )
        except AttributeError:
            return False

    def _get_sklearn_summary(self) -> str:
        """
        Return a display label for sklearn compatibility.

        Returns:
            Versioned sklearn label for sklearn models, otherwise ``False``.
        """
        if self._is_sklearn_model():
            return f"True (v{sklearn.__version__})"
        return "False"

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
        if self._state.fold_ids is not None:
            cv_folds = int(len(np.unique(self._state.fold_ids)))

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
                "sklearn": self._get_sklearn_summary(),
                "params": self._get_model_params(),
            },
            "data": {
                "features": n_features,
                "splits": splits_counts,
                "total": sum(splits_counts.values()),
                "cv_folds": cv_folds,
                "is_crossval": self._state.is_crossval,
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
