from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .render import fig_to_base64, fig_to_file, render_html, render_json, render_md, render_pdf
from .report import Report


@dataclass
class ComparisonState:
    split: str | None = None
    model_type: str | None = None
    models: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    plots: list[dict] = field(default_factory=list)
    built: bool = False


class ComparisonReport:
    def __init__(
        self,
        reports: list[Report],
        title: str | None = None,
        author: str | None = None,
        description: str | None = None,
        split: str | None = None,
        theme: str = "light",
        cmap: str = "viridis",
    ):
        self.reports = reports
        self.title = title
        self.author = author
        self.description = description
        self.split = split
        self.theme = theme
        self.cmap = cmap

        self._state = ComparisonState()

    def build(self) -> ComparisonReport:
        if len(self.reports) < 2:
            raise ValueError("ComparisonReport requires at least 2 reports.")

        payloads = [report.to_dict() for report in self.reports]
        model_types = {payload["model"]["type"] for payload in payloads}
        if len(model_types) != 1:
            raise ValueError(
                "All reports in a comparison must have the same model type."
            )

        split = self._resolve_split(payloads)
        metric_ids = self._get_common_scalar_metric_ids(payloads, split)
        if not metric_ids:
            raise ValueError(f"No shared scalar metrics found for split '{split}'.")

        model_keys = self._build_model_keys(payloads)
        descriptions = [self._get_report_description(payload) for payload in payloads]

        self._state.split = split
        self._state.model_type = next(iter(model_types))
        self._state.models = self._build_model_rows(payloads, model_keys, descriptions)
        self._state.metrics = self._build_metric_rows(
            payloads, model_keys, metric_ids, split
        )
        self._state.plots = self._build_plots(split)
        self._state.built = True
        return self

    def summary(self) -> ComparisonReport:
        self._require_built()

        print("Comparison Report")
        if self.title:
            print(f"  title: {self.title}")
        if self.author:
            print(f"  author: {self.author}")
        if self.description:
            print(f"  description: {self.description}")
        print(f"  baseline: {self._state.models[0]['key']}")
        print()

        print("Models")
        for i, model in enumerate(self._state.models, 1):
            suffix = " [baseline]" if model["is_baseline"] else ""
            detail = f": {model['description']}" if model["description"] else ""
            print(f"  {i}. {model['key']} ({model['name']}){suffix}{detail}")
        print()

        print("Metrics")
        for metric in self._state.metrics:
            parts = []
            baseline_key = metric["keys"][0]
            for key in metric["keys"]:
                value = metric["values"][key]
                if key == baseline_key:
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value:.4f} ({metric['deltas'][key]:+.4f})")
            parts.append(f"best={metric['best_key']}")
            print(f"  {metric['metric_name']}: {', '.join(parts)}")

        return self

    def to_html(self, path: str) -> ComparisonReport:
        self._require_built()

        context = self.to_dict()
        context["plots"] = self._serialize_plots()
        render_html("comparison", self.theme, context, path=path)

        return self

    def to_pdf(self, path: str) -> ComparisonReport:
        self._require_built()

        context = self.to_dict()
        context["plots"] = self._serialize_plots()
        render_pdf("comparison", self.theme, context, path=path)
        return self

    def to_json(self, path: str) -> ComparisonReport:
        self._require_built()

        render_json("comparison", self.theme, self.to_dict(), path=path)

        return self

    def to_md(self, path: str, image_dir: str | None = None) -> ComparisonReport:
        self._require_built()

        if image_dir is None:
            image_dir = str(Path(path).parent / "images")

        Path(image_dir).mkdir(exist_ok=True)

        context = self.to_dict()
        context["plots"] = self._serialize_plots(image_dir)
        render_md("comparison", self.theme, context, path=path)

        return self

    def to_dict(self) -> dict:
        self._require_built()

        return {
            "meta": {
                "title": self.title,
                "author": self.author,
                "description": self.description,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "comparison": {
                "baseline_key": self._state.models[0]["key"],
                "split": self._state.split,
            },
            "models": self._state.models,
            "metrics": self._state.metrics,
        }

    def _require_built(self) -> None:
        if not self._state.built:
            raise ValueError("Call build() first.")

    def _resolve_split(self, payloads: list[dict]) -> str:
        available_split_sets = []
        for payload in payloads:
            metric_values = payload["metrics"]
            if not metric_values:
                raise ValueError("Compared reports must include built metrics.")

            first_metric = next(iter(metric_values.values()))
            split_names = set(first_metric["values"].keys()) - {"per_class"}
            available_split_sets.append(split_names)

        common_splits = set.intersection(*available_split_sets)
        if not common_splits:
            raise ValueError("Compared reports do not share a common metric split.")

        if self.split is not None:
            if self.split not in common_splits:
                raise ValueError(
                    f"Split '{self.split}' is not shared by all compared reports."
                )
            return self.split

        if "test" in common_splits:
            return "test"
        if "cv" in common_splits:
            return "cv"
        return sorted(common_splits)[0]

    def _get_common_scalar_metric_ids(
        self, payloads: list[dict], split: str
    ) -> list[str]:
        baseline_metrics = payloads[0]["metrics"]
        common_ids = []
        for metric_id in baseline_metrics:
            if all(
                metric_id in payload["metrics"]
                and self._can_extract_metric_value(payload["metrics"][metric_id], split)
                for payload in payloads
            ):
                common_ids.append(metric_id)
        return common_ids

    def _build_model_rows(
        self,
        payloads: list[dict],
        model_keys: list[str],
        descriptions: list[str | None],
    ) -> list[dict]:
        rows = []
        for i, (payload, model_key, description) in enumerate(
            zip(payloads, model_keys, descriptions)
        ):
            params = payload["model"].get("params", {})
            cv_folds = payload.get("data", {}).get("cv_folds")
            if cv_folds is not None:
                data_label = f"{cv_folds}-fold CV"
            else:
                data_label = "Train/Test"

            tuning_summary = payload.get("tuning", {}).get("summary") or {}
            best_params = tuning_summary.get("best_params", {})
            if isinstance(best_params, dict) and best_params:
                tuned_count = len(best_params)
                tuned_label = f"{tuned_count} param" if tuned_count == 1 else f"{tuned_count} params"
            else:
                tuned_label = "None"
            rows.append(
                {
                    "index": i,
                    "key": model_key,
                    "description": description,
                    "name": payload["model"]["name"],
                    "type": payload["model"]["type"],
                    "data_label": data_label,
                    "tuned_label": tuned_label,
                    "params": params,
                    "param_count": len(params) if isinstance(params, dict) else 0,
                    "is_baseline": i == 0,
                }
            )
        return rows

    def _build_model_keys(self, payloads: list[dict]) -> list[str]:
        model_names = [payload["model"]["name"] for payload in payloads]
        name_counts = Counter(model_names)
        seen_counts: dict[str, int] = {}
        model_keys = []

        for model_name in model_names:
            if name_counts[model_name] == 1:
                model_keys.append(model_name)
                continue

            occurrence = seen_counts.get(model_name, 0) + 1
            seen_counts[model_name] = occurrence
            model_keys.append(f"{model_name} ({occurrence})")

        return model_keys

    def _build_metric_rows(
        self,
        payloads: list[dict],
        model_keys: list[str],
        metric_ids: list[str],
        split: str,
    ) -> list[dict]:
        rows = []
        baseline_key = model_keys[0]
        for metric_id in metric_ids:
            baseline_metric = payloads[0]["metrics"][metric_id]
            values = {
                model_key: self._get_metric_value(payload["metrics"][metric_id], split)
                for model_key, payload in zip(model_keys, payloads)
            }
            baseline_value = values[baseline_key]
            direction = baseline_metric.get("direction", "max")
            if direction == "min":
                best_key = min(values, key=lambda key: values[key])
            else:
                best_key = max(values, key=lambda key: values[key])
            rows.append(
                {
                    "metric_id": metric_id,
                    "metric_name": baseline_metric["name"],
                    "direction": direction,
                    "keys": model_keys,
                    "values": values,
                    "deltas": {
                        model_key: float(value - baseline_value)
                        for model_key, value in values.items()
                    },
                    "best_key": best_key,
                }
            )
        return rows

    def _build_plots(self, split: str) -> list[dict]:
        plot_ids = self._get_comparison_plot_ids()
        if not plot_ids:
            return []

        plot_groups = []
        for plot_id in plot_ids:
            cards = []
            for model, report in zip(self._state.models, self.reports):
                split_payload = {split: report._state.splits[split]}
                plots = report._state.handler.build_plots(
                    split_payload,
                    self.theme,
                    exclude=[candidate for candidate in report._state.handler._plots() if candidate != plot_id],
                    cmap=self.cmap,
                )
                plot_data = plots.get(plot_id)
                if plot_data is None:
                    continue
                if fig_axes := plot_data["fig"].axes:
                    fig_axes[0].set_title(model["key"])
                cards.append(
                    {
                        "model_key": model["key"],
                        "name": plot_data["name"],
                        "fig": plot_data["fig"],
                    }
                )
            if cards:
                plot_groups.append(
                    {
                        "plot_id": plot_id,
                        "name": cards[0]["name"],
                        "cards": cards,
                    }
                )
        return plot_groups

    def _get_comparison_plot_ids(self) -> list[str]:
        if self._state.model_type == "Classification":
            return ["confusion_matrix", "per_class_metrics"]
        if self._state.model_type == "Regression":
            return ["predicted_vs_actual", "residual_hist"]
        return []

    def _serialize_plots(self, image_dir: str | None = None) -> list[dict]:
        return [
            {
                "plot_id": group["plot_id"],
                "name": group["name"],
                "cards": [
                    (
                        {
                            "model_key": card["model_key"],
                            "name": card["name"],
                            "image": fig_to_base64(card["fig"]),
                        }
                        if image_dir is None
                        else {
                            "model_key": card["model_key"],
                            "name": card["name"],
                            "path": fig_to_file(
                                card["fig"],
                                str(
                                    Path(image_dir)
                                    / (
                                        "comparison_"
                                        f"{group['plot_id']}_"
                                        f"{''.join(char.lower() if char.isalnum() else '_' for char in card['model_key']).strip('_')}.png"
                                    )
                                ),
                            ),
                        }
                    )
                    for card in group["cards"]
                ],
            }
            for group in self._state.plots
        ]

    def _can_extract_metric_value(self, metric_payload: dict, split: str) -> bool:
        try:
            self._get_metric_value(metric_payload, split)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _get_metric_value(self, metric_payload: dict, split: str) -> float:
        value = metric_payload["values"][split]
        if split == "cv":
            return float(value["mean"])
        return float(value)

    def _get_report_description(self, payload: dict) -> str | None:
        meta_description = payload.get("meta", {}).get("description")
        if meta_description:
            return str(meta_description)
        return None
