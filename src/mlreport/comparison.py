from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .render import (
    fig_to_base64,
    fig_to_file,
    render_html,
    render_json,
    render_md,
    render_pdf,
    render_txt,
)
from .report import Report


@dataclass
class ComparisonState:
    split: str | None = None
    mixed_splits: bool = False
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
        """
        Initialize a comparison report from already-built model reports.

        Args:
            reports: Reports to compare.
            title: Optional comparison title.
            author: Optional report author.
            description: Optional comparison description.
            split: Optional split name to compare across all reports.
            theme: Render theme name.
            cmap: Matplotlib colormap name for comparison plots.
        """
        self.reports = reports
        self.title = title
        self.author = author
        self.description = description
        self.split = split
        self.theme = theme
        self.cmap = cmap

        self._state = ComparisonState()

    def build(self) -> ComparisonReport:
        """
        Validate reports and build comparison tables and plots.

        Returns:
            Self for method chaining.
        """
        if len(self.reports) < 2:
            raise ValueError("ComparisonReport requires at least 2 reports.")

        payloads = [report.to_dict() for report in self.reports]
        model_types = {payload["model"]["type"] for payload in payloads}
        if len(model_types) != 1:
            raise ValueError(
                "All reports in a comparison must have the same model type."
            )

        model_keys = self._build_model_keys(payloads)
        splits = [self._resolve_split(payload) for payload in payloads]
        metric_ids = self._get_common_scalar_metric_ids(payloads, splits)
        if not metric_ids:
            raise ValueError(
                "No compatible scalar metrics found across compared reports."
            )
        descriptions = [self._get_report_description(payload) for payload in payloads]

        self._state.split = splits[0] if len(set(splits)) == 1 else None
        self._state.mixed_splits = len(set(splits)) > 1
        self._state.model_type = next(iter(model_types))
        self._state.models = self._build_model_rows(
            payloads, model_keys, descriptions, splits
        )
        self._state.metrics = self._build_metric_rows(
            payloads, model_keys, metric_ids, splits
        )
        self._state.plots = self._build_plots(splits)
        self._state.built = True
        return self

    def to_txt(self, path: str | None = None) -> ComparisonReport:
        """
        Render the comparison report to TXT.

        Args:
            path: Optional output file path.

        Returns:
            Self for method chaining.
        """
        self._require_built()

        context = self.to_dict()
        content = render_txt("comparison", self.theme, context, path=path)
        if path is None:
            print(content)
        return self

    def to_html(
        self, path: str | None = None, include_model_reports: bool = True
    ) -> ComparisonReport | str:
        """
        Render the comparison report to HTML.

        Args:
            path: Optional output file path. When omitted, the rendered HTML
                string is returned.
            include_model_reports: Whether to append each source model report
                after the comparison report.

        Returns:
            Self for method chaining when written to a path, otherwise the
            rendered HTML string.
        """
        self._require_built()

        context = self.to_dict()
        context["plots"] = self._serialize_plots()
        context["model_reports"] = (
            self._render_model_report_html_fragments() if include_model_reports else []
        )
        if path is None:
            content = render_html("comparison", self.theme, context, path=None)
            if not isinstance(content, str):
                raise TypeError("render_html() did not return HTML content.")
            return content

        render_html("comparison", self.theme, context, path=path)
        return self

    def to_pdf(self, path: str, include_model_reports: bool = True) -> ComparisonReport:
        """
        Render the comparison report to PDF.

        Args:
            path: Output file path.
            include_model_reports: Whether to append each source model report
                after the comparison report.

        Returns:
            Self for method chaining.
        """
        self._require_built()
        if path is None:
            raise TypeError(
                "path is required. Pass an output path like 'comparison.pdf'."
            )

        context = self.to_dict()
        context["plots"] = self._serialize_plots()
        context["model_reports"] = (
            self._render_model_report_html_fragments() if include_model_reports else []
        )
        render_pdf("comparison", self.theme, context, path=path)
        return self

    def to_json(
        self, path: str | None = None, include_model_reports: bool = True
    ) -> ComparisonReport | str:
        """
        Render the comparison report to JSON.

        Args:
            path: Optional output file path. When omitted, the rendered JSON
                string is returned.
            include_model_reports: Whether to include each source model report
                in a ``model_reports`` object keyed by model index.

        Returns:
            Self for method chaining when written to a path, otherwise the
            rendered JSON string.
        """
        self._require_built()

        data = self.to_dict()
        if include_model_reports:
            data["model_reports"] = self._render_model_report_json_payloads()

        if path is None:
            return json.dumps(data, indent=4)

        render_json("comparison", self.theme, data, path=path)

        return self

    def to_md(
        self,
        path: str | None = None,
        image_dir: str | None = None,
        include_model_reports: bool = True,
    ) -> ComparisonReport | str:
        """
        Render the comparison report to Markdown.

        Args:
            path: Optional output file path. When omitted, the rendered
                Markdown string is returned.
            image_dir: Optional directory for exported plot images.
            include_model_reports: Whether to append each source model report
                after the comparison report.

        Returns:
            Self for method chaining when written to a path, otherwise the
            rendered Markdown string.
        """
        self._require_built()

        if image_dir is None:
            image_dir = str(Path(path).parent / "images") if path else "images"

        Path(image_dir).mkdir(parents=True, exist_ok=True)

        context = self.to_dict()
        context["plots"] = self._serialize_plots(image_dir)
        context["model_reports"] = (
            self._render_model_report_md_fragments(image_dir)
            if include_model_reports
            else []
        )
        if path is None:
            content = render_md("comparison", self.theme, context, path=None)
            if not isinstance(content, str):
                raise TypeError("render_md() did not return Markdown content.")
            return content

        render_md("comparison", self.theme, context, path=path)
        return self

    def to_dict(self) -> dict:
        """
        Convert the built comparison into a renderer-friendly dictionary.

        Returns:
            Report payload with metadata, model rows, and metric rows.
        """
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
                "mixed_splits": self._state.mixed_splits,
            },
            "models": self._state.models,
            "metrics": self._state.metrics,
        }

    def _require_built(self) -> None:
        """
        Validate that ``build()`` has been called.
        """
        if not self._state.built:
            raise ValueError("Call build() first.")

    def _resolve_split(self, payload: dict) -> str:
        """
        Resolve which split should be compared for one report payload.

        Args:
            payload: Built report payload.

        Returns:
            Split name selected for comparison.
        """
        metric_values = payload["metrics"]
        if not metric_values:
            raise ValueError("Compared reports must include built metrics.")

        first_metric = next(iter(metric_values.values()))
        split_names = set(first_metric["values"].keys()) - {"per_class"}

        if self.split is not None:
            if self.split not in split_names:
                raise ValueError(
                    f"Split '{self.split}' is not available for all compared reports."
                )
            return self.split

        if "test" in split_names:
            return "test"
        if "cv" in split_names:
            return "cv"
        raise ValueError("Compared reports must include either a 'test' or 'cv' split.")

    def _get_common_scalar_metric_ids(
        self, payloads: list[dict], splits: list[str]
    ) -> list[str]:
        """
        Return metric ids that can be compared as scalar values for all reports.

        Args:
            payloads: Built report payloads.
            splits: Resolved split name for each payload.

        Returns:
            Metric ids common to all reports and extractable as scalars.
        """
        baseline_metrics = payloads[0]["metrics"]
        common_ids = []
        for metric_id in baseline_metrics:
            if all(
                metric_id in payload["metrics"]
                and self._can_extract_metric_value(payload["metrics"][metric_id], split)
                for payload, split in zip(payloads, splits)
            ):
                common_ids.append(metric_id)
        return common_ids

    def _build_model_rows(
        self,
        payloads: list[dict],
        model_keys: list[str],
        descriptions: list[str | None],
        splits: list[str],
    ) -> list[dict]:
        """
        Build display rows describing each compared model.

        Args:
            payloads: Built report payloads.
            model_keys: Unique display keys for each model.
            descriptions: Optional descriptions from the source reports.
            splits: Resolved split name for each report.

        Returns:
            List of model row dictionaries for rendering.
        """
        rows = []
        for i, (payload, model_key, description, split) in enumerate(
            zip(payloads, model_keys, descriptions, splits)
        ):
            params = payload["model"].get("params", {})
            cv_folds = payload.get("data", {}).get("cv_folds")
            if split == "cv" and cv_folds is not None:
                data_label = f"{cv_folds}-fold CV"
            elif split == "cv":
                data_label = "CV Predictions"
            elif split == "test":
                data_label = "Train/Test Split"
            else:
                data_label = split.capitalize()

            tuning_summary = payload.get("tuning", {}).get("summary") or {}
            best_params = tuning_summary.get("best_params", {})
            if isinstance(best_params, dict) and best_params:
                tuned_count = len(best_params)
                tuned_label = (
                    f"{tuned_count} param"
                    if tuned_count == 1
                    else f"{tuned_count} params"
                )
            else:
                tuned_label = "None"
            rows.append(
                {
                    "index": i,
                    "key": model_key,
                    "title_name": model_key,
                    "description": description,
                    "name": payload["model"]["name"],
                    "type": payload["model"]["type"],
                    "comparison_split": split,
                    "data_label": data_label,
                    "tuned_label": tuned_label,
                    "params": params,
                    "param_count": len(params) if isinstance(params, dict) else 0,
                    "is_baseline": i == 0,
                }
            )
        return rows

    def _build_model_keys(self, payloads: list[dict]) -> list[str]:
        """
        Create unique model keys for display and metric lookups.

        Args:
            payloads: Built report payloads.

        Returns:
            Unique model keys, preserving model names when possible.
        """
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
        splits: list[str],
    ) -> list[dict]:
        """
        Build comparison rows for each shared scalar metric.

        Args:
            payloads: Built report payloads.
            model_keys: Unique display keys for each model.
            metric_ids: Metric ids to compare.
            splits: Resolved split name for each report.

        Returns:
            List of metric row dictionaries with values, deltas, and best model.
        """
        rows = []
        baseline_key = model_keys[0]
        for metric_id in metric_ids:
            baseline_metric = payloads[0]["metrics"][metric_id]
            values = {
                model_key: self._get_metric_value(payload["metrics"][metric_id], split)
                for model_key, payload, split in zip(model_keys, payloads, splits)
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
                    "splits": {
                        model_key: split for model_key, split in zip(model_keys, splits)
                    },
                    "deltas": {
                        model_key: float(value - baseline_value)
                        for model_key, value in values.items()
                    },
                    "best_key": best_key,
                    "best_index": model_keys.index(best_key) + 1,
                }
            )
        return rows

    def _build_plots(self, splits: list[str]) -> list[dict]:
        """
        Build grouped comparison plots for compatible model types.

        Args:
            splits: Resolved split name for each report.

        Returns:
            Plot groups, each containing one plot card per compared report.
        """
        plot_ids = self._get_comparison_plot_ids()
        if not plot_ids:
            return []

        plot_groups = []
        for plot_id in plot_ids:
            cards = []
            for model, report, split in zip(self._state.models, self.reports, splits):
                split_payload = {split: report._state.splits[split]}
                plots = report._state.handler.build_plots(
                    split_payload,
                    self.theme,
                    exclude=[
                        candidate
                        for candidate in report._state.handler._plots()
                        if candidate != plot_id
                    ],
                    cmap=self.cmap,
                )
                plot_data = plots.get(plot_id)
                if plot_data is None:
                    continue
                if fig_axes := plot_data["fig"].axes:
                    fig_axes[0].set_title(
                        f"{model['title_name']} [Model {model['index'] + 1}]"
                    )
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
        """
        Return plot ids appropriate for the compared model type.

        Returns:
            Plot ids to render for the comparison.
        """
        if self._state.model_type == "Classification":
            return ["confusion_matrix", "per_class_metrics"]
        if self._state.model_type == "Regression":
            return ["predicted_vs_actual", "residual_hist"]
        return []

    def _serialize_plots(self, image_dir: str | None = None) -> list[dict]:
        """
        Convert comparison plot figures to embedded images or files.

        Args:
            image_dir: Optional directory to write plot image files. When omitted,
                plots are embedded as base64 strings.

        Returns:
            Serialized plot group payloads for renderers.
        """
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

    def _render_model_report_html_fragments(self) -> list[str]:
        """
        Render source model reports as HTML fragments for appended HTML output.

        Returns:
            List of report container HTML fragments.
        """
        fragments = []
        for index, report in enumerate(self.reports):
            html = report.to_html(title_prefix=f"[Model {index + 1}] ")
            if not isinstance(html, str):
                raise TypeError("Report.to_html() did not return HTML content.")
            fragments.append(self._extract_model_report_container(html))
        return fragments

    def _render_model_report_json_payloads(self) -> dict[str, dict]:
        """
        Render source model reports as JSON payloads for comparison JSON output.

        Returns:
            Mapping from ``Model X`` keys to report JSON payloads.
        """
        payloads = {}
        for index, report in enumerate(self.reports):
            model_key = f"Model {index + 1}"
            content = report.to_json(title_prefix=f"[{model_key}] ")
            if not isinstance(content, str):
                raise TypeError("Report.to_json() did not return JSON content.")
            payloads[model_key] = json.loads(content)
        return payloads

    def _render_model_report_md_fragments(self, image_dir: str) -> list[str]:
        """
        Render source model reports as Markdown fragments for appended MD output.

        Args:
            image_dir: Base directory for exported model report plot images.

        Returns:
            List of model report Markdown fragments.
        """
        fragments = []
        for index, report in enumerate(self.reports):
            model_key = f"Model {index + 1}"
            content = report.to_md(
                image_dir=str(Path(image_dir) / f"model_{index + 1}"),
                title_prefix=f"[{model_key}] ",
            )
            if not isinstance(content, str):
                raise TypeError("Report.to_md() did not return Markdown content.")
            fragments.append(self._remove_report_md_footer(content))
        return fragments

    def _extract_model_report_container(self, html: str) -> str:
        """
        Extract the visible report container from a full report HTML document.

        Args:
            html: Full rendered report HTML.

        Returns:
            HTML fragment containing the model report container.
        """
        start_marker = '<div class="container">'
        start = html.find(start_marker)
        if start == -1:
            return html

        lightbox_marker = '<div class="lightbox"'
        end = html.find(lightbox_marker, start)
        if end == -1:
            end = html.rfind("</body>")
        if end == -1:
            end = len(html)

        fragment = html[start:end].strip()
        fragment = self._remove_report_footer(fragment)
        return fragment.replace(
            start_marker,
            '<div class="container model-report-container">',
            1,
        )

    def _remove_report_footer(self, html: str) -> str:
        """
        Remove the standalone report footer from an appended report fragment.

        Args:
            html: Extracted report container HTML.

        Returns:
            Report container HTML without its footer block.
        """
        footer_start = html.rfind("<footer>")
        footer_end = html.rfind("</footer>")
        if footer_start == -1 or footer_end == -1 or footer_end < footer_start:
            return html
        return (
            html[:footer_start].rstrip()
            + "\n"
            + html[footer_end + len("</footer>") :].lstrip()
        )

    def _remove_report_md_footer(self, markdown: str) -> str:
        """
        Remove the standalone report footer from an appended Markdown fragment.

        Args:
            markdown: Rendered report Markdown.

        Returns:
            Markdown without the trailing generated footer block.
        """
        footer_marker = "\n---\n\n*Generated at "
        footer_start = markdown.rfind(footer_marker)
        if footer_start == -1:
            return markdown.strip()
        return markdown[:footer_start].rstrip()

    def _can_extract_metric_value(self, metric_payload: dict, split: str) -> bool:
        """
        Return whether a metric payload has a comparable scalar for a split.

        Args:
            metric_payload: Built metric payload from a report.
            split: Split name to extract.

        Returns:
            ``True`` when the metric can be converted to a scalar value.
        """
        try:
            self._get_metric_value(metric_payload, split)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _get_metric_value(self, metric_payload: dict, split: str) -> float:
        """
        Extract a scalar metric value for a split.

        Args:
            metric_payload: Built metric payload from a report.
            split: Split name to extract.

        Returns:
            Scalar metric value. For fold-summary CV metrics, this is the mean.
        """
        value = metric_payload["values"][split]
        if split == "cv" and isinstance(value, dict):
            return float(value["mean"])
        return float(value)

    def _get_report_description(self, payload: dict) -> str | None:
        """
        Extract a report description for comparison model cards.

        Args:
            payload: Built report payload.

        Returns:
            Description string when present.
        """
        meta_description = payload.get("meta", {}).get("description")
        if meta_description:
            return str(meta_description)
        return None
