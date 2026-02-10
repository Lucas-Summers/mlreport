from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from jinja2 import Environment, FileSystemLoader
from sklearn.base import ClusterMixin, is_classifier, is_regressor

from .handlers.base import ModelHandler
from .handlers.classification import ClassificationHandler
from .handlers.clustering import ClusteringHandler
from .handlers.regression import RegressionHandler

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class ReportState:
    handler: ModelHandler
    splits: dict[str, tuple] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    built: bool = False


class Report:
    def __init__(
        self,
        model,
        title: str | None = None,
        author: str | None = None,
        description: str | None = None,
        theme: str = "light",
    ):
        self.model = model
        self.title = title
        self.author = author
        self.description = description
        self.theme = theme

        self._state = ReportState(handler=self._get_handler(model))

    def available_metrics(self) -> None:
        """Print available metrics for this report's model type."""
        for i, (id, name) in enumerate(self._state.handler._metrics().items(), 1):
            print(f"  {i}. {id} — {name}")

    def available_plots(self) -> None:
        """Print available plots for this report's model type."""
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

        If y_pred is not provided, model.predict(X) is called.
        Returns self for chaining.
        """
        if y_pred is None:
            y_pred = self.model.predict(X)

        self._state.splits[name] = (
            np.asarray(X),
            np.asarray(y),
            np.asarray(y_pred),
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

        Returns self for chaining.
        """
        if not self._state.splits:
            raise ValueError("No data splits added. Use add_split() first.")

        self._state.metrics = self._state.handler.compute_metrics(
            self._state.splits, exclude_metrics or []
        )
        self._state.plots = self._state.handler.generate_plots(
            self._state.splits, self._get_plot_colors(), exclude_plots or []
        )
        self._state.built = True
        return self

    def summary(self) -> Report:
        """Print a text summary of the report. Returns self for chaining."""
        self._require_built()

        print(f"Model: {self.model.__class__.__name__}")
        print(f"Splits: {', '.join(self._state.splits.keys())}")
        print()

        for _, metric_data in self._state.metrics.items():
            print(f"{metric_data['name']}:")
            for split_name, value in metric_data["values"].items():
                print(f"  {split_name}: {value:.4f}")
            print()

        return self

    def to_html(self, path: str) -> Report:
        """Render report to HTML. Returns self for chaining."""
        self._require_built()

        plots_with_images = {
            plot_id: {
                "name": plot_data["name"],
                "image": self._fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.plots.items()
        }

        context = self._to_dict()
        context["plots"] = plots_with_images

        template = self._get_template(f"{self.theme}.html")
        html = template.render(**context)

        with open(path, "w") as f:
            f.write(html)

        return self

    def to_pdf(self, path: str) -> Report:
        """Render report to PDF. Returns self for chaining."""
        self._require_built()

        from weasyprint import HTML

        plots_with_images = {
            plot_id: {
                "name": plot_data["name"],
                "image": self._fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.plots.items()
        }

        context = self._to_dict()
        context["plots"] = plots_with_images

        template = self._get_template(f"{self.theme}.html")
        html = template.render(**context)

        HTML(string=html).write_pdf(path)
        return self

    def to_json(self, path: str) -> Report:
        """Render report to JSON. Returns self for chaining."""
        self._require_built()

        plots_with_images = {
            plot_id: {
                "name": plot_data["name"],
                "image": self._fig_to_base64(plot_data["fig"]),
            }
            for plot_id, plot_data in self._state.plots.items()
        }

        data = self._to_dict()
        data["plots"] = plots_with_images

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        return self

    def to_md(self, path: str, image_dir: str | None = None) -> Report:
        """Render report to Markdown. Returns self for chaining."""
        self._require_built()

        if image_dir is None:
            image_dir = str(Path(path).parent / "images")

        Path(image_dir).mkdir(exist_ok=True)

        plots_with_paths = {}
        for plot_id, plot_data in self._state.plots.items():
            img_path = f"{image_dir}/{plot_id}.png"
            self._fig_to_file(plot_data["fig"], img_path)
            plots_with_paths[plot_id] = {
                "name": plot_data["name"],
                "path": img_path,
            }

        context = self._to_dict()
        context["plots"] = plots_with_paths

        template = self._get_template("report.md")
        md = template.render(**context)

        with open(path, "w") as f:
            f.write(md)

        return self

    def _require_built(self) -> None:
        if not self._state.built:
            raise ValueError("Call build() first.")

    def _get_plot_colors(self) -> tuple[str, str]:
        """Extract --main-fg and --main-bg CSS variables from the template.

        Returns (foreground, background) hex colors.
        """
        template_path = TEMPLATE_DIR / f"{self.theme}.html"
        text = template_path.read_text()
        fg_match = re.search(r"--main-fg\s*:\s*(#[0-9a-fA-F]{3,8})", text)
        bg_match = re.search(r"--main-bg\s*:\s*(#[0-9a-fA-F]{3,8})", text)
        fg = fg_match.group(1) if fg_match else "#000000"
        bg = bg_match.group(1) if bg_match else "#ffffff"
        return (fg, bg)

    def _get_handler(self, model) -> ModelHandler:
        """Return the appropriate handler for the model type."""
        if isinstance(model, ClusterMixin):
            return ClusteringHandler()
        elif is_classifier(model):
            return ClassificationHandler()
        elif is_regressor(model):
            return RegressionHandler()
        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    def _get_template(self, name: str):
        """Load a Jinja2 template."""
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        return env.get_template(name)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def _fig_to_file(self, fig, path: str) -> str:
        """Save matplotlib figure to file."""
        fig.savefig(
            path,
            dpi=100,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)
        return path

    def _to_dict(self) -> dict:
        """Convert report to dictionary format for renderers."""
        first_split = next(iter(self._state.splits.values()))
        X = first_split[0]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        splits_counts = {
            name: len(split[0]) for name, split in self._state.splits.items()
        }
        data_info = {
            "features": n_features,
            "splits": splits_counts,
            "total": sum(splits_counts.values()),
        }

        return {
            "meta": {
                "title": self.title,
                "author": self.author,
                "description": self.description,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "model": {
                "name": self.model.__class__.__name__,
                "type": self._state.handler.__class__.__name__.replace("Handler", ""),
                "version": sklearn.__version__,
                "params": self.model.get_params(),
            },
            "data": data_info,
            "metrics": self._state.metrics,
            "plots": self._state.plots,
        }
