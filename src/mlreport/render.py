from __future__ import annotations

import base64
import json
import logging
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from .theme import get_style_css, get_template, get_theme_css


def _ensure_parent_dir(path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=100,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def fig_to_file(fig, path: str) -> str:
    target = _ensure_parent_dir(path)
    fig.savefig(
        target,
        dpi=100,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    return str(target)


def render_html(report_type: str, theme: str, context: dict, path: str | None = None):
    html_context = dict(context)
    html_context["base_css"] = get_style_css("base.css")
    html_context["page_css"] = get_style_css(f"{report_type}.css")
    html_context["theme_css"] = get_theme_css(theme)

    template = get_template(f"{report_type}.html")
    content = template.render(**html_context)

    if path is not None:
        _ensure_parent_dir(path).write_text(content)
        return True
    return content


def render_md(report_type: str, theme: str, context: dict, path: str | None = None):
    del theme
    template = get_template(f"{report_type}.md")
    content = template.render(**context)

    if path is not None:
        _ensure_parent_dir(path).write_text(content)
        return True
    return content


def render_json(report_type: str, theme: str, context: dict, path: str | None = None):
    del report_type, theme
    if path is not None:
        _ensure_parent_dir(path).write_text(json.dumps(context, indent=4))
        return True
    return context


def render_pdf(report_type: str, theme: str, context: dict, path: str | None = None):
    from weasyprint import HTML

    html = render_html(report_type, theme, context, path=None)
    warnings.filterwarnings("ignore")
    for name in ("weasyprint", "fontTools"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    if path is not None:
        HTML(string=html).write_pdf(str(_ensure_parent_dir(path)))
        return True
    return HTML(string=html).write_pdf()
