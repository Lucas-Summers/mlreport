from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"
THEME_DIR = TEMPLATE_DIR / "themes"
STYLE_DIR = TEMPLATE_DIR / "styles"


def get_template(
    name: str,
    *,
    trim_blocks: bool = False,
    lstrip_blocks: bool = False,
    **globals_,
):
    """
    Load a Jinja2 template by name from the packaged templates directory.

    Args:
        name: Template filename.

    Returns:
        Loaded Jinja2 template object.
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=trim_blocks,
        lstrip_blocks=lstrip_blocks,
    )
    env.globals.update(globals_)
    return env.get_template(name)


def get_theme_css(theme: str) -> str:
    """
    Load the active theme stylesheet contents.

    Args:
        theme: Theme name matching a packaged ``.css`` file.

    Returns:
        CSS source for the selected theme.
    """
    theme_path = THEME_DIR / f"{theme}.css"
    if not theme_path.exists():
        raise ValueError(f"Unknown theme: {theme}")
    return theme_path.read_text()


def get_style_css(name: str) -> str:
    """
    Load a shared stylesheet by name from the packaged styles directory.

    Args:
        name: Stylesheet filename.

    Returns:
        CSS source for the requested stylesheet.
    """
    style_path = STYLE_DIR / name
    if not style_path.exists():
        raise ValueError(f"Unknown stylesheet: {name}")
    return style_path.read_text()


def get_plot_colors(theme: str) -> tuple[str, str]:
    """
    Extract plot foreground and background colors from the active theme CSS.

    Args:
        theme: Theme name matching a packaged ``.css`` file.

    Returns:
        Foreground and background hex colors.
    """
    text = get_theme_css(theme)
    fg_match = re.search(r"--main-fg\s*:\s*(#[0-9a-fA-F]{3,8})", text)
    bg_match = re.search(r"--main-bg\s*:\s*(#[0-9a-fA-F]{3,8})", text)
    fg = fg_match.group(1) if fg_match else "#000000"
    bg = bg_match.group(1) if bg_match else "#ffffff"
    return (fg, bg)


def get_palette(cmap: str, n_colors: int) -> list:
    """
    Build a subtle sampled color palette from a matplotlib colormap.

    Args:
        cmap: Colormap name.
        n_colors: Number of colors to generate.

    Returns:
        List of RGBA tuples sampled from the colormap.
    """
    colormap = plt.get_cmap(cmap)
    if n_colors <= 1:
        stops = np.asarray([0.6], dtype=float)
    else:
        stops = np.linspace(0.15, 0.85, n_colors)
    return [colormap(float(stop)) for stop in stops]
