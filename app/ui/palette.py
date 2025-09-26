from __future__ import annotations
"""
Palette helpers for Fusion style.

Provides a light and a dark palette tuned for the Fusion style so the app can
look uniform across platforms while offering a simple theme toggle.

Usage
  from app.ui.palette import apply_palette, Theme
  apply_palette(qApp, Theme.LIGHT)  # or Theme.DARK
"""
from enum import Enum
from PySide6.QtGui import QPalette, QColor


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"


def make_dark_palette() -> QPalette:
    """Return a dark QPalette (based on Qt's common Fusion dark example)."""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(53, 53, 53))
    p.setColor(QPalette.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.Base, QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ToolTipBase, QColor(250, 250, 250))
    p.setColor(QPalette.ToolTipText, QColor(20, 20, 20))
    p.setColor(QPalette.Text, QColor(220, 220, 220))
    p.setColor(QPalette.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Link, QColor(42, 130, 218))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(20, 20, 20))
    # Disabled state tweaks
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(150, 150, 150))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(150, 150, 150))
    return p


def make_light_palette() -> QPalette:
    """Return a slightly refined light palette for Fusion.

    The default Fusion light palette is already good; these are minor tweaks
    for tooltip contrast and link color.
    """
    p = QPalette()
    # Let Qt initialize defaults for the current platform/style by creating
    # a palette and only overriding a few roles for consistency.
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 235))
    p.setColor(QPalette.ToolTipText, QColor(20, 20, 20))
    p.setColor(QPalette.Link, QColor(30, 100, 220))
    return p


def apply_palette(app, theme: Theme) -> None:
    """Apply the requested palette to the QApplication."""
    if theme == Theme.DARK:
        app.setPalette(make_dark_palette())
    else:
        app.setPalette(make_light_palette())

