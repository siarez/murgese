import os
from PySide6.QtWidgets import QApplication, QStyleFactory
from app.ui.main_window import MainWindow
from app.ui.palette import apply_palette, Theme
import sys

def main():
    app = QApplication(sys.argv)
    # Use a uniform, cross‑platform style across OSes
    app.setStyle(QStyleFactory.create("Fusion"))
    # Apply theme (default light). Allow override via env QT_THEME=dark|light
    theme_env = os.environ.get('QT_THEME', '').strip().lower()
    theme = Theme.DARK if theme_env == 'dark' else Theme.LIGHT
    apply_palette(app, theme)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
