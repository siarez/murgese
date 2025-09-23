from __future__ import annotations
from PySide6 import QtWidgets

from .eq_tab import EqTab
from .crossover_tab import CrossoverTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAS3251 EQ Designer")
        self.resize(1100, 700)

        tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(tabs)

        self.eq_tab = EqTab(self)
        tabs.addTab(self.eq_tab, "EQ")

        self.xo_tab = CrossoverTab(self)
        tabs.addTab(self.xo_tab, "Crossover")
