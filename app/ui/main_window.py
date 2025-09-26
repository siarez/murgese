from __future__ import annotations
from PySide6 import QtWidgets

from .eq_tab import EqTab
from .crossover_tab import CrossoverTab
from .exporter import export_pf5_from_ui
from .input_mixer_tab import InputMixerTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAS3251 EQ Designer")
        self.resize(1100, 700)

        tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(tabs)

        # Add tabs in process-flow order: Input Mixer first, then EQ, then Crossover
        self.mixer_tab = InputMixerTab(self)
        tabs.addTab(self.mixer_tab, "Input Mixer")

        self.eq_tab = EqTab(self)
        tabs.addTab(self.eq_tab, "EQ")

        self.xo_tab = CrossoverTab(self)
        tabs.addTab(self.xo_tab, "Crossover")

        # Toolbar with Export action
        tb = self.addToolBar('Main')
        act_export = tb.addAction('Export Config')
        act_export.triggered.connect(self._on_export)

    def _on_export(self):
        out = export_pf5_from_ui(self, self.xo_tab)
        if out:
            QtWidgets.QMessageBox.information(self, 'Export', f'Wrote {out}')
