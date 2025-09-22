from __future__ import annotations
from typing import List
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np


from app.eqcore import (
    FilterType, BiquadParams, design_biquad,
    default_freq_grid, sos_response_db
)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAS3251 EQ Designer (PySide6)")
        self.resize(1100, 700)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left controls
        left = self._build_left_controls()
        root.addWidget(left, 0)

        # Right: plot + table
        right = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(8)
        root.addWidget(right, 1)

        self.plot = self._build_plot()
        right_v.addWidget(self.plot, 1)

        table_box = self._build_table_box()
        right_v.addWidget(table_box, 0)

        # State
        self._fs = 48000.0
        self._freqs = default_freq_grid(self._fs, n=2048, fmin=10.0)
        self._section_curves: List[pg.PlotDataItem] = []

        # Seed one row
        self._add_row(FilterType.PEAK, 1000.0, 0.70, 4.5, active=True)
        self.table.selectRow(0)
        self._sync_editors_from_row(0)
        self._update_plot()

    # ---------- Left controls ----------
    def _build_left_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self.cb_type = QtWidgets.QComboBox()
        for t in FilterType:
            self.cb_type.addItem(t.value, t)
        self.cb_type.currentIndexChanged.connect(self._apply_editors_to_selected)

        self.spin_fs = QtWidgets.QDoubleSpinBox()
        self.spin_fs.setRange(8_000, 384_000)
        self.spin_fs.setValue(48_000)
        self.spin_fs.setSingleStep(1_000)
        self.spin_fs.setSuffix(" Hz")
        self.spin_fs.setDecimals(0)
        self.spin_fs.valueChanged.connect(self._on_fs_changed)

        self.spin_f0 = QtWidgets.QDoubleSpinBox()
        self.spin_f0.setRange(5, 96_000)
        self.spin_f0.setValue(1000.0)
        self.spin_f0.setSingleStep(10.0)
        self.spin_f0.setSuffix(" Hz")
        self.spin_f0.setDecimals(2)
        self.spin_f0.valueChanged.connect(self._apply_editors_to_selected)

        self.spin_q = QtWidgets.QDoubleSpinBox()
        self.spin_q.setRange(0.1, 50.0)
        self.spin_q.setValue(0.70)
        self.spin_q.setSingleStep(0.1)
        self.spin_q.setDecimals(3)
        self.spin_q.valueChanged.connect(self._apply_editors_to_selected)

        self.spin_gain = QtWidgets.QDoubleSpinBox()
        self.spin_gain.setRange(-24.0, 24.0)
        self.spin_gain.setValue(4.5)
        self.spin_gain.setSingleStep(0.5)
        self.spin_gain.setSuffix(" dB")
        self.spin_gain.setDecimals(2)
        self.spin_gain.valueChanged.connect(self._apply_editors_to_selected)

        btn_add = QtWidgets.QPushButton("Add Filter")
        btn_dup = QtWidgets.QPushButton("Duplicate")
        btn_del = QtWidgets.QPushButton("Remove Selected")
        btn_add.clicked.connect(lambda: self._add_row(
            self.cb_type.currentData(), float(self.spin_f0.value()),
            float(self.spin_q.value()), float(self.spin_gain.value()), active=True))
        btn_dup.clicked.connect(self._duplicate_selected)
        btn_del.clicked.connect(self._remove_selected)

        form.addRow("Filter type", self.cb_type)
        form.addRow("Sample rate", self.spin_fs)
        form.addRow("f₀ / f_c", self.spin_f0)
        form.addRow("Q", self.spin_q)
        form.addRow("Gain", self.spin_gain)
        form.addRow(btn_add)
        form.addRow(btn_dup)
        form.addRow(btn_del)
        form.addRow(QtWidgets.QLabel(" "))
        form.addRow(self._build_readout_group())
        return w

    def _build_readout_group(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Selected Section Coeffs (a0=1)")
        layout = QtWidgets.QFormLayout(gb)
        self.lbl_b0 = QtWidgets.QLabel("-")
        self.lbl_b1 = QtWidgets.QLabel("-")
        self.lbl_b2 = QtWidgets.QLabel("-")
        self.lbl_a1 = QtWidgets.QLabel("-")
        self.lbl_a2 = QtWidgets.QLabel("-")
        for k, lbl in (("b0", self.lbl_b0), ("b1", self.lbl_b1), ("b2", self.lbl_b2),
                       ("a1", self.lbl_a1), ("a2", self.lbl_a2)):
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            layout.addRow(k, lbl)
        return gb

    # ---------- Plot ----------
    def _build_plot(self) -> pg.PlotWidget:
        pg.setConfigOptions(antialias=True)
        pw = pg.PlotWidget()
        pw.setBackground("default")
        pw.setLogMode(x=True, y=False)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("bottom", "Frequency", units="Hz")
        pw.setLabel("left", "Magnitude", units="dB")
        pw.addLegend(offset=(10, 10))
        self.curve_cascade = pw.plot([], [], pen=pg.mkPen(width=2), name="Cascade")
        pw.setYRange(-24, 24, padding=0.05)
        return pw

    # ---------- Table ----------
    def _build_table_box(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Filters")
        v = QtWidgets.QVBoxLayout(gb)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Active", "Type", "f₀ / Hz", "Q", "Gain / dB", "Color"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        v.addWidget(self.table)
        return gb

    def _add_row(self, typ: FilterType, f0: float, q: float, gain: float, active: bool = True):
        row = self.table.rowCount()
        self.table.insertRow(row)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(active)
        chk.stateChanged.connect(self._update_plot)
        self.table.setCellWidget(row, 0, chk)

        cb = QtWidgets.QComboBox()
        for t in FilterType:
            cb.addItem(t.value, t)
        cb.setCurrentIndex(list(FilterType).index(typ))
        cb.currentIndexChanged.connect(lambda _=None, r=row: self._on_type_changed(r))
        self.table.setCellWidget(row, 1, cb)

        spin_f0 = self._mk_dspin(5, 96_000, f0, 10.0, " Hz", 2)
        spin_q  = self._mk_dspin(0.1, 50.0, q, 0.1, "", 3)
        spin_g  = self._mk_dspin(-24.0, 24.0, gain, 0.5, " dB", 2)
        for col, spin in [(2, spin_f0), (3, spin_q), (4, spin_g)]:
            spin.valueChanged.connect(self._update_plot)
            self.table.setCellWidget(row, col, spin)

        color = pg.intColor(row, hues=12, values=1, maxValue=255)
        swatch = QtWidgets.QLabel("   ")
        swatch.setAutoFillBackground(True)
        pal = swatch.palette()
        pal.setColor(QtGui.QPalette.Window, color)  # 'color' is already a QColor
        swatch.setPalette(pal)
        swatch.setAutoFillBackground(True)
        swatch.setPalette(pal)
        swatch.setFrameShape(QtWidgets.QFrame.Box)
        self.table.setCellWidget(row, 5, swatch)

        self._update_plot()

    def _duplicate_selected(self):
        r = self._selected_row()
        if r is None: return
        p = self._params_from_row(r)
        self._add_row(p.typ, p.f0, p.q, p.gain_db, active=True)
        self.table.selectRow(self.table.rowCount()-1)

    def _remove_selected(self):
        r = self._selected_row()
        if r is None: return
        self.table.removeRow(r)
        self._update_plot()

    def _selected_row(self) -> int | None:
        sel = self.table.selectionModel().selectedRows()
        return sel[0].row() if sel else None

    def _mk_dspin(self, lo, hi, val, step, suffix, decimals) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        if suffix: s.setSuffix(" " + suffix.strip())
        return s

    def _params_from_row(self, row: int) -> BiquadParams:
        typ = self.table.cellWidget(row, 1).currentData()
        f0  = float(self.table.cellWidget(row, 2).value())
        q   = float(self.table.cellWidget(row, 3).value())
        g   = float(self.table.cellWidget(row, 4).value())
        return BiquadParams(typ=typ, fs=self._fs, f0=f0, q=q, gain_db=g)

    def _row_active(self, row: int) -> bool:
        return bool(self.table.cellWidget(row, 0).isChecked())

    def _row_color(self, row: int):
        return pg.intColor(row, hues=12, values=1, maxValue=255)

    # ---------- Sync editors <-> row ----------
    def _on_row_selected(self):
        r = self._selected_row()
        if r is None: return
        self._sync_editors_from_row(r)

    def _on_type_changed(self, row: int):
        if self._selected_row() == row:
            cb = self.table.cellWidget(row, 1)
            self.cb_type.setCurrentIndex(cb.currentIndex())
        self._update_plot()

    def _sync_editors_from_row(self, row: int):
        cb = self.table.cellWidget(row, 1)
        self.cb_type.setCurrentIndex(cb.currentIndex())

        self.spin_f0.blockSignals(True); self.spin_q.blockSignals(True); self.spin_gain.blockSignals(True)
        self.spin_f0.setValue(self.table.cellWidget(row, 2).value())
        self.spin_q.setValue(self.table.cellWidget(row, 3).value())
        self.spin_gain.setValue(self.table.cellWidget(row, 4).value())
        self.spin_f0.blockSignals(False); self.spin_q.blockSignals(False); self.spin_gain.blockSignals(False)

        self._update_coeff_readout(row)

    def _apply_editors_to_selected(self, *_):
        r = self._selected_row()
        if r is None: return

        cb_row = self.table.cellWidget(r, 1)
        cb_row.blockSignals(True)
        cb_row.setCurrentIndex(self.cb_type.currentIndex())
        cb_row.blockSignals(False)

        f0_box = self.table.cellWidget(r, 2)
        q_box  = self.table.cellWidget(r, 3)
        g_box  = self.table.cellWidget(r, 4)
        f0_box.blockSignals(True); f0_box.setValue(self.spin_f0.value()); f0_box.blockSignals(False)
        q_box.blockSignals(True);  q_box.setValue(self.spin_q.value());  q_box.blockSignals(False)
        g_box.blockSignals(True);  g_box.setValue(self.spin_gain.value()); g_box.blockSignals(False)

        self._update_coeff_readout(r)
        self._update_plot()

    def _update_coeff_readout(self, row: int):
        p = self._params_from_row(row)
        sos = design_biquad(p)
        self.lbl_b0.setText(f"{sos.b0:.9f}")
        self.lbl_b1.setText(f"{sos.b1:.9f}")
        self.lbl_b2.setText(f"{sos.b2:.9f}")
        self.lbl_a1.setText(f"{sos.a1:.9f}")
        self.lbl_a2.setText(f"{sos.a2:.9f}")

    def _on_fs_changed(self):
        fs_new = float(self.spin_fs.value())
        if abs(fs_new - self._fs) > 1e-9:
            self._fs = fs_new
            self._freqs = default_freq_grid(self._fs, n=2048, fmin=10.0)
            self._update_plot()

    # ---------- Plot update ----------
    def _update_plot(self):
        # clear per-section curves
        for c in self._section_curves:
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self._section_curves.clear()

        acc_db = np.zeros_like(self._freqs, dtype=float)
        for row in range(self.table.rowCount()):
            if not self._row_active(row):
                continue
            p = self._params_from_row(row)
            sos = design_biquad(p)
            H_sec = sos_response_db(sos, self._freqs, self._fs)
            curve = self.plot.plot(self._freqs, H_sec,
                                   pen=pg.mkPen(self._row_color(row), width=1),
                                   name=f"Section {row+1}")
            self._section_curves.append(curve)
            acc_db += H_sec

        self.curve_cascade.setData(self._freqs, acc_db)
