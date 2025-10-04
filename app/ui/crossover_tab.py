from __future__ import annotations
from typing import List
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np

from app.eqcore import (
    FilterType, BiquadParams, design_biquad,
    default_freq_grid, sos_response_db,
    sos_response_complex, cascade_response_complex,
    design_first_order_lpf, design_first_order_hpf,
)
from .util import mk_dspin, row_color, build_plot, q_to_hex_twos


class CrossoverTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._xo_num_sections = 5

        # Root split: left coeffs, right controls+plot+tables
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left pane: coefficients readouts
        left = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)

        # Hex toggle
        self.chk_hex = QtWidgets.QCheckBox("Show 2's complement hex")
        self.chk_hex.toggled.connect(self._update_xo_plots)
        left_v.addWidget(self.chk_hex)

        gb_ca = QtWidgets.QGroupBox("Channel A Coeffs (a0=1)")
        ca_v = QtWidgets.QVBoxLayout(gb_ca)
        self.txt_coeffs_A = QtWidgets.QPlainTextEdit()
        self.txt_coeffs_A.setReadOnly(True)
        self.txt_coeffs_A.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.txt_coeffs_A.setFont(font)
        self.txt_coeffs_A.setToolTip(
            "TI PPC3/TAS register convention stores denominator as A1 = −a1, A2 = −a2.\n"
            "Hex values shown for A1/A2 follow this convention."
        )
        ca_v.addWidget(self.txt_coeffs_A)
        noteA = QtWidgets.QLabel("A1/A2 shown as −a1/−a2 (TI PPC3)")
        noteA.setStyleSheet("color: palette(mid); font-size: 11px;")
        noteA.setToolTip(
            "TI PPC3/TAS register convention stores denominator as A1 = −a1, A2 = −a2.\n"
            "Hex values shown for A1/A2 follow this convention."
        )
        ca_v.addWidget(noteA)
        left_v.addWidget(gb_ca)

        gb_cb = QtWidgets.QGroupBox("Channel B Coeffs (a0=1)")
        cb_v = QtWidgets.QVBoxLayout(gb_cb)
        self.txt_coeffs_B = QtWidgets.QPlainTextEdit()
        self.txt_coeffs_B.setReadOnly(True)
        self.txt_coeffs_B.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.txt_coeffs_B.setFont(font)
        cb_v.addWidget(self.txt_coeffs_B)
        self.txt_coeffs_B.setToolTip(
            "TI PPC3/TAS register convention stores denominator as A1 = −a1, A2 = −a2.\n"
            "Hex values shown for A1/A2 follow this convention."
        )
        noteB = QtWidgets.QLabel("A1/A2 shown as −a1/−a2 (TI PPC3)")
        noteB.setStyleSheet("color: palette(mid); font-size: 11px;")
        noteB.setToolTip(
            "TI PPC3/TAS register convention stores denominator as A1 = −a1, A2 = −a2.\n"
            "Hex values shown for A1/A2 follow this convention."
        )
        cb_v.addWidget(noteB)
        left_v.addWidget(gb_cb)

        root.addWidget(left, 0)

        # Right pane
        right = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(8)

        # Controls row (removed global ripple; ripple is per-BQ now)

        # Plot: cascades A, B, and electrical sum
        self.plot_xo, curve_sum = build_plot("Electrical Sum")
        self.curve_xo_sum = curve_sum
        self.curve_xo_A = self.plot_xo.plot([], [], pen=pg.mkPen((100, 150, 255), width=2), name="Cascade A")
        self.curve_xo_B = self.plot_xo.plot([], [], pen=pg.mkPen((255, 150, 100), width=2), name="Cascade B")
        right_v.addWidget(self.plot_xo, 1)

        # Tables row
        tables = QtWidgets.QHBoxLayout()
        tables.setSpacing(8)
        right_v.addLayout(tables)

        root.addWidget(right, 1)

        # Channel A group
        gb_a = QtWidgets.QGroupBox("Channel A")
        la = QtWidgets.QVBoxLayout(gb_a)
        # Channel A polarity + delay row
        row_a = QtWidgets.QWidget()
        row_a_h = QtWidgets.QHBoxLayout(row_a)
        row_a_h.setContentsMargins(0, 0, 0, 0)
        row_a_h.setSpacing(8)
        self.chk_xo_pol_A = QtWidgets.QCheckBox("Invert polarity")
        self.chk_xo_pol_A.toggled.connect(self._update_xo_plots)
        row_a_h.addWidget(self.chk_xo_pol_A)
        # Delay control: 0..16 samples
        lbl_a = QtWidgets.QLabel("Delay:")
        row_a_h.addWidget(lbl_a)
        self.spin_delay_A = QtWidgets.QSpinBox()
        self.spin_delay_A.setRange(0, 16)
        self.spin_delay_A.setValue(0)
        self.spin_delay_A.setSuffix(" samples")
        self.spin_delay_A.setToolTip("TAS3251 per-channel delay (0–16 samples)")
        self.spin_delay_A.valueChanged.connect(self._on_delay_changed)
        row_a_h.addWidget(self.spin_delay_A)
        self.lbl_delay_A = QtWidgets.QLabel("≈ 0.0 cm (0.0 in)")
        self.lbl_delay_A.setToolTip("Approximate distance sound travels for this delay at 343 m/s")
        row_a_h.addWidget(self.lbl_delay_A)
        row_a_h.addStretch(1)
        la.addWidget(row_a)

        # Channel A gain (dB)
        row_ag = QtWidgets.QWidget()
        row_ag_h = QtWidgets.QHBoxLayout(row_ag)
        row_ag_h.setContentsMargins(0, 0, 0, 0)
        row_ag_h.setSpacing(8)
        row_ag_h.addWidget(QtWidgets.QLabel("Gain:"))
        self.spin_gain_A = QtWidgets.QDoubleSpinBox()
        self.spin_gain_A.setRange(-20.0, 20.0)
        self.spin_gain_A.setDecimals(2)
        self.spin_gain_A.setSingleStep(0.50)
        self.spin_gain_A.setSuffix(" dB")
        self.spin_gain_A.setToolTip("Channel A gain applied before summation")
        self.spin_gain_A.valueChanged.connect(self._update_xo_plots)
        row_ag_h.addWidget(self.spin_gain_A)
        self.lbl_gain_A = QtWidgets.QLabel("≈ 1.00×")
        row_ag_h.addWidget(self.lbl_gain_A)
        self.spin_gain_A.valueChanged.connect(self._on_gain_changed)
        row_ag_h.addStretch(1)
        la.addWidget(row_ag)
        self.table_xo_A = QtWidgets.QTableWidget(self._xo_num_sections, 6)
        self.table_xo_A.setHorizontalHeaderLabels(
            ["#", "Mode", "Topology", "Ripple / dB", "Cut-off / Hz", "Color"])
        self.table_xo_A.verticalHeader().setVisible(False)
        self.table_xo_A.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_xo_A.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_xo_A.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_xo_A.horizontalHeader().setStretchLastSection(True)
        self.table_xo_A.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table_xo_A.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        self.table_xo_A.setColumnWidth(5, 36)
        # Smooth, slower scrolling
        self.table_xo_A.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_xo_A.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_xo_A.verticalScrollBar().setSingleStep(6)   # pixels per wheel step
        self.table_xo_A.horizontalScrollBar().setSingleStep(10)
        la.addWidget(self.table_xo_A)

        # Channel B group
        gb_b = QtWidgets.QGroupBox("Channel B")
        lb = QtWidgets.QVBoxLayout(gb_b)
        # Channel B polarity + delay row
        row_b = QtWidgets.QWidget()
        row_b_h = QtWidgets.QHBoxLayout(row_b)
        row_b_h.setContentsMargins(0, 0, 0, 0)
        row_b_h.setSpacing(8)
        self.chk_xo_pol_B = QtWidgets.QCheckBox("Invert polarity")
        self.chk_xo_pol_B.toggled.connect(self._update_xo_plots)
        row_b_h.addWidget(self.chk_xo_pol_B)
        lbl_b = QtWidgets.QLabel("Delay:")
        row_b_h.addWidget(lbl_b)
        self.spin_delay_B = QtWidgets.QSpinBox()
        self.spin_delay_B.setRange(0, 16)
        self.spin_delay_B.setValue(0)
        self.spin_delay_B.setSuffix(" samples")
        self.spin_delay_B.setToolTip("TAS3251 per-channel delay (0–16 samples)")
        self.spin_delay_B.valueChanged.connect(self._on_delay_changed)
        row_b_h.addWidget(self.spin_delay_B)
        self.lbl_delay_B = QtWidgets.QLabel("≈ 0.0 cm (0.0 in)")
        self.lbl_delay_B.setToolTip("Approximate distance sound travels for this delay at 343 m/s")
        row_b_h.addWidget(self.lbl_delay_B)
        row_b_h.addStretch(1)
        lb.addWidget(row_b)

        # Channel B gain (dB)
        row_bg = QtWidgets.QWidget()
        row_bg_h = QtWidgets.QHBoxLayout(row_bg)
        row_bg_h.setContentsMargins(0, 0, 0, 0)
        row_bg_h.setSpacing(8)
        row_bg_h.addWidget(QtWidgets.QLabel("Gain:"))
        self.spin_gain_B = QtWidgets.QDoubleSpinBox()
        self.spin_gain_B.setRange(-20.0, 20.0)
        self.spin_gain_B.setDecimals(2)
        self.spin_gain_B.setSingleStep(0.50)
        self.spin_gain_B.setSuffix(" dB")
        self.spin_gain_B.setToolTip("Channel B gain applied before summation")
        self.spin_gain_B.valueChanged.connect(self._update_xo_plots)
        row_bg_h.addWidget(self.spin_gain_B)
        self.lbl_gain_B = QtWidgets.QLabel("≈ 1.00×")
        row_bg_h.addWidget(self.lbl_gain_B)
        self.spin_gain_B.valueChanged.connect(self._on_gain_changed)
        row_bg_h.addStretch(1)
        lb.addWidget(row_bg)
        self.table_xo_B = QtWidgets.QTableWidget(self._xo_num_sections, 6)
        self.table_xo_B.setHorizontalHeaderLabels(
            ["#", "Mode", "Topology", "Ripple / dB", "Cut-off / Hz", "Color"])
        self.table_xo_B.verticalHeader().setVisible(False)
        self.table_xo_B.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_xo_B.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_xo_B.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_xo_B.horizontalHeader().setStretchLastSection(True)
        self.table_xo_B.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table_xo_B.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        self.table_xo_B.setColumnWidth(5, 36)
        # Smooth, slower scrolling
        self.table_xo_B.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_xo_B.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_xo_B.verticalScrollBar().setSingleStep(6)
        self.table_xo_B.horizontalScrollBar().setSingleStep(10)
        lb.addWidget(self.table_xo_B)

        tables.addWidget(gb_a, 1)
        tables.addWidget(gb_b, 1)

        # State
        self._fs = 48000.0
        self._freqs = default_freq_grid(self._fs, n=2048, fmin=10.0)
        self._xo_curves_A: List[pg.PlotDataItem] = []
        self._xo_curves_B: List[pg.PlotDataItem] = []
        # Apply initial view limits now that _fs is defined
        from .util import apply_viewbox_limits, add_reset_button
        apply_viewbox_limits(self.plot_xo, self._fs)
        add_reset_button(self.plot_xo, self._fs)

        # Populate rows
        self._init_xo_rows(self.table_xo_A, channel='A')
        self._init_xo_rows(self.table_xo_B, channel='B')
        self._update_xo_plots()
        self._update_delay_labels()
        self._update_gain_labels()

    def set_fs(self, fs: float):
        if abs(fs - getattr(self, '_fs', 48000.0)) > 1e-9:
            self._fs = float(fs)
            self._freqs = default_freq_grid(self._fs, n=2048, fmin=10.0)
            from .util import apply_viewbox_limits
            apply_viewbox_limits(self.plot_xo, self._fs)
            self._update_xo_plots()
            self._update_delay_labels()
            self._update_gain_labels()

    def _on_delay_changed(self, *_):
        # Update labels and plots (phase of electrical sum changes)
        self._update_delay_labels()
        self._update_xo_plots()

    def _on_gain_changed(self, *_):
        # Update linear gain labels and plots
        self._update_gain_labels()
        self._update_xo_plots()

    def _update_gain_labels(self):
        def fmt(db: float) -> str:
            lin = 10.0 ** (float(db) / 20.0)
            return f"≈ {lin:.2f}×"
        if hasattr(self, 'spin_gain_A') and hasattr(self, 'lbl_gain_A'):
            self.lbl_gain_A.setText(fmt(self.spin_gain_A.value()))
        if hasattr(self, 'spin_gain_B') and hasattr(self, 'lbl_gain_B'):
            self.lbl_gain_B.setText(fmt(self.spin_gain_B.value()))

    def _update_delay_labels(self):
        # Compute and show distance equivalents for current delays
        def fmt(samples: int) -> str:
            if self._fs <= 0:
                return "≈ n/a"
            t = float(samples) / float(self._fs)
            dist_m = 343.0 * t
            cm = dist_m * 100.0
            inch = dist_m * 39.37007874
            return f"≈ {cm:.1f} cm ({inch:.1f} in)"
        if hasattr(self, 'spin_delay_A'):
            self.lbl_delay_A.setText(fmt(int(self.spin_delay_A.value())))
        if hasattr(self, 'spin_delay_B'):
            self.lbl_delay_B.setText(fmt(int(self.spin_delay_B.value())))

    def _init_xo_rows(self, table: QtWidgets.QTableWidget, channel: str):
        for row in range(self._xo_num_sections):
            self._setup_xo_row(table, row, channel)

    def _setup_xo_row(self, table: QtWidgets.QTableWidget, row: int, channel: str):
        # Number
        num_item = QtWidgets.QTableWidgetItem(str(row + 1))
        num_item.setFlags(num_item.flags() & ~QtCore.Qt.ItemIsEditable)
        num_item.setTextAlignment(QtCore.Qt.AlignCenter)
        table.setItem(row, 0, num_item)

        # Mode
        cb_mode = QtWidgets.QComboBox()
        cb_mode.addItems(["All-pass", "Phase shift 1st", "Phase shift 2nd", "Low-pass", "High-pass"])  # AP, AP1, AP2, LPF/HPF
        cb_mode.currentIndexChanged.connect(lambda _=None, r=row, t=table: self._on_mode_changed(t, r))
        table.setCellWidget(row, 1, cb_mode)

        # Topology
        cb_topo = QtWidgets.QComboBox()
        cb_topo.addItems(["Butterworth 1st", "Butterworth 2nd", "Bessel", "Chebyshev I"]) 
        cb_topo.currentIndexChanged.connect(lambda _=None, t=table, r=row: (self._update_xo_row_enabled(t, r), self._update_xo_plots()))
        table.setCellWidget(row, 2, cb_topo)

        # Ripple (per BQ). Not used in Phase shift.
        spin_ripple = mk_dspin(0.01, 3.0, 0.50, 0.10, " dB", 2)
        spin_ripple.setToolTip("Chebyshev ripple (dB). Not used in Phase shift.")
        spin_ripple.valueChanged.connect(self._update_xo_plots)
        table.setCellWidget(row, 3, spin_ripple)

        # Cut-off / center frequency
        spin_fc = mk_dspin(5, 96_000, 2000.0, 10.0, " Hz", 2)
        spin_fc.valueChanged.connect(self._update_xo_plots)
        table.setCellWidget(row, 4, spin_fc)

        # Color swatch (fill entire cell background)
        color = row_color(row if channel == 'A' else row + 6)
        color_item = QtWidgets.QTableWidgetItem("")
        # Not editable and not selectable so selection highlight won't gray it out
        color_item.setFlags(color_item.flags() & ~QtCore.Qt.ItemIsEditable & ~QtCore.Qt.ItemIsSelectable)
        color_item.setBackground(color)
        color_item.setToolTip(color.name())
        table.setItem(row, 5, color_item)

        # Initialize enabled state
        self._update_xo_row_enabled(table, row)

    def _on_mode_changed(self, table: QtWidgets.QTableWidget, row: int):
        self._update_xo_row_enabled(table, row)
        self._update_xo_plots()

    def _xo_is_allpass_row(self, table: QtWidgets.QTableWidget, row: int) -> bool:
        cb_mode = table.cellWidget(row, 1)
        return (cb_mode is not None) and cb_mode.currentText().lower().startswith('all')

    def _update_xo_row_enabled(self, table: QtWidgets.QTableWidget, row: int):
        is_ap = self._xo_is_allpass_row(table, row)
        # Disable/enable fields depending on mode
        mode = table.cellWidget(row, 1).currentText().lower() if table.cellWidget(row, 1) else ""
        # Topology widget
        w_topo = table.cellWidget(row, 2)
        w_ripple = table.cellWidget(row, 3)
        w_fc = table.cellWidget(row, 4)
        if w_topo:
            # Topology disabled for All-pass and both Phase shift modes
            w_topo.setEnabled(not (is_ap or mode.startswith('phase shift')))
        if w_fc:
            # Frequency enabled for all except All-pass (unity)
            w_fc.setEnabled(not is_ap)
        if w_ripple:
            if mode.startswith('phase shift 2nd'):
                # Use Ripple control as Q for Phase shift 2nd
                w_ripple.setEnabled(True)
                w_ripple.setSuffix(" Q")
            else:
                # Default behavior: enabled only for Chebyshev I and non-All-pass
                cb_topo = table.cellWidget(row, 2)
                is_cheby = (cb_topo.currentText().lower().startswith('cheby')) if cb_topo else False
                w_ripple.setEnabled((not is_ap) and is_cheby)
                w_ripple.setSuffix(" dB")

    def _xo_q_for_topology(self, topo: str, ripple_db: float) -> float:
        t = topo.lower()
        if "bessel" in t:
            return 1/np.sqrt(3)  # ~0.5774
        if "cheby" in t:
            # 2nd-order Chebyshev Type I Q from ripple
            # epsilon = sqrt(10^(r/10)-1); alpha = (1/2) asinh(1/epsilon)
            # Q = sqrt(cosh(2*alpha)) / (2*sinh(alpha))
            eps = max(1e-9, np.sqrt(10.0**(ripple_db/10.0) - 1.0))
            alpha = 0.5 * np.arcsinh(1.0/eps)
            Q = np.sqrt(np.cosh(2.0*alpha)) / (2.0*np.sinh(alpha))
            return float(Q)
        # default to Butterworth 2nd order
        return 1/np.sqrt(2)

    def _xo_row_active(self, table: QtWidgets.QTableWidget, row: int) -> bool:
        w = table.cellWidget(row, 1)
        return bool(w.isChecked()) if w else False

    def _design_xo_biquad(self, table: QtWidgets.QTableWidget, row: int, invert_polarity: bool = False):
        mode = table.cellWidget(row, 1).currentText()
        topo = table.cellWidget(row, 2).currentText()
        ripple_db = float(table.cellWidget(row, 3).value())
        fc   = float(table.cellWidget(row, 4).value())
        if mode.lower().startswith('all'):
            sos = BiquadParams(typ=FilterType.ALLPASS, fs=self._fs, f0=fc, q=1.0, gain_db=0.0)
            sos = design_biquad(sos)
        elif mode.lower().startswith('phase shift 1st'):
            # 1st-order all-pass using center frequency only
            sos = design_biquad(BiquadParams(typ=FilterType.ALLPASS1, fs=self._fs, f0=fc, q=1.0, gain_db=0.0))
        elif mode.lower().startswith('phase shift 2nd'):
            # 2nd-order all-pass using Q from the Ripple/Q control
            qv = max(1e-3, float(table.cellWidget(row, 3).value()))
            sos = design_biquad(BiquadParams(typ=FilterType.ALLPASS2, fs=self._fs, f0=fc, q=qv, gain_db=0.0))
        else:
            typ = FilterType.LPF if mode.lower().startswith('low') else FilterType.HPF
            # Topology selection
            t = topo.lower()
            if "butterworth 1st" in t:
                sos = design_first_order_lpf(self._fs, fc) if typ == FilterType.LPF else design_first_order_hpf(self._fs, fc)
            else:
                q = float(self._xo_q_for_topology(topo, ripple_db))
                p = BiquadParams(typ=typ, fs=self._fs, f0=fc, q=q, gain_db=0.0)
                sos = design_biquad(p)
        if invert_polarity:
            # For display/export we reflect polarity by flipping numerator.
            sos.b0 = -sos.b0
            sos.b1 = -sos.b1
            sos.b2 = -sos.b2
        return sos

    def _update_xo_plots(self):
        # Clear previous per-biquad curves from the single XO plot
        for curves in (getattr(self, '_xo_curves_A', []), getattr(self, '_xo_curves_B', [])):
            for c in curves:
                try:
                    self.plot_xo.removeItem(c)
                except Exception:
                    pass
            curves.clear()
        self._xo_curves_A = []
        self._xo_curves_B = []

        # Channel A
        acc_db_A = np.zeros_like(self._freqs, dtype=float)
        for row in range(self.table_xo_A.rowCount()):
            self._update_xo_row_enabled(self.table_xo_A, row)
            invert_this = self.chk_xo_pol_A.isChecked() and (row == 0)
            sos = self._design_xo_biquad(self.table_xo_A, row, invert_polarity=invert_this)
            H_sec = sos_response_db(sos, self._freqs, self._fs)
            if not (self._xo_is_allpass_row(self.table_xo_A, row) or self.table_xo_A.cellWidget(row,1).currentText().lower().startswith('phase')):
                curve = self.plot_xo.plot(self._freqs, H_sec,
                                          pen=pg.mkPen(row_color(row), width=1, style=QtCore.Qt.DashLine),
                                          name=f"A{row+1}")
                self._xo_curves_A.append(curve)
            acc_db_A += H_sec
        # Apply channel A gain to cascade curve
        gA_db = float(getattr(self, 'spin_gain_A', None).value()) if hasattr(self, 'spin_gain_A') else 0.0
        self.curve_xo_A.setData(self._freqs, acc_db_A + gA_db)

        # Channel B
        acc_db_B = np.zeros_like(self._freqs, dtype=float)
        for row in range(self.table_xo_B.rowCount()):
            self._update_xo_row_enabled(self.table_xo_B, row)
            invert_this = self.chk_xo_pol_B.isChecked() and (row == 0)
            sos = self._design_xo_biquad(self.table_xo_B, row, invert_polarity=invert_this)
            H_sec = sos_response_db(sos, self._freqs, self._fs)
            color = row_color(row + 6)
            if not (self._xo_is_allpass_row(self.table_xo_B, row) or self.table_xo_B.cellWidget(row,1).currentText().lower().startswith('phase')):
                curve = self.plot_xo.plot(self._freqs, H_sec,
                                          pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                                          name=f"B{row+1}")
                self._xo_curves_B.append(curve)
            acc_db_B += H_sec
        gB_db = float(getattr(self, 'spin_gain_B', None).value()) if hasattr(self, 'spin_gain_B') else 0.0
        self.curve_xo_B.setData(self._freqs, acc_db_B + gB_db)

        # Electrical sum using complex responses (accounts for phase/polarity)
        sections_A = []
        for row in range(self.table_xo_A.rowCount()):
            invert_this = self.chk_xo_pol_A.isChecked() and (row == 0)
            sections_A.append(self._design_xo_biquad(self.table_xo_A, row, invert_polarity=invert_this))
        sections_B = []
        for row in range(self.table_xo_B.rowCount()):
            invert_this = self.chk_xo_pol_B.isChecked() and (row == 0)
            sections_B.append(self._design_xo_biquad(self.table_xo_B, row, invert_polarity=invert_this))

        H_A = cascade_response_complex(sections_A, self._freqs, self._fs) if sections_A else np.ones_like(self._freqs, dtype=complex)
        H_B = cascade_response_complex(sections_B, self._freqs, self._fs) if sections_B else np.ones_like(self._freqs, dtype=complex)
        # Apply per-channel sample delays (phase-only, magnitude unchanged)
        nA = int(getattr(self, 'spin_delay_A', None).value() if hasattr(self, 'spin_delay_A') else 0)
        nB = int(getattr(self, 'spin_delay_B', None).value() if hasattr(self, 'spin_delay_B') else 0)
        if (nA | nB) != 0:
            w = 2.0 * np.pi * self._freqs / self._fs
            if nA:
                H_A = H_A * np.exp(-1j * w * nA)
            if nB:
                H_B = H_B * np.exp(-1j * w * nB)
        # Apply linear gains for the electrical sum
        gA = 10.0 ** (gA_db/20.0)
        gB = 10.0 ** (gB_db/20.0)
        H_sum = H_A * gA + H_B * gB
        acc_db_sum = 20.0 * np.log10(np.maximum(np.abs(H_sum), 1e-12))
        self.curve_xo_sum.setData(self._freqs, acc_db_sum)

        # Update coefficients text areas
        def fmt_lines(table, invert_channel):
            lines = []
            for row in range(table.rowCount()):
                invert_this = invert_channel and (row == 0)
                sos = self._design_xo_biquad(table, row, invert_polarity=invert_this)
                if self.chk_hex.isChecked():
                    b0 = q_to_hex_twos(sos.b0, 31)
                    b1 = q_to_hex_twos(sos.b1, 30)
                    b2 = q_to_hex_twos(sos.b2, 31)
                    # TI PPC3/TAS convention stores denominator as -a1, -a2
                    a1 = q_to_hex_twos(-sos.a1, 30)
                    a2 = q_to_hex_twos(-sos.a2, 31)
                    lines.append(
                        f"{row+1:>2}: b0={b0}, b1={b1}, b2={b2}, a1={a1}, a2={a2}"
                    )
                else:
                    lines.append(
                        f"{row+1:>2}: b0={sos.b0:.4f}, b1={sos.b1:.4f}, b2={sos.b2:.4f}, a1={sos.a1:.4f}, a2={sos.a2:.4f}"
                    )
            return "\n".join(lines)

        self.txt_coeffs_A.setPlainText(fmt_lines(self.table_xo_A, self.chk_xo_pol_A.isChecked()))
        self.txt_coeffs_B.setPlainText(fmt_lines(self.table_xo_B, self.chk_xo_pol_B.isChecked()))
