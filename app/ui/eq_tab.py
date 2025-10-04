from __future__ import annotations
from typing import List
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np

from app.eqcore import (
    FilterType, BiquadParams, design_biquad,
    default_freq_grid, sos_response_db
)
from .util import mk_dspin, row_color, build_plot, q_to_hex_twos


class EqTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        # Show 14 user-configurable biquads; BQ15 is reserved for Stage Gain (computed only)
        self._num_sections = 14

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: coefficients readout
        left = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left)
        vleft.setContentsMargins(0, 0, 0, 0)
        vleft.setSpacing(0)
        vleft.addWidget(self._build_readout_group())
        root.addWidget(left, 0)

        # Right: plot + table
        right = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(8)
        root.addWidget(right, 1)

        self.plot, self.curve_cascade = build_plot("Cascade")
        right_v.addWidget(self.plot, 1)
        box = self._build_table_box()
        right_v.addWidget(box, 0)

        # State
        self._fs = 48000.0
        self._freqs = default_freq_grid(self._fs, n=2048, fmin=10.0)
        self._section_curves: List[pg.PlotDataItem] = []
        # Apply initial view limits now that _fs is defined
        from .util import apply_viewbox_limits, add_reset_button
        apply_viewbox_limits(self.plot, self._fs)
        add_reset_button(self.plot, self._fs)

        self._init_fixed_rows()
        self.table.selectRow(0)
        self._update_plot()

    def set_fs(self, fs: float):
        if abs(fs - getattr(self, '_fs', 48000.0)) > 1e-9:
            self._fs = float(fs)
            self._freqs = default_freq_grid(self._fs, n=2048, fmin=10.0)
            from .util import apply_viewbox_limits
            apply_viewbox_limits(self.plot, self._fs)
            self._update_plot()

    def _build_readout_group(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Selected Filters Coeffs (a0=1)")
        layout = QtWidgets.QVBoxLayout(gb)
        self.chk_designed = QtWidgets.QCheckBox("Show designed (pre-normalization)")
        self.chk_designed.toggled.connect(self._refresh_selected_coeffs)
        layout.addWidget(self.chk_designed)
        self.chk_hex = QtWidgets.QCheckBox("Show 2's complement hex")
        self.chk_hex.toggled.connect(self._refresh_selected_coeffs)
        layout.addWidget(self.chk_hex)
        self.txt_coeffs = QtWidgets.QPlainTextEdit()
        self.txt_coeffs.setReadOnly(True)
        self.txt_coeffs.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.txt_coeffs.setFont(font)
        # Clarify TI PPC3 denominator sign convention
        self.txt_coeffs.setToolTip(
            "TI PPC3/TAS register convention stores denominator as A1 = −a1, A2 = −a2.\n"
            "Hex values shown for A1/A2 follow this convention."
        )
        layout.addWidget(self.txt_coeffs)
        note = QtWidgets.QLabel("A1/A2 shown as −a1/−a2 (TI PPC3)")
        note.setStyleSheet("color: palette(mid); font-size: 11px;")
        note.setToolTip(
            "TI PPC3/TAS register convention stores denominator as A1 = −a1, A2 = −a2.\n"
            "Hex values shown for A1/A2 follow this convention."
        )
        layout.addWidget(note)
        # Saturation warning for Stage Gain (BQ15)
        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setStyleSheet("color: #b00020; font-weight: bold;")
        self.lbl_warn.setVisible(False)
        layout.addWidget(self.lbl_warn)
        return gb

    def _build_table_box(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Filters")
        v = QtWidgets.QVBoxLayout(gb)
        self.table = QtWidgets.QTableWidget(self._num_sections, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Type", "f₀ / Hz", "Q", "Gain / dB", "Color"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        # Make color column compact
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        v.addWidget(self.table)
        # After the widget is in place, fix the color column width
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(5, 36)
        # Smooth, slower scrolling for consistency
        self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table.verticalScrollBar().setSingleStep(6)
        self.table.horizontalScrollBar().setSingleStep(10)
        return gb

    def _init_fixed_rows(self):
        defaults = [
            (FilterType.PEAK, 1000.0, 0.70, 4.5, True)
        ]
        while len(defaults) < self._num_sections:
            defaults.append((FilterType.PEAK, 1000.0, 0.70, 0.0, True))
        for row, (typ, f0, q, gain, active) in enumerate(defaults):
            self._setup_row(row, typ, f0, q, gain, active)

    def _setup_row(self, row: int, typ: FilterType, f0: float, q: float,
                   gain: float, active: bool):
        # Number column
        num_item = QtWidgets.QTableWidgetItem(str(row + 1))
        num_item.setFlags(num_item.flags() & ~QtCore.Qt.ItemIsEditable)
        num_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.table.setItem(row, 0, num_item)

        # Type
        cb = QtWidgets.QComboBox()
        for t in FilterType:
            cb.addItem(t.value, t)
        cb.setCurrentIndex(list(FilterType).index(typ))
        cb.currentIndexChanged.connect(lambda _=None, r=row: self._on_type_changed(r))
        self.table.setCellWidget(row, 1, cb)

        # Params
        spin_f0 = mk_dspin(5, 96_000, f0, 10.0, " Hz", 2)
        spin_q  = mk_dspin(0.1, 50.0, q, 0.1, "", 3)
        spin_q.setToolTip(
            "RBJ Q for peaking/filters.\n"
            "Note: In TI PPC3 peaking, the displayed Q ≈ 2× RBJ Q\n"
            "(e.g., PPC3 Q=1.4 ≈ RBJ Q=0.7)."
        )
        spin_g  = mk_dspin(-24.0, 24.0, gain, 0.5, " dB", 2)
        for col, spin in [(2, spin_f0), (3, spin_q), (4, spin_g)]:
            spin.valueChanged.connect(self._update_plot)
            spin.valueChanged.connect(self._refresh_selected_coeffs)
            self.table.setCellWidget(row, col, spin)

        # Color swatch (fill entire cell background)
        color = row_color(row)
        color_item = QtWidgets.QTableWidgetItem("")
        # Not editable and not selectable so selection highlight won't gray it out
        color_item.setFlags(color_item.flags() & ~QtCore.Qt.ItemIsEditable & ~QtCore.Qt.ItemIsSelectable)
        color_item.setBackground(color)
        color_item.setToolTip(color.name())
        self.table.setItem(row, 5, color_item)

        # Initialize enabled state
        self._update_row_enabled(row)

    def _params_from_row(self, row: int) -> BiquadParams:
        typ = self.table.cellWidget(row, 1).currentData()
        f0  = float(self.table.cellWidget(row, 2).value())
        q   = float(self.table.cellWidget(row, 3).value())
        g   = float(self.table.cellWidget(row, 4).value())
        return BiquadParams(typ=typ, fs=self._fs, f0=f0, q=q, gain_db=g)

    def _row_active(self, row: int) -> bool:
        return True

    def _is_allpass_row(self, row: int) -> bool:
        cb = self.table.cellWidget(row, 1)
        t = cb.currentData() if cb else None
        return t == FilterType.ALLPASS

    def _update_row_enabled(self, row: int):
        is_ap = self._is_allpass_row(row)
        cb = self.table.cellWidget(row, 1)
        typ = cb.currentData() if cb else None
        uses_gain = typ in (FilterType.PEAK, FilterType.LSHELF, FilterType.HSHELF)

        # f0 and Q disabled only for All-pass
        for col in (2, 3):
            w = self.table.cellWidget(row, col)
            if w is not None:
                w.setEnabled(not is_ap)
        # Gain disabled for All-pass and for types without gain (LPF/HPF/BPF/Notch)
        w_gain = self.table.cellWidget(row, 4)
        if w_gain is not None:
            w_gain.setEnabled((not is_ap) and uses_gain)

    def _on_type_changed(self, row: int):
        self._update_row_enabled(row)
        self._update_plot()

    def _refresh_selected_coeffs(self):
        lines = []
        hex_mode = getattr(self, 'chk_hex', None) is not None and self.chk_hex.isChecked()
        designed_mode = getattr(self, 'chk_designed', None) is not None and self.chk_designed.isChecked()

        mapped = []
        n = self.table.rowCount()
        if designed_mode:
            # Show raw designed coefficients for rows 1..14, and Stage Gain (BQ15)=1.0
            for row in range(n):
                p = self._params_from_row(row)
                sos = design_biquad(p)
                mapped.append((sos.b0, sos.b1, sos.b2, sos.a1, sos.a2))
            mapped.append((1.0, 0.0, 0.0, 0.0, 0.0))
        else:
            # Hardware-mapped per datasheet 7.5.2.4 + Table 7-25
            G = 1.0
            for row in range(n):
                p = self._params_from_row(row)
                sos = design_biquad(p)
                Si = max(abs(sos.b0), abs(sos.b1) / 2.0, abs(sos.b2))
                b0 = sos.b0 / Si
                b1 = sos.b1 / Si
                b2 = sos.b2 / Si
                a1 = sos.a1
                a2 = sos.a2
                G *= Si
                mapped.append((b0, b1, b2, a1, a2))
            mapped.append((G, 0.0, 0.0, 0.0, 0.0))

        for row, (b0, b1, b2, a1, a2) in enumerate(mapped):
            if hex_mode:
                if row == n:  # Stage Gain row (BQ15)
                    # Stage gain uses widened formats per TI flow (b0/b2: 5.27, b1: 6.26)
                    hb0 = q_to_hex_twos(b0, 27)
                    hb1 = q_to_hex_twos(b1, 26)
                    hb2 = q_to_hex_twos(b2, 27)
                else:
                    hb0 = q_to_hex_twos(b0, 31)
                    hb1 = q_to_hex_twos(b1, 30)
                    hb2 = q_to_hex_twos(b2, 31)
                # TI PPC3/TAS convention stores denominator as -a1, -a2
                # Use standard rounding quantization
                ha1 = q_to_hex_twos(-a1, 30)
                ha2 = q_to_hex_twos(-a2, 31)
                # Label stage gain explicitly for clarity
                if row == n:
                    lines.append(f"Stage Gain (BQ15): b0={hb0}, b1={hb1}, b2={hb2}, a1={ha1}, a2={ha2}")
                else:
                    lines.append(f"{row+1:>2}: b0={hb0}, b1={hb1}, b2={hb2}, a1={ha1}, a2={ha2}")
            else:
                if row == n:
                    lines.append(f"Stage Gain (BQ15): b0={b0:.4f}, b1={b1:.4f}, b2={b2:.4f}, a1={a1:.4f}, a2={a2:.4f}")
                else:
                    lines.append(f"{row+1:>2}: b0={b0:.4f}, b1={b1:.4f}, b2={b2:.4f}, a1={a1:.4f}, a2={a2:.4f}")
        self.txt_coeffs.setPlainText("\n".join(lines) if lines else "")

        # Update saturation warning for Stage Gain (hardware-mapped only)
        if not designed_mode:
            stage_b0 = mapped[-1][0]
            scaled = int(round(stage_b0 * (1 << 27)))
            would_sat = scaled > 0x7FFFFFFF or scaled < -0x80000000 or stage_b0 >= 16.0 or stage_b0 <= -16.0
            if would_sat:
                self.lbl_warn.setText("Warning: Stage Gain (BQ15) saturates Q5.27. Reduce upstream gains.")
                self.lbl_warn.setVisible(True)
            else:
                self.lbl_warn.setVisible(False)
        else:
            self.lbl_warn.setVisible(False)

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
            p = self._params_from_row(row)
            sos = design_biquad(p)
            H_sec = sos_response_db(sos, self._freqs, self._fs)
            if not self._is_allpass_row(row):
                curve = self.plot.plot(self._freqs, H_sec,
                                       pen=pg.mkPen(row_color(row), width=1),
                                       name=f"Section {row+1}")
                self._section_curves.append(curve)
            acc_db += H_sec

        self.curve_cascade.setData(self._freqs, acc_db)
        self._refresh_selected_coeffs()
