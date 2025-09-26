from __future__ import annotations
from PySide6 import QtWidgets, QtCore, QtGui
import math

from .util import mk_dspin, q_to_hex_twos


class InputMixerTab(QtWidgets.QWidget):
    """UI for INPUT MIXER coefficients (Left/Right mix) in Q9.23.

    Registers:
      - LefttoLeft   (L_out from L_in)
      - RighttoLeft  (L_out from R_in)
      - LefttoRight  (R_out from L_in)
      - RighttoRight (R_out from R_in)
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        # Match other tabs: left readout, right controls
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: hex readout
        left = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left)
        vleft.setContentsMargins(0, 0, 0, 0)
        vleft.setSpacing(6)

        gb = QtWidgets.QGroupBox("INPUT MIXER Coeffs (Q9.23 two's complement)")
        v = QtWidgets.QVBoxLayout(gb)
        self.txt_hex = QtWidgets.QPlainTextEdit()
        self.txt_hex.setReadOnly(True)
        self.txt_hex.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.txt_hex.setFont(font)
        v.addWidget(self.txt_hex)
        vleft.addWidget(gb)

        root.addWidget(left, 0)

        # Right: controls
        right = QtWidgets.QWidget()
        vright = QtWidgets.QVBoxLayout(right)
        vright.setContentsMargins(0, 0, 0, 0)
        vright.setSpacing(8)

        # Brief help text (rich text so we can control line height)
        help_html = (
            "<div style='line-height: 140%;'>"
            "Set the 2×2 mix matrix in signed Q9.23 (≈ −256 to +255.999999).<br>"
            "Identity mix: L‑to‑L = 1.0, R‑to‑R = 1.0, others = 0.0.<br>"
            "L_out = L_in × L‑to‑L + R_in × R‑to‑L<br>"
            "R_out = L_in × L‑to‑R + R_in × R‑to‑R"
            "</div>"
        )
        help_lbl = QtWidgets.QLabel(help_html)
        help_lbl.setTextFormat(QtCore.Qt.RichText)
        help_lbl.setWordWrap(True)
        # Bump font slightly for readability
        f = help_lbl.font()
        if f.pointSize() > 0:
            f.setPointSize(f.pointSize() + 2)
        else:
            f.setPixelSize(14)
        help_lbl.setFont(f)
        vright.addWidget(help_lbl)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        vright.addLayout(grid)

        # Spinboxes (Q9.23), range ~[-256, 256)
        def spin(default: float):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-256.0, 255.999999)
            s.setDecimals(6)
            s.setSingleStep(0.01)
            s.setValue(default)
            s.valueChanged.connect(self._refresh_hex)
            return s

        row = 0
        grid.addWidget(QtWidgets.QLabel("L_out = L_in ×"), row, 0)
        self.spin_LtoL = spin(1.0)
        grid.addWidget(self.spin_LtoL, row, 1)
        grid.addWidget(QtWidgets.QLabel("+ R_in ×"), row, 2)
        self.spin_RtoL = spin(0.0)
        grid.addWidget(self.spin_RtoL, row, 3)

        row += 1
        grid.addWidget(QtWidgets.QLabel("R_out = L_in ×"), row, 0)
        self.spin_LtoR = spin(0.0)
        grid.addWidget(self.spin_LtoR, row, 1)
        grid.addWidget(QtWidgets.QLabel("+ R_in ×"), row, 2)
        self.spin_RtoR = spin(1.0)
        grid.addWidget(self.spin_RtoR, row, 3)

        # dB matrix below controls
        db_group = QtWidgets.QGroupBox("Effective Gains (dB)")
        db_grid = QtWidgets.QGridLayout(db_group)
        db_grid.setHorizontalSpacing(12)
        db_grid.setVerticalSpacing(6)
        # headers
        db_grid.addWidget(QtWidgets.QLabel(""), 0, 0)
        db_grid.addWidget(QtWidgets.QLabel("L_in"), 0, 1)
        db_grid.addWidget(QtWidgets.QLabel("R_in"), 0, 2)
        # row labels
        db_grid.addWidget(QtWidgets.QLabel("L_out"), 1, 0)
        db_grid.addWidget(QtWidgets.QLabel("R_out"), 2, 0)
        # value labels
        self.lbl_db_LtoL = QtWidgets.QLabel("")
        self.lbl_db_RtoL = QtWidgets.QLabel("")
        self.lbl_db_LtoR = QtWidgets.QLabel("")
        self.lbl_db_RtoR = QtWidgets.QLabel("")
        db_grid.addWidget(self.lbl_db_LtoL, 1, 1)
        db_grid.addWidget(self.lbl_db_RtoL, 1, 2)
        db_grid.addWidget(self.lbl_db_LtoR, 2, 1)
        db_grid.addWidget(self.lbl_db_RtoR, 2, 2)
        vright.addWidget(db_group)

        # Spacer to push grid up
        vright.addStretch(1)
        root.addWidget(right, 1)

        self._refresh_hex()

    def _q923_hex(self, v: float) -> str:
        # 9.23 signed => 23 fractional bits
        return q_to_hex_twos(float(v), 23)

    def _refresh_hex(self):
        lines = []
        lines.append(f"LefttoLeft   = {self.spin_LtoL.value():.6f}  ->  {self._q923_hex(self.spin_LtoL.value())}")
        lines.append(f"RighttoLeft  = {self.spin_RtoL.value():.6f}  ->  {self._q923_hex(self.spin_RtoL.value())}")
        lines.append(f"LefttoRight  = {self.spin_LtoR.value():.6f}  ->  {self._q923_hex(self.spin_LtoR.value())}")
        lines.append(f"RighttoRight = {self.spin_RtoR.value():.6f}  ->  {self._q923_hex(self.spin_RtoR.value())}")
        self.txt_hex.setPlainText("\n".join(lines))

        # Update dB labels (20*log10(|gain|); show −∞ for 0, mark inversion on negative gains)
        def fmt_db(v: float) -> str:
            if v == 0.0:
                return "−∞ dB"
            db = 20.0 * math.log10(abs(v))
            inv = " (inv)" if v < 0.0 else ""
            return f"{db:.2f} dB{inv}"

        self.lbl_db_LtoL.setText(fmt_db(self.spin_LtoL.value()))
        self.lbl_db_RtoL.setText(fmt_db(self.spin_RtoL.value()))
        self.lbl_db_LtoR.setText(fmt_db(self.spin_LtoR.value()))
        self.lbl_db_RtoR.setText(fmt_db(self.spin_RtoR.value()))

    def values_hex(self) -> dict[str, str]:
        """Return mapping of INPUT MIXER names -> hex string in Q9.23."""
        return {
            'LefttoLeft': self._q923_hex(self.spin_LtoL.value()),
            'RighttoLeft': self._q923_hex(self.spin_RtoL.value()),
            'LefttoRight': self._q923_hex(self.spin_LtoR.value()),
            'RighttoRight': self._q923_hex(self.spin_RtoR.value()),
        }
