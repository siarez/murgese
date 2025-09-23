from __future__ import annotations
from PySide6 import QtWidgets
import pyqtgraph as pg


def mk_dspin(lo, hi, val, step, suffix, decimals) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(" " + suffix.strip())
    return s


def row_color(index: int):
    return pg.intColor(index, hues=12, values=1, maxValue=255)


def build_plot(name_cascade: str):
    pg.setConfigOptions(antialias=True)
    pw = pg.PlotWidget()
    pw.setBackground("default")
    pw.setLogMode(x=True, y=False)
    pw.showGrid(x=True, y=True, alpha=0.3)
    pw.setLabel("bottom", "Frequency", units="Hz")
    pw.setLabel("left", "Magnitude", units="dB")
    pw.addLegend(offset=(10, 10))
    curve = pw.plot([], [], pen=pg.mkPen(width=2), name=name_cascade)
    pw.setYRange(-24, 24, padding=0.05)
    return pw, curve


def q_to_hex_twos(v: float, frac_bits: int) -> str:
    scale = 1 << frac_bits
    x = int(round(v * scale))
    # Saturate to 32-bit signed range
    if x > 0x7FFFFFFF:
        x = 0x7FFFFFFF
    if x < -0x80000000:
        x = -0x80000000
    u = x & 0xFFFFFFFF
    return f"0x{u:08X}"
