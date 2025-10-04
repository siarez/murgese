from __future__ import annotations
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from app.eqcore import SOS, default_freq_grid, sos_response_db, design_biquad, BiquadParams, FilterType
from .util import build_plot, hex_twos_to_float


@dataclass
class DetectResult:
    kind: str
    f0: Optional[float]
    q: Optional[float]
    gain_db: Optional[float]
    note: str = ""


class CoefCheckTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._fs = 48000.0
        self._eff_fs = self._fs
        self._freqs = default_freq_grid(self._eff_fs, n=2048, fmin=10.0)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Inputs group (two sets side-by-side)
        gb_in = QtWidgets.QGroupBox("Enter SOS Coefficients (two's complement hex)")
        grid = QtWidgets.QGridLayout(gb_in)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        labels = ['b0 (Q1.31)', 'b1 (Q2.30)', 'b2 (Q1.31)', 'a1 (Q2.30)', 'a2 (Q1.31)']
        # Headers
        grid.addWidget(QtWidgets.QLabel("Set A"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Set B"), 0, 3)
        self.edits_A = []
        self.edits_B = []
        for i, lab in enumerate(labels):
            row = i + 1
            grid.addWidget(QtWidgets.QLabel(lab+":"), row, 0)
            eA = QtWidgets.QLineEdit(); eA.setPlaceholderText('0x00000000'); eA.setMinimumWidth(170)
            eB = QtWidgets.QLineEdit(); eB.setPlaceholderText('0x00000000'); eB.setMinimumWidth(170)
            self.edits_A.append(eA); self.edits_B.append(eB)
            grid.addWidget(eA, row, 1)
            grid.addWidget(eB, row, 3)
        # Stage B0 (Q5.27) row
        row = len(labels) + 1
        grid.addWidget(QtWidgets.QLabel("stage B0 (Q5.27):"), row, 0)
        self.edit_stage_A = QtWidgets.QLineEdit(); self.edit_stage_A.setPlaceholderText('0x00000000'); self.edit_stage_A.setMinimumWidth(170)
        self.edit_stage_B = QtWidgets.QLineEdit(); self.edit_stage_B.setPlaceholderText('0x00000000'); self.edit_stage_B.setMinimumWidth(170)
        grid.addWidget(self.edit_stage_A, row, 1)
        grid.addWidget(self.edit_stage_B, row, 3)
        # Full JSON paste row
        row += 1
        grid.addWidget(QtWidgets.QLabel("full JSON (Enter to apply):"), row, 0)
        self.edit_json_A = QtWidgets.QLineEdit(); self.edit_json_A.setPlaceholderText("{'B0':'0x..','B1':'0x..','B2':'0x..','A1':'0x..','A2':'0x..','ST':'0x..'}"); self.edit_json_A.setMinimumWidth(170)
        self.edit_json_B = QtWidgets.QLineEdit(); self.edit_json_B.setPlaceholderText(self.edit_json_A.placeholderText()); self.edit_json_B.setMinimumWidth(170)
        self.edit_json_A.returnPressed.connect(lambda: self._apply_json('A'))
        self.edit_json_B.returnPressed.connect(lambda: self._apply_json('B'))
        grid.addWidget(self.edit_json_A, row, 1)
        grid.addWidget(self.edit_json_B, row, 3)
        # Options column
        opts = QtWidgets.QVBoxLayout()
        self.combo_sign = QtWidgets.QComboBox()
        self.combo_sign.addItems(["Standard (a1,a2)", "TI PPC3 (-a1,-a2)"])
        self.combo_sign.setCurrentIndex(1)
        self.combo_sign.setToolTip(
            "Sign convention for denominator A1/A2.\n"
            "Note: For peaking EQ, TI PPC3's Q ≈ 2× RBJ Q\n"
            "(e.g., PPC3 Q=1.4 ≈ RBJ Q=0.7)."
        )
        opts.addWidget(QtWidgets.QLabel("Sign Convention:"))
        opts.addWidget(self.combo_sign)
        self.combo_rate = QtWidgets.QComboBox()
        self.combo_rate.addItems(["fs (48 kHz)", "fs/2 (24 kHz)", "2×fs (96 kHz)"])
        opts.addWidget(QtWidgets.QLabel("Biquad Rate:"))
        opts.addWidget(self.combo_rate)
        # Toggle for best-fit overlay for Set A
        self.chk_fit = QtWidgets.QCheckBox("Show Best Fit A")
        self.chk_fit.setChecked(False)
        self.chk_fit.toggled.connect(self._on_analyze)
        opts.addWidget(self.chk_fit)
        # PPC3 Q note
        note_q = QtWidgets.QLabel("Note: PPC3 peaking Q ≈ 2× RBJ Q")
        note_q.setStyleSheet("color: palette(mid); font-size: 11px;")
        note_q.setToolTip("For peaking, PPC3's Q is about twice the RBJ Q definition.")
        opts.addWidget(note_q)
        opts.addStretch(1)
        w_opts = QtWidgets.QWidget()
        w_opts.setLayout(opts)
        grid.addWidget(w_opts, 0, 4, row+1, 1)
        # Analyze button across row
        self.btn = QtWidgets.QPushButton("Analyze")
        self.btn.clicked.connect(self._on_analyze)
        grid.addWidget(self.btn, row+1, 0, 1, 5)
        root.addWidget(gb_in)

        # Results + plot
        h = QtWidgets.QHBoxLayout()
        h.setSpacing(8)
        root.addLayout(h)

        # Results panel
        gb_res = QtWidgets.QGroupBox("Detection Result")
        vres = QtWidgets.QVBoxLayout(gb_res)
        self.lbl_kind = QtWidgets.QLabel("—")
        self.lbl_f0 = QtWidgets.QLabel("f0: —")
        self.lbl_q = QtWidgets.QLabel("Q: —")
        self.lbl_gain = QtWidgets.QLabel("Gain: —")
        self.lbl_note = QtWidgets.QLabel("")
        self.lbl_note.setWordWrap(True)
        for w in (self.lbl_kind, self.lbl_f0, self.lbl_q, self.lbl_gain, self.lbl_note):
            vres.addWidget(w)
        vres.addStretch(1)
        h.addWidget(gb_res, 0)

        # Plot panel
        # Restore shared plotting behavior (custom axis, grid, etc.)
        self.plot, self.curve_sos = build_plot("Measured A")
        from .util import apply_viewbox_limits, add_reset_button
        apply_viewbox_limits(self.plot, self._fs)
        add_reset_button(self.plot, self._fs)
        # Use cosmetic pens via pg.mkPen to avoid transform-scaled thickness on log-x
        self.curve_sos_B = self.plot.plot([], [], pen=pg.mkPen('#ff4081', width=2), name="Measured B")
        self.curve_fit = self.plot.plot([], [], pen=pg.mkPen('#00bcd4', width=2), name="Best Fit A")
        # Track our own data items and prune any extras (safety against accidental duplicates)
        self._plot_items_allowed = {self.curve_sos, self.curve_sos_B, self.curve_fit}
        h.addWidget(self.plot, 1)

    def set_fs(self, fs: float):
        if abs(fs - getattr(self, '_fs', 48000.0)) > 1e-9:
            self._fs = float(fs)
            self._eff_fs = self._fs
            self._freqs = default_freq_grid(self._eff_fs, n=2048, fmin=10.0)
            from .util import apply_viewbox_limits
            apply_viewbox_limits(self.plot, self._fs)

    def _parse_hex(self, which: str = 'A') -> Optional[SOS]:
        try:
            edits = self.edits_A if which == 'A' else self.edits_B
            def val_or_zero(w: QtWidgets.QLineEdit, bits: int) -> float:
                s = w.text().strip() if w and w.text() is not None else ''
                if not s:
                    s = '0x00000000'
                return hex_twos_to_float(s, bits)
            b0 = val_or_zero(edits[0], 31)
            b1 = val_or_zero(edits[1], 30)
            b2 = val_or_zero(edits[2], 31)
            a1 = val_or_zero(edits[3], 30)
            a2 = val_or_zero(edits[4], 31)
            a1, a2 = self._apply_sign_convention(a1, a2)
            return SOS(b0, b1, b2, a1, a2)
        except Exception:
            return None

    def _on_analyze(self):
        # Prune any stray PlotDataItems before updating to prevent overdraw artifacts
        try:
            items = getattr(self.plot, 'listDataItems', lambda: [])()
            for it in list(items):
                if it not in getattr(self, '_plot_items_allowed', set()):
                    try:
                        self.plot.removeItem(it)
                    except Exception:
                        pass
        except Exception:
            pass
        # Update effective fs from rate selection
        rate_txt = self.combo_rate.currentText() if hasattr(self, 'combo_rate') else 'fs (48 kHz)'
        if rate_txt.startswith('fs/2'):
            self._eff_fs = self._fs / 2.0
        elif rate_txt.startswith('2×fs') or rate_txt.startswith('2xfs') or rate_txt.startswith('2×'):
            self._eff_fs = self._fs * 2.0
        else:
            self._eff_fs = self._fs
        self._freqs = default_freq_grid(self._eff_fs, n=2048, fmin=10.0)

        sosA = self._parse_hex('A')
        sosB = self._parse_hex('B')
        if sosA is None and sosB is None:
            QtWidgets.QMessageBox.warning(self, 'Coef Check', 'Invalid hex inputs for both sets. Expect 32-bit two\'s complement like 0x7FFFFFFF.')
            return
        # Plot A
        if sosA is not None:
            HdbA = sos_response_db(sosA, self._freqs, self._eff_fs)
            # Apply stage B0 if provided (Q5.27)
            try:
                txt = self.edit_stage_A.text().strip()
                if txt:
                    stA = hex_twos_to_float(txt, 27)
                    if stA > 0:
                        import math
                        HdbA = HdbA + (20.0 * math.log10(stA))
            except Exception:
                pass
            self.curve_sos.setData(self._freqs, HdbA)
            resA = self._detect_hp_lpf_peak(sosA, HdbA)
            self._show_result(resA)
        else:
            self.curve_sos.setData([], [])
        # Plot B
        if sosB is not None:
            HdbB = sos_response_db(sosB, self._freqs, self._eff_fs)
            try:
                txt = self.edit_stage_B.text().strip()
                if txt:
                    stB = hex_twos_to_float(txt, 27)
                    if stB > 0:
                        import math
                        HdbB = HdbB + (20.0 * math.log10(stB))
            except Exception:
                pass
            self.curve_sos_B.setData(self._freqs, HdbB)
        else:
            self.curve_sos_B.setData([], [])

    def _apply_json(self, which: str):
        import json as _json
        import ast as _ast
        txt = self.edit_json_A.text() if which == 'A' else self.edit_json_B.text()
        if not txt:
            return
        data = None
        try:
            data = _json.loads(txt)
        except Exception:
            try:
                data = _ast.literal_eval(txt)
            except Exception:
                QtWidgets.QMessageBox.warning(self, 'Coef Check', 'Invalid JSON/dict. Expect keys B0,B1,B2,A1,A2 and optional ST.')
                return
        if not isinstance(data, dict):
            QtWidgets.QMessageBox.warning(self, 'Coef Check', 'Parsed value is not an object/dict.')
            return
        # normalize keys to upper for convenience (accept b0, a1, st)
        data_u = {str(k).upper(): v for k, v in data.items()}
        def norm_hex(v) -> str:
            if isinstance(v, str):
                s = v.strip()
                if s.lower().startswith('0x'):
                    try:
                        n = int(s, 16)
                        return f"0x{n & 0xFFFFFFFF:08X}"
                    except Exception:
                        return s
                # allow raw hex without 0x
                try:
                    n = int(s, 16)
                    return f"0x{n & 0xFFFFFFFF:08X}"
                except Exception:
                    return s
            else:
                try:
                    n = int(v)
                    return f"0x{n & 0xFFFFFFFF:08X}"
                except Exception:
                    return ''
        edits = self.edits_A if which == 'A' else self.edits_B
        key_map = [('B0', 0), ('B1', 1), ('B2', 2), ('A1', 3), ('A2', 4)]
        for k, idx in key_map:
            if k in data_u:
                hx = norm_hex(data_u[k])
                edits[idx].setText(hx)
        # Stage key variants: ST or STAGE or BQ15B0
        stage_val = None
        for sk in ('ST', 'STAGE', 'BQ15B0'):
            if sk in data_u:
                stage_val = data_u[sk]
                break
        if stage_val is not None:
            hx = norm_hex(stage_val)
            if which == 'A':
                self.edit_stage_A.setText(hx)
            else:
                self.edit_stage_B.setText(hx)

    def _show_result(self, res: DetectResult):
        self.lbl_kind.setText(f"Type: {res.kind}")
        self.lbl_f0.setText(f"f0: {res.f0:.2f} Hz" if res.f0 else "f0: —")
        self.lbl_q.setText(f"Q: {res.q:.3f}" if res.q else "Q: —")
        self.lbl_gain.setText(f"Gain: {res.gain_db:.2f} dB" if res.gain_db else "Gain: —")
        self.lbl_note.setText(res.note or "")

        # Update fit curve if we have a parametric match
        p = None
        if res.kind == 'LPF' and res.f0 and res.q:
            p = BiquadParams(typ=FilterType.LPF, fs=self._eff_fs, f0=res.f0, q=res.q, gain_db=0.0)
        elif res.kind == 'HPF' and res.f0 and res.q:
            p = BiquadParams(typ=FilterType.HPF, fs=self._eff_fs, f0=res.f0, q=res.q, gain_db=0.0)
        elif res.kind == 'Peaking' and res.f0 and res.q and res.gain_db is not None:
            p = BiquadParams(typ=FilterType.PEAK, fs=self._eff_fs, f0=res.f0, q=res.q, gain_db=res.gain_db)
        if p is not None and getattr(self, 'chk_fit', None) is not None and self.chk_fit.isChecked():
            sos_fit = design_biquad(p)
            H_fit = sos_response_db(sos_fit, self._freqs, self._eff_fs)
            self.curve_fit.setData(self._freqs, H_fit)
        else:
            self.curve_fit.setData([], [])
            pass

    def _apply_sign_convention(self, a1: float, a2: float) -> tuple[float, float]:
        mode = self.combo_sign.currentText() if hasattr(self, 'combo_sign') else 'Standard (a1,a2)'
        if 'PPC3' in mode:
            return -a1, -a2
        return a1, a2

    # ---- Detection helpers ----
    def _detect_hp_lpf_peak(self, sos: SOS, Hdb: np.ndarray) -> DetectResult:
        # Compute quick DC / Nyquist magnitudes
        def mag_at(z1: float) -> float:
            if z1 == 1.0:
                num = sos.b0 + sos.b1 + sos.b2
                den = 1.0 + sos.a1 + sos.a2
            else:  # z = -1
                num = sos.b0 - sos.b1 + sos.b2
                den = 1.0 - sos.a1 + sos.a2
            if den == 0:
                return np.inf
            return float(abs(num / den))

        mag_dc = mag_at(1.0)
        mag_ny = mag_at(-1.0)
        dc_db = 20*np.log10(max(mag_dc, 1e-12))
        ny_db = 20*np.log10(max(mag_ny, 1e-12))

        # Side plateaus from response
        n = len(Hdb)
        low_avg = float(np.median(Hdb[: max(8, n//20)]))
        high_avg = float(np.median(Hdb[-max(8, n//20):]))

        # Try LPF / HPF classification
        # Criteria: one side near 0 dB (+/- 1.5 dB), other at least 6 dB lower
        near0 = lambda x: abs(x) < 1.5
        if near0(low_avg) and (high_avg < low_avg - 6.0):
            f0, q = self._estimate_fc_q(Hdb, passband='low')
            return DetectResult('LPF', f0=f0, q=q, gain_db=0.0)
        if near0(high_avg) and (low_avg < high_avg - 6.0):
            f0, q = self._estimate_fc_q(Hdb, passband='high')
            return DetectResult('HPF', f0=f0, q=q, gain_db=0.0)

        # Otherwise, consider peaking: both sides near 0 dB, mid‑band deviation
        mid_idx = int(np.argmax(np.abs(Hdb)))
        peak_db = float(Hdb[mid_idx])
        if near0(low_avg) and near0(high_avg) and abs(peak_db) > 0.75:
            f0, q, g = self._estimate_peak_params(Hdb)
            return DetectResult('Peaking', f0=f0, q=q, gain_db=g)

        # Fallback: pick the better of LPF/HPF by error fitting
        cand = []
        for kind in ('LPF','HPF'):
            f0, q = self._estimate_fc_q(Hdb, passband=('low' if kind=='LPF' else 'high'))
            p = BiquadParams(typ=FilterType.LPF if kind=='LPF' else FilterType.HPF,
                             fs=self._eff_fs, f0=f0 or 1000.0, q=q or 0.707, gain_db=0.0)
            H = sos_response_db(design_biquad(p), self._freqs, self._eff_fs)
            err = float(np.mean((H - Hdb)**2))
            cand.append((err, kind, f0, q))
        err, kind, f0, q = sorted(cand, key=lambda x: x[0])[0]
        return DetectResult(kind, f0=f0, q=q, gain_db=0.0, note="Uncertain; best fit among LPF/HPF.")

    def _estimate_fc_q(self, Hdb: np.ndarray, passband: str) -> Tuple[Optional[float], Optional[float]]:
        # Reference: passband median near either low or high end
        n = len(Hdb)
        if passband == 'low':
            ref = float(np.median(Hdb[: max(8, n//20)]))
            target = ref - 3.0
            search = Hdb
        else:
            ref = float(np.median(Hdb[-max(8, n//20):]))
            target = ref - 3.0
            search = Hdb[::-1]
        # Find -3 dB crossing index
        idx = None
        for i in range(1, len(search)):
            if (search[i-1] - target) * (search[i] - target) <= 0:
                idx = i
                break
        if idx is None:
            return None, None
        # Interpolate in log frequency
        if passband == 'low':
            f1, f2 = self._freqs[idx-1], self._freqs[idx]
            y1, y2 = Hdb[idx-1], Hdb[idx]
        else:
            # reversed
            f1, f2 = self._freqs[-idx], self._freqs[-idx-1]
            y1, y2 = Hdb[-idx], Hdb[-idx-1]
        t = 0.0 if y2==y1 else (target - y1) / (y2 - y1)
        fc = float(f1 * (f2/f1)**t)

        # Rough Q estimate via local slope shape: refine by small grid search
        qs = np.geomspace(0.3, 3.0, 12)
        fs = np.geomspace(fc*0.7, fc*1.3, 15)
        best = (1e9, None, None)
        for f0 in fs:
            for q in qs:
                typ = FilterType.LPF if passband=='low' else FilterType.HPF
                p = BiquadParams(typ=typ, fs=self._fs, f0=float(f0), q=float(q), gain_db=0.0)
                H = sos_response_db(design_biquad(p), self._freqs, self._fs)
                err = float(np.mean((H - Hdb)**2))
                if err < best[0]:
                    best = (err, f0, q)
        return float(best[1]) if best[1] else None, float(best[2]) if best[2] else None

    def _estimate_peak_params(self, Hdb: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        idx0 = int(np.argmax(np.abs(Hdb)))
        f0 = float(self._freqs[idx0])
        g = float(Hdb[idx0])
        # Find -3 dB points around peak relative to peak level
        target = g - 3.0 if g >= 0 else g + 3.0
        # Search left
        i1 = None
        for i in range(idx0-1, 0, -1):
            if (Hdb[i] - target) * (Hdb[i+1] - target) <= 0:
                i1 = i
                break
        # Search right
        i2 = None
        for i in range(idx0+1, len(Hdb)-1):
            if (Hdb[i] - target) * (Hdb[i+1] - target) <= 0:
                i2 = i
                break
        if i1 is not None and i2 is not None:
            # Interpolate both sides in log freq
            def interp(i):
                f1, f2 = self._freqs[i], self._freqs[i+1]
                y1, y2 = Hdb[i], Hdb[i+1]
                t = 0.0 if y2==y1 else (target - y1) / (y2 - y1)
                return float(f1 * (f2/f1)**t)
            fL = interp(i1)
            fR = interp(i2)
            bw = max(fR - fL, 1e-9)
            q0 = f0 / bw
        else:
            q0 = 0.707

        # Refine by small grid search around initial guesses
        f_candidates = np.geomspace(f0*0.85, f0*1.15, 11)
        q_candidates = np.geomspace(max(0.2, q0*0.5), min(20.0, q0*2.0), 13)
        g_candidates = np.linspace(g*0.8, g*1.2, 15)
        best = (1e9, None, None, None)
        for ff in f_candidates:
            for qq in q_candidates:
                for gg in g_candidates:
                    p = BiquadParams(typ=FilterType.PEAK, fs=self._fs, f0=float(ff), q=float(qq), gain_db=float(gg))
                    H = sos_response_db(design_biquad(p), self._freqs, self._fs)
                    err = float(np.mean((H - Hdb)**2))
                    if err < best[0]:
                        best = (err, ff, qq, gg)
        return float(best[1]) if best[1] else None, float(best[2]) if best[2] else None, float(best[3]) if best[3] else None
