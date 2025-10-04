"""
EQ core: biquad design and evaluation utilities.

This module provides:
  - FilterType: high-level filter intent (RBJ “cookbook” family + first-order LPF/HPF)
  - BiquadParams: input parameters to design a single section (a0 normalized to 1)
  - SOS: second-order section coefficients (b0,b1,b2,a1,a2) with a0=1
  - Design helpers for first-order LPF/HPF and RBJ-based biquads
  - Fast vectorized frequency-response evaluators (magnitude dB and complex)

Notes
  - All designs normalize the direct-form IIR such that a0 == 1; callers writing
    to hardware with different normalization should scale accordingly.
  - For RBJ designs, see “Cookbook formulae for audio EQ biquad filter coefficients”
    by Robert Bristow‑Johnson. The formulas used here mirror that reference.
  - Numerics: functions are vectorized with NumPy and avoid log of zero by
    clamping magnitudes to small epsilons.
"""
from __future__ import annotations
import enum
import numpy as np
from dataclasses import dataclass

class FilterType(str, enum.Enum):
    """Supported biquad topologies at the intent level.

    Values are stringy for friendlier display; logic should compare by identity.
    """
    PEAK = "Peak (Peaking EQ)"
    LSHELF = "Low Shelf"
    HSHELF = "High Shelf"
    LPF = "Low-pass"
    HPF = "High-pass"
    BPF = "Band-pass (const peak)"
    NOTCH = "Notch"
    ALLPASS = "All-pass (Unity)"
    ALLPASS1 = "All-pass 1st (Phase)"
    ALLPASS2 = "All-pass 2nd (Phase)"

@dataclass
class BiquadParams:
    """User-level parameters to design a single biquad section.

    Attributes
    - typ: FilterType (intent)
    - fs: Sample rate in Hz
    - f0: Corner / center frequency in Hz
    - q:  Quality factor (bandwidth shaping)
    - gain_db: Gain in dB (used by peak/shelves; ignored for pure LPF/HPF/BPF/NOTCH)
    """
    typ: FilterType
    fs: float       # sample rate (Hz)
    f0: float       # center / cutoff (Hz)
    q: float        # Q or bandwidth shaping
    gain_db: float  # dB (used for peak/shelves; ignored for pure HPF/LPF/BPF/Notch as expected)

@dataclass
class SOS:
    """Second-order section coefficients with a0 normalized to 1.

    The transfer function is:
        H(z) = (b0 + b1 z^{-1} + b2 z^{-2}) / (1 + a1 z^{-1} + a2 z^{-2}).
    """
    # normalized so a0 = 1
    b0: float; b1: float; b2: float
    a1: float; a2: float

def _first_order_common_k(fs: float, f0: float) -> float:
    """Bilinear-transform prewarped factor for 1st-order prototypes.

    k = 1 / tan(pi * f0 / fs)
    """
    return 1.0 / np.tan(np.pi * f0 / fs)

def design_first_order_lpf(fs: float, f0: float) -> SOS:
    """Design a 1st-order low-pass (RBJ-equivalent) with a0==1.

    Uses bilinear transform on analog prototype H(s)=1/(s+1) scaled by prewarp.
    """
    k = _first_order_common_k(fs, f0)
    den = (k + 1.0)
    b0 = 1.0 / den
    b1 = 1.0 / den
    b2 = 0.0
    a1 = (1.0 - k) / den
    a2 = 0.0
    return SOS(b0, b1, b2, a1, a2)

def design_first_order_hpf(fs: float, f0: float) -> SOS:
    """Design a 1st-order high-pass with a0==1 via bilinear transform."""
    k = _first_order_common_k(fs, f0)
    den = (k + 1.0)
    b0 = k / den
    b1 = -k / den
    b2 = 0.0
    a1 = (1.0 - k) / den
    a2 = 0.0
    return SOS(b0, b1, b2, a1, a2)

def _rbj_peak(p: BiquadParams) -> SOS:
    """RBJ peaking EQ design (gain_db used; a0 normalized to 1)."""
    A = 10.0 ** (p.gain_db / 40.0)
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)

    b0 = 1 + alpha*A
    b1 = -2*cosw
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*cosw
    a2 = 1 - alpha/A
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_lowshelf(p: BiquadParams) -> SOS:
    """RBJ low-shelf design (gain_db, Q as slope control)."""
    A = 10.0 ** (p.gain_db / 40.0)
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 =    A*((A+1) - (A-1)*cosw + 2*sqrtA*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw)
    b2 =    A*((A+1) - (A-1)*cosw - 2*sqrtA*alpha)
    a0 =       (A+1) + (A-1)*cosw + 2*sqrtA*alpha
    a1 =   -2*((A-1) + (A+1)*cosw)
    a2 =       (A+1) + (A-1)*cosw - 2*sqrtA*alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_highshelf(p: BiquadParams) -> SOS:
    """RBJ high-shelf design (gain_db, Q as slope control)."""
    A = 10.0 ** (p.gain_db / 40.0)
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 =    A*((A+1) + (A-1)*cosw + 2*sqrtA*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw)
    b2 =    A*((A+1) + (A-1)*cosw - 2*sqrtA*alpha)
    a0 =       (A+1) - (A-1)*cosw + 2*sqrtA*alpha
    a1 =    2*((A-1) - (A+1)*cosw)
    a2 =       (A+1) - (A-1)*cosw - 2*sqrtA*alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_lpf(p: BiquadParams) -> SOS:
    """RBJ 2nd-order low-pass design (Q defines damping)."""
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    b0 = (1 - cosw)/2
    b1 = 1 - cosw
    b2 = (1 - cosw)/2
    a0 = 1 + alpha
    a1 = -2*cosw
    a2 = 1 - alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_hpf(p: BiquadParams) -> SOS:
    """RBJ 2nd-order high-pass design (Q defines damping)."""
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    b0 =  (1 + cosw)/2
    b1 = -(1 + cosw)
    b2 =  (1 + cosw)/2
    a0 = 1 + alpha
    a1 = -2*cosw
    a2 = 1 - alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_bpf_const_peak(p: BiquadParams) -> SOS:
    """RBJ band-pass (constant peak gain) design.

    This is the “constant 0 dB peak gain” variant (b1=0), often preferable for
    visualizing band emphasis without gain at DC/nyquist.
    """
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    # Constant 0 dB peak gain at f0
    b0 =  alpha
    b1 =  0.0
    b2 = -alpha
    a0 =  1 + alpha
    a1 = -2*cosw
    a2 =  1 - alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_notch(p: BiquadParams) -> SOS:
    """RBJ notch design (zeros on unit circle at ±e^{jw0})."""
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    b0 = 1
    b1 = -2*cosw
    b2 = 1
    a0 = 1 + alpha
    a1 = -2*cosw
    a2 = 1 - alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _allpass1(p: BiquadParams) -> SOS:
    """1st-order digital all-pass with unity magnitude.

    H(z) = (a + z^{-1}) / (1 + a z^{-1}) where
      a = (1 - tan(w0/2)) / (1 + tan(w0/2)),  w0 = 2*pi*f0/fs

    Q is ignored for this topology.
    """
    w0 = 2*np.pi*p.f0/p.fs
    t = np.tan(w0/2.0)
    if np.isinf(t):
        a = -1.0  # limit as tan -> inf
    else:
        a = (1.0 - t) / (1.0 + t)
    # With a0 normalized to 1
    b0 = a
    b1 = 1.0
    b2 = 0.0
    a1 = a
    a2 = 0.0
    return SOS(b0, b1, b2, a1, a2)

def _rbj_allpass2(p: BiquadParams) -> SOS:
    """RBJ 2nd-order all-pass (phase) biquad with a0 normalized to 1.

    Magnitude is unity for all frequencies; phase rotates around f0 with Q.
    """
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    b0 = 1 - alpha
    b1 = -2*cosw
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2*cosw
    a2 = 1 - alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def _rbj_allpass2(p: BiquadParams) -> SOS:
    """RBJ 2nd-order all-pass (phase) biquad with a0 normalized to 1.

    Magnitude is unity for all frequencies; phase rotates around f0 with Q.
    """
    w0 = 2*np.pi*p.f0/p.fs
    alpha = np.sin(w0)/(2*p.q)
    cosw = np.cos(w0)
    b0 = 1 - alpha
    b1 = -2*cosw
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2*cosw
    a2 = 1 - alpha
    inv = 1.0/a0
    return SOS(b0*inv, b1*inv, b2*inv, a1*inv, a2*inv)

def design_biquad(p: BiquadParams) -> SOS:
    """Dispatch to the appropriate biquad design based on FilterType.

    Returns an SOS with a0 normalized to 1. ALLPASS returns a unity pass-through.
    Raises ValueError for unsupported types.
    """
    if p.typ == FilterType.ALLPASS:
        return SOS(1.0, 0.0, 0.0, 0.0, 0.0)
    if p.typ == FilterType.ALLPASS1:
        return _allpass1(p)
    if p.typ == FilterType.ALLPASS2:
        return _rbj_allpass2(p)
    if p.typ == FilterType.PEAK:
        return _rbj_peak(p)
    if p.typ == FilterType.LSHELF:
        return _rbj_lowshelf(p)
    if p.typ == FilterType.HSHELF:
        return _rbj_highshelf(p)
    if p.typ == FilterType.LPF:
        return _rbj_lpf(p)
    if p.typ == FilterType.HPF:
        return _rbj_hpf(p)
    if p.typ == FilterType.BPF:
        return _rbj_bpf_const_peak(p)
    if p.typ == FilterType.NOTCH:
        return _rbj_notch(p)
    raise ValueError("Unsupported filter type")

def default_freq_grid(fs: float, n: int = 1024, fmin: float = 10.0) -> np.ndarray:
    """Log-spaced frequency grid from fmin to Nyquist inclusive.

    Parameters
    - fs: sample rate (Hz)
    - n: number of points
    - fmin: lowest frequency (Hz), clamped to >= 1e-3
    """
    fmax = fs / 2.0
    return np.geomspace(max(fmin, 1e-3), fmax, n)

def sos_response_db(sos: SOS, f: np.ndarray, fs: float) -> np.ndarray:
    """Evaluate magnitude in dB for a single SOS over frequencies f.

    Vectorized direct evaluation of H(e^{jw}) using trigonometric identities to
    avoid explicit complex exponentials for speed.
    """
    w = 2*np.pi*f/fs
    cw = np.cos(w); sw = np.sin(w)
    # Using cos(2w) = 2cos^2(w) - 1 and sin(2w) = 2sin(w)cos(w)
    cos2w = 2*cw*cw - 1.0
    sin2w = 2*sw*cw

    num_re = sos.b0 + sos.b1*cw + sos.b2*cos2w
    num_im = -(sos.b1*sw + sos.b2*sin2w)
    den_re = 1.0 + sos.a1*cw + sos.a2*cos2w
    den_im = -(sos.a1*sw + sos.a2*sin2w)

    denom = den_re*den_re + den_im*den_im
    re = (num_re*den_re + num_im*den_im)/denom
    im = (num_im*den_re - num_re*den_im)/denom
    mag = np.hypot(re, im)
    return 20*np.log10(np.maximum(mag, 1e-20))

def cascade_response_db(sections: list[SOS], f: np.ndarray, fs: float) -> np.ndarray:
    """Evaluate cascade magnitude in dB by summing per-section dB magnitudes.

    Equivalent to multiplying magnitudes in linear and taking 20*log10.
    Phase interactions are ignored; use cascade_response_complex for full H.
    """
    acc = np.zeros_like(f, dtype=float)
    for s in sections:
        acc += sos_response_db(s, f, fs)
    return acc

def sos_response_complex(sos: SOS, f: np.ndarray, fs: float) -> np.ndarray:
    """Return complex H(e^{jw}) for a single SOS at frequencies f."""
    w = 2*np.pi*f/fs
    cw = np.cos(w); sw = np.sin(w)
    cos2w = 2*cw*cw - 1.0
    sin2w = 2*sw*cw

    num_re = sos.b0 + sos.b1*cw + sos.b2*cos2w
    num_im = -(sos.b1*sw + sos.b2*sin2w)
    den_re = 1.0 + sos.a1*cw + sos.a2*cos2w
    den_im = -(sos.a1*sw + sos.a2*sin2w)

    denom = den_re*den_re + den_im*den_im
    re = (num_re*den_re + num_im*den_im)/denom
    im = (num_im*den_re - num_re*den_im)/denom
    return re + 1j*im

def cascade_response_complex(sections: list[SOS], f: np.ndarray, fs: float) -> np.ndarray:
    """Complex cascade response by multiplying section responses in linear domain."""
    H = np.ones_like(f, dtype=complex)
    for s in sections:
        H *= sos_response_complex(s, f, fs)
    return H
