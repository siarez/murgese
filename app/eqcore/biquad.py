from __future__ import annotations
import enum
import numpy as np
from dataclasses import dataclass

class FilterType(str, enum.Enum):
    PEAK = "Peak (Peaking EQ)"
    LSHELF = "Low Shelf"
    HSHELF = "High Shelf"
    LPF = "Low-pass"
    HPF = "High-pass"
    BPF = "Band-pass (const peak)"
    NOTCH = "Notch"
    ALLPASS = "All-pass (Unity)"

@dataclass
class BiquadParams:
    typ: FilterType
    fs: float       # sample rate (Hz)
    f0: float       # center / cutoff (Hz)
    q: float        # Q or bandwidth shaping
    gain_db: float  # dB (used for peak/shelves; ignored for pure HPF/LPF/BPF/Notch as expected)

@dataclass
class SOS:
    # normalized so a0 = 1
    b0: float; b1: float; b2: float
    a1: float; a2: float

def _first_order_common_k(fs: float, f0: float) -> float:
    # Bilinear transform prewarped factor
    # k = 1 / tan(pi*f0/fs)
    return 1.0 / np.tan(np.pi * f0 / fs)

def design_first_order_lpf(fs: float, f0: float) -> SOS:
    k = _first_order_common_k(fs, f0)
    den = (k + 1.0)
    b0 = 1.0 / den
    b1 = 1.0 / den
    b2 = 0.0
    a1 = (1.0 - k) / den
    a2 = 0.0
    return SOS(b0, b1, b2, a1, a2)

def design_first_order_hpf(fs: float, f0: float) -> SOS:
    k = _first_order_common_k(fs, f0)
    den = (k + 1.0)
    b0 = k / den
    b1 = -k / den
    b2 = 0.0
    a1 = (1.0 - k) / den
    a2 = 0.0
    return SOS(b0, b1, b2, a1, a2)

def _rbj_peak(p: BiquadParams) -> SOS:
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

def design_biquad(p: BiquadParams) -> SOS:
    if p.typ == FilterType.ALLPASS:
        return SOS(1.0, 0.0, 0.0, 0.0, 0.0)
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
    # log-spaced up to Nyquist
    fmax = fs / 2.0
    return np.geomspace(max(fmin, 1e-3), fmax, n)

def sos_response_db(sos: SOS, f: np.ndarray, fs: float) -> np.ndarray:
    # Evaluate H(e^{jw}) for one section; vectorized
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
    # Sum dB of each section's magnitude (since multiplying mags in linear)
    acc = np.zeros_like(f, dtype=float)
    for s in sections:
        acc += sos_response_db(s, f, fs)
    return acc

def sos_response_complex(sos: SOS, f: np.ndarray, fs: float) -> np.ndarray:
    # Complex frequency response H(e^{jw}) for one section
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
    H = np.ones_like(f, dtype=complex)
    for s in sections:
        H *= sos_response_complex(s, f, fs)
    return H
