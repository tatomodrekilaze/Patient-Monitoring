"""Shared 1-D signal tools: resampling, bandpass, spectral peak."""
import numpy as np
from scipy import signal as sps


def resample_uniform(t, x, fs, duration=None):
    """Interpolate irregular samples onto a uniform grid ending at t[-1]."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if len(t) < 2:
        return None, None
    t_end = t[-1]
    t_start = t[0] if duration is None else max(t[0], t_end - duration)
    n = int((t_end - t_start) * fs) + 1
    if n < 2:
        return None, None
    tu = t_end - np.arange(n)[::-1] / fs
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    return tu, np.stack([np.interp(tu, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def bandpass(x, fs, lo, hi, order=2):
    nyq = fs / 2
    hi = min(hi, nyq * 0.95)
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x, axis=0)


def spectral_peak(x, fs, lo, hi, nfft=4096, halfwidth=0.06):
    """Dominant frequency in [lo, hi] and the share of band power around it (0..1)."""
    x = np.asarray(x, float) - np.mean(x)
    n = max(nfft, len(x))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= lo) & (f <= hi)
    total = spec[band].sum()
    if not band.any() or total <= 0:
        return None, 0.0
    fb, sb = f[band], spec[band]
    fp = fb[np.argmax(sb)]
    near = np.abs(fb - fp) <= halfwidth
    return float(fp), float(sb[near].sum() / total)
