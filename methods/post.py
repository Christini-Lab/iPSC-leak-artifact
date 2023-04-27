#
# Data post-processing methods
#
import numpy as np
import pywt
import scipy.interpolate
from scipy.signal import find_peaks


# Scaling for "angles" on APs
# 1000 ms period, 120 mV range
T_SCALE = 1000
V_SCALE = 120


def _reconstruct(x0, coefficients, levels, wavelet, mode):
    """
    Reconstruct a signal from the given ``coefficients`` with a fixed number of
    detail ``levels`` (starts at 1).
    """
    assert levels >= 1, 'levels must be 1 or greater'

    i = len(coefficients) - 1 - levels
    y1 = pywt.waverec(coefficients[:1 + levels], wavelet, mode=mode)

    # Reconstruct x values
    # I checked this against values returned when also waveletting x0
    x1 = x0[::2**i]                      # This returns a view
    if i > 0:
        x1 = x1 + 0.5 * (x1[1] - x1[0])  # x1+=...  changes x0!

        # Remove extra points introduced by padding, and rescale y1
        a = (len(y1) - len(x1)) // 2
        b = a + len(x1)
        y1 = y1[a:b] / 2**(i / 2)

    # Wavelet decomposition sometimes extrapolates. We don't want this,
    # so chop off any extrapolated points
    if x1[-1] > x0[-1]:
        x1 = x1[:-1]
        y1 = y1[:-1]
    if len(y1) > len(x1):
        y1 = y1[:-1]

    # Return
    return x1, y1


def denoise(x, y, threshold, wavelet='haar'):
    """
    Performs Haar-wavelet denoising on time series ``(x, y)`` and returns a
    downsampled series ``(x1, y1, z1)`` such that
    ``max(z1 = y1 - y) < threshold``.
    """
    # Input values as arrays
    x0, y0 = np.asarray(x), np.asarray(y)

    # Wavelet decomposition
    mode = 'smooth'
    coeffs = pywt.wavedec(y0, wavelet, mode=mode)

    # Interpolant for error values
    f = scipy.interpolate.interp1d(x0, y0)

    # Reconstruct with j detail levels
    x1 = x0
    for j in range(1, len(coeffs)):
        x1, y1 = _reconstruct(x0, coeffs, j, wavelet, mode)
        z1 = f(x1) - y1
        e = np.max(np.abs(z1))
        if e < threshold:
            return x1, y1, z1

    raise RuntimeError(f'Final error ({e}) is above threshold ({threshold}).')


def split(x, y, levels, wavelet='haar'):
    """
    Performs Haar-wavelet decomposition on time series ``(x, y)`` and returns a
    downsampled series ``(x1, y1, z1)`` where ``y1`` contains ``levels`` detail
    levels, and ``z1 = y1 - y(x1)``.
    """
    assert levels >= 1, 'levels must be 1 or greater'

    # Input values as arrays
    x0, y0 = np.asarray(x), np.asarray(y)

    # Decompose and reconstruct
    mode = 'smooth'

    coeffs = pywt.wavedec(y0, wavelet, mode=mode)
    x1, y1 = _reconstruct(x0, coeffs, levels, wavelet, mode)
    f = scipy.interpolate.interp1d(x0, y0)
    return x1, y1, f(x1) - y1


def ipeaks(x, y, lower=0, n=6):
    """
    Find peaks in time series ``x, y``, and returns their indices.

    A peak is the highest point in any contiguous region where ``y > lower``
    for at least ``n`` samples.
    """
    x, y = np.asarray(x), np.asarray(y)

    # Find regions
    above = np.nonzero(y > lower)[0]    # Indices in y that are above lower
    i1 = np.nonzero(above[1:] - above[:-1] > 1)[0] + 1
    i1 = np.concatenate(([0], i1))  # Indices in `above` where regions start
    i2 = np.concatenate((i1[1:], [len(above)])) - 1  # Indices where they end
    n = np.nonzero(i2 - i1 + 1 >= n)[0]  # Regions with size >= n
    i1, i2 = i1[n], i2[n]
    i1 = above[i1]  # Indices in y where regions start and end
    i2 = above[i2]
    del(above, n)

    # Find peaks
    peaks = np.empty(i1.shape, dtype=int)
    for i, js in enumerate(zip(i1, i2)):
        j1, j2 = js
        peaks[i] = j1 + np.argmax(y[j1:j2])

    return peaks


def rotated_peaks(t, v, a, i0=None, i1=None, tscale=None, vscale=None):
    """
    Takes a time series ``(t, v)``, divides it into sections, rotates each
    section by ``a`` radians and returns a tuple ``(ts, vs)`` with the times
    and voltages corresponding to the peaks in the rotated signal.

    Arguments:

    ``t``
        An array of times.
    ``v``
        An array of values.
    ``a``
        An angle in radians, or an array of angles.
    ``i0``
        An optional list of indices where the regions start. If not given, this
        will be set using a call to :meth:`peaks`.
    ``i1``
        An optional list of indices where the regions end. If not given, this
        will be set using a call to :meth:`peaks`.
    ``tscale``
        An optional "natural scale" for the points in ``t``.
    ``vscale``
        An optional "natural scale" for the points in ``v``.

    """
    # Get regio indices
    if i0 is None or i1 is None:
        peaks = ipeaks(t, v)
        i0, i1 = peaks[:-1], peaks[1:]
    assert len(i0) == len(i1)

    # Get t and v scales
    if tscale is None:
        tscale = T_SCALE
    if vscale is None:
        vscale = V_SCALE

    # Get angles
    if np.isscalar(a):
        cos, sin = np.ones(len(i0)) * np.cos(a), np.ones(len(i0)) * np.sin(a)
    else:
        assert len(a) == len(i0)
        cos, sin = np.cos(a), np.sin(a)

    # Calculate
    ts, vs = [], []
    for i0, i1, cos, sin in zip(i0, i1, cos, sin):
        x, y = t[i0:i1] / tscale, v[i0:i1] / vscale
        x, y = x * cos - y * sin, x * sin + y * cos
        k = np.argmax(y)
        x, y = x[k], y[k]
        ts.append(x * cos + y * sin)
        vs.append(-x * sin + y * cos)
    return np.array(ts) * tscale, np.array(vs) * vscale


def inearest(t, x):
    """
    Returns the indices in ``t`` that are closest to ``x``, assuming ``t`` is a
    sorted numpy array.
    """
    # Get indices where ``needle`` could be inserted
    j = np.searchsorted(t, x)

    # j are now all indices such that x[i] <= t[j[i]], but their value might be
    # closer to t[js - 1], so check and decrease by 1 if necessary
    j[j == 0] += 1    # Will be undone by next step
    j[x - t[j - 1] < t[j] - x] -= 1
    return j


def aps(t, v, tmin=-50, tmax=800, denoising_levels=13):
    """
    Takes a time series ``t, v``, finds the peaks, and returns a list of time
    series such that each contains a single AP, centered with its peak at t=0.

    Returns a tuple ``(t, vs)`` where ``t`` is an array of times ranging from
    ``tmin`` to ``tmax``, and with the same sampling rate as ``t``. The array
    ``vs`` contains an AP on each row, and is padded and/or truncated to match
    the range ``tmin, tmax``.
    """
    t0, v0 = t, v
    del(t, v)

    # Find peaks in denoised signal, translate to original signal
    t1, v1, _ = split(t0, v0, denoising_levels)
    dt1 = np.mean(t1[1:] - t1[:-1])
    na = max(4, 1 + int(20 // dt1))
    peaks = ipeaks(t1, v1, n=na)
    peaks = inearest(t0, t1[peaks])
    del(t1, v1, na)

    # Create output arrays
    dt0 = np.mean(t0[1:] - t0[:-1])
    t = np.arange(tmin, tmax, dt0)
    n = len(t)
    m = len(peaks) - 1
    vs = np.empty((m, n))
    vs[:] = np.nan

    # Populate vs
    n0 = len(t0)
    for k in range(m):
        # Indices for slice of v0: i, j
        i = peaks[k] + int(tmin // dt0)
        j = i + n

        # Indices for insertion in vs: a, b
        a, b = 0, n
        if i < 0:
            a = -i
            i = 0
        if j > n0:
            b -= j - n0
            j = n0

        # Store
        vs[k, a:b] = v0[i:j]

    return t, vs


def biomarkers(t, v, denoise=13):
    """
    Calculates and returns a set of biomarkers for all APs in the given signal.
    """
    # Scale
    tsc, vsc = T_SCALE, V_SCALE

    # De-noise
    if denoise:
        t1, v1, e1a = split(t, v, 13)
    else:
        t1, v1 = t, v

    # Find peaks
    dt1 = np.mean(t1[1:] - t1[:-1])
    na = max(4, 1 + int(20 // dt1))
    peaks = ipeaks(t1, v1, lower=-20, n=na)

    tpeaks = t1[peaks]
    dt_peak = tpeaks[1:] - tpeaks[:-1]
    v_peak = v1[peaks[:-1]]

    # Find mdps
    if len(v_peak) > 1:
        i0, i1 = peaks[0], peaks[-1]
        mdps = i0 + ipeaks(t1[i0:i1 + 1], -v1[i0:i1 + 1], lower=20, n=na)
        if len(mdps) != (len(peaks) -1):
            import pdb
            pdb.set_trace()
        assert len(mdps) == len(peaks) - 1
        dt_mdp = t1[mdps] - tpeaks[:-1]
        v_mdp = v1[mdps]
    else:
        v_mdp = []

    # Check that we have at least 2 peaks and 1 mdp
    if len(v_mdp) < 1:
        # Flat signal, return mdp only
        m = np.zeros(17)
        m[10] = np.min(v1)
        return np.array([m]).T

    # Find two 2 AP points
    i0, i1 = peaks[:-1], mdps
    ap_angles = np.arctan2((v1[i0] - v1[i1]) / vsc, (t1[i1] - t1[i0]) / tsc)
    tap1, v_ap1 = rotated_peaks(t1, v1, ap_angles, i0, i1)
    tap2, v_ap2 = rotated_peaks(t1, v1, ap_angles + np.pi, i0, i1)
    dt_ap1 = tap1 - tpeaks[:-1]
    dt_ap2 = tap2 - tpeaks[:-1]

    # Find diastolic point
    i0, i1 = mdps, peaks[1:]
    d_angles = np.pi + np.arctan2(
        (v1[i0] - v1[i1]) / vsc, (t1[i1] - t1[i0]) / tsc)
    td, v_d = rotated_peaks(t1, v1, d_angles, i0, i1)
    dt_d = tpeaks[1:] - td

    # Find upstroke points
    id1 = inearest(t1, td)
    i0, i1 = id1, peaks[1:]
    up_angles = np.arctan2((v1[i0] - v1[i1]) / vsc, (t1[i1] - t1[i0]) / tsc)
    tup1, v_up1 = rotated_peaks(t1, v1, up_angles + np.pi, i0, i1)
    tup2, v_up2 = rotated_peaks(t1, v1, up_angles, i0, i1)
    dt_up1 = tpeaks[1:] - tup1
    dt_up2 = tpeaks[1:] - tup2

    # Find upstroke velocity
    # ddt1 = 0.5 * dt1 + t1[:-1]
    ddv1 = (v1[1:] - v1[:-1]) / (t1[1:] - t1[:-1])
    iup1 = inearest(t1, tup1)
    iup2 = inearest(t1, tup2)
    imax = np.array([i + np.argmax(ddv1[i:j]) for i, j in zip(iup1, iup2)])
    # t_max = ddt1[imax]
    # v_max = 0.5 * (v1[imax] + v1[imax + 1])
    dvdt_max = ddv1[imax]

    import matplotlib.pyplot as plt
    # Find APD90s
    peak_idxs = find_peaks(np.diff(v), height=.1, distance=100)[0]
    min_v_idxs = find_peaks(-v[peak_idxs[0]:], height=30, distance=100)[0]+peak_idxs[0]
    apd90 = []
    for i, dvdt_idx in enumerate(peak_idxs[:-1]):
        v_range = v[dvdt_idx: min_v_idxs[i]]
        t_range = t[dvdt_idx: min_v_idxs[i]]
        amp = np.max(v_range) - np.min(v_range) 
        apd90_v = np.max(v_range) - amp*.9
        pk_idx = np.argmax(v_range)
        
        apd90_idx = np.argmin(np.abs(v_range[pk_idx:] - apd90_v)) + pk_idx
        apd90.append(t_range[apd90_idx] - t_range[0])

    apd90 = np.array(apd90)
    amplitudes = np.array([0, 0, 0])

    return [
        dt_peak,
        dt_ap1,
        dt_ap2,
        dt_mdp,
        dt_d,
        dt_up1,
        dt_up2,
        v_peak,
        v_ap1,
        v_ap2,
        v_mdp,
        v_d,
        v_up1,
        v_up2,
        dvdt_max,
        apd90,
        amplitudes
    ]

