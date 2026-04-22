"""EP_utilshdf5_genEP.py

Creates new h5 files for each EP train.

"""

# Standard Libraries #
import numpy as np
import scipy.signal as sp_sig

# Local Packages #

# Functions #
def find_stim_pulse_times(signal, sample_freq, stim_freq, min_height, min_distance_tol):

    expected_ipi = (1/stim_freq) * sample_freq
    expected_ipi_tol = (expected_ipi*min_distance_tol, expected_ipi*(2-min_distance_tol))

    signal_absderiv = np.abs(np.diff(signal, axis=0)).max(axis=1)
    peak_window_inds, _ = sp_sig.find_peaks(
        signal_absderiv,
        height=min_height,
        distance=expected_ipi_tol[0]
    )

    ##
    if len(peak_window_inds) < 3:
        return np.array([])

    ##
    observed_ipi = np.diff(peak_window_inds)
    incorrect_ipi = ((observed_ipi > expected_ipi_tol[1]) | 
                     (observed_ipi < expected_ipi_tol[0]))

    if incorrect_ipi[0]:
        peak_window_inds = peak_window_inds[1:]

    if incorrect_ipi[-1]:
        peak_window_inds = peak_window_inds[:-1]


    ##
    observed_ipi = np.diff(peak_window_inds)
    incorrect_ipi = ((observed_ipi > expected_ipi_tol[1]) | 
                     (observed_ipi < expected_ipi_tol[0]))

    if np.median(incorrect_ipi):
        return np.array([])

    return peak_window_inds

"""End of code

"""
