import numpy as np
from scipy.signal import butter, lfilter, find_peaks


def filter_ecg(signal, fs, lowcut=0.5, highcut=40, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)


def extracted_features(signal, fs):
    # Adaptive threshold based on positive signal portion
    positive = signal[signal > 0]
    threshold = np.mean(positive) + 0.3 * np.std(positive) if len(positive) > 0 else np.percentile(signal, 90)
    peaks, _ = find_peaks(
        signal,
        distance=int(0.4 * fs),      # min 0.4s between peaks = max 150 BPM
        height=threshold,
        prominence=0.5,               # must stand out clearly from neighbors
        wlen=int(fs * 0.6),           # local window for prominence calculation
    )

    rr_intervals = np.diff(peaks) / fs
    amplitudes = signal[peaks]

    return {
        'peaks_indices': peaks,
        'rr_intervals': rr_intervals,
        'amplitudes': amplitudes,
        'mean_hr': 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
    }