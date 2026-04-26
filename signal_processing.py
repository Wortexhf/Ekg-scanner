import numpy as np
from scipy.signal import butter, lfilter, find_peaks


def filter_ecg(signal, fs, lowcut=0.5, highcut=40, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)


def extracted_features(signal, fs):
    positive = signal[signal > 0]
    threshold = np.mean(positive) + 0.3 * np.std(positive) if len(positive) > 0 else np.percentile(signal, 90)
    peaks, _ = find_peaks(
        signal,
        distance=int(0.4 * fs),      
        height=threshold,
        prominence=0.5,               
        wlen=int(fs * 0.6),           
    )

    rr_intervals = np.diff(peaks) / fs
    amplitudes = signal[peaks]

    return {
        'peaks_indices': peaks,
        'rr_intervals': rr_intervals,
        'amplitudes': amplitudes,
        'mean_hr': 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
    }