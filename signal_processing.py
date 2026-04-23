import numpy as np
from scipy.signal import butter, lfilter, find_peaks

def filter_ecg(signal, fs, lowcut=0.5, highcut=40, order=5):
    nyquist = 0.5 * fs   
    low = lowcut / nyquist
    high = highcut / nyquist 
    b, a = butter(order, [low, high], btype='band') 
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def extracted_features(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(0.6 * fs), height=np.mean(signal) * 1.5)  # розпакуй tuple!
    rr_intervals = np.diff(peaks) / fs
    amplitudes = signal[peaks]
    features = {
        'peaks_indices': peaks,
        'rr_intervals': rr_intervals,
        'amplitudes': amplitudes,
        'mean_hr': 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    }
    return features