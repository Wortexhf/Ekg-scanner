from dataset import ecg_record
import numpy as np
from signal_proceesing import extracted_features, filter_ecg


record, ann = ecg_record("")

if __name__ == "main":
    raw_signal = record.p_signal[:, 0]
    fs = record.fs
    clean_signal = filter_ecg(raw_signal, fs) 
    data = extracted_features(clean_signal, fs) 
    print(f"R picks: {len(data['peaks_indices'])}")
    print(f"Impuls: {data['mean_hr']:.2f} BPM")
    print(f"Medium amplitude: {np.mean(data['amplitudes']):.2f} mV")
