import numpy as np
import joblib
import wfdb
from scipy.signal import butter, lfilter, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

TRAIN_RECORDS = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109']

LABEL_MAP = {
    'N': 'Normal',
    'V': 'PVC',         
    'A': 'APC',          
    'L': 'LBBB',        
    'R': 'RBBB',        
    '/': 'Paced',       
}

def filter_ecg(signal, fs, lowcut=0.5, highcut=40, order=5):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return lfilter(b, a, signal)

def extract_beat_features(signal, fs, peak_idx, all_peaks):
    """Витягує ознаки одного удару"""
    pos = np.searchsorted(all_peaks, peak_idx)
    
    rr_pre = (all_peaks[pos] - all_peaks[pos - 1]) / fs if pos > 0 else 0
    rr_post = (all_peaks[pos + 1] - all_peaks[pos]) / fs if pos < len(all_peaks) - 1 else rr_pre
    
    win = int(0.1 * fs)
    start = max(0, peak_idx - win)
    end = min(len(signal), peak_idx + win)
    segment = signal[start:end]
    
    amplitude = signal[peak_idx]
    segment_std = np.std(segment)
    segment_mean = np.mean(segment)
    
    return [rr_pre, rr_post, rr_post / rr_pre if rr_pre > 0 else 1,
            amplitude, segment_std, segment_mean]

def load_training_data():
    X, y = [], []
    
    for rec_id in TRAIN_RECORDS:
        print(f"Loading record {rec_id}...")
        try:
            record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
            ann = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
            
            signal = filter_ecg(record.p_signal[:, 0], record.fs)
            peaks, _ = find_peaks(signal, distance=int(0.6 * record.fs),
                                  height=np.mean(signal) * 1.5)
            
            for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
                if symbol not in LABEL_MAP:
                    continue
                
                # Знайди найближчий пік
                nearest = peaks[np.argmin(np.abs(peaks - sample))]
                features = extract_beat_features(signal, record.fs, nearest, peaks)
                
                X.append(features)
                y.append(LABEL_MAP[symbol])
                
        except Exception as e:
            print(f"Error loading {rec_id}: {e}")
            continue
    
    return np.array(X), np.array(y)

def train_model():
    print("Loading data...")
    X, y = load_training_data()
    
    print(f"Total samples: {len(X)}")
    print(f"Classes: {np.unique(y, return_counts=True)}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    joblib.dump(model, 'ecg_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("\nModel saved: ecg_model.pkl")
    
    return model, le

def load_model():
    model = joblib.load('ecg_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

def classify_signal(signal, fs, peaks):
    """Класифікує кожен удар в сигналі"""
    try:
        model, le = load_model()
    except FileNotFoundError:
        print("Model not found, training...")
        model, le = train_model()
    
    results = []
    for peak in peaks:
        features = extract_beat_features(signal, fs, peak, peaks)
        pred = model.predict([features])[0]
        label = le.inverse_transform([pred])[0]
        results.append((peak, label))
    
    return results

if __name__ == "__main__":
    train_model()