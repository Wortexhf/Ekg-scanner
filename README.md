# 🫀 EKG Analyzer

A desktop application for real-time ECG signal visualization and automatic arrhythmia classification, built with Python and Tkinter.

---

## Features

- **Load ECG records** from the MIT-BIH Arrhythmia Database (records 100–234)
- **Bandpass filtering** to remove noise and baseline wander
- **R-peak detection** using adaptive thresholding
- **Beat classification** via a Random Forest model trained on MIT-BIH data:
  - Normal sinus rhythm
  - Premature Ventricular Contraction (PVC)
  - Atrial Premature Contraction (APC)
  - Left Bundle Branch Block (LBBB)
  - Right Bundle Branch Block (RBBB)
  - Paced beat
- **Progressive signal rendering** — the ECG draws itself smoothly on load
- **Live playback mode** — scroll through the signal in real time with beat labels
- **Stats panel** — R-peak count, mean heart rate, average amplitude

---

## Project Structure

```
ekg_signals/
├── main.py              # Application entry point & UI
├── dataset.py           # MIT-BIH record loading (via wfdb)
├── signal_processing.py # Bandpass filter & feature extraction
├── classifier.py        # Model training & beat classification
├── visualizer.py        # Matplotlib-based ECG visualizer
├── requirements.txt     # Python dependencies
└── README.md
```

After the first run, two model files are created automatically:
- `ecg_model.pkl` — trained Random Forest classifier
- `label_encoder.pkl` — label encoder for class names

---

## Installation

**Python 3.9 or higher is recommended.**

```bash
# Clone the repository
git clone <your-repo-url>
cd ekg_signals

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will automatically download training records from PhysioNet and train the model (~1–2 minutes depending on your connection). Subsequent launches use the saved model.

---

## Usage

```bash
python main.py
```

1. Enter a **Record ID** (100–234) in the left panel
2. Click **⬇ Load** — the signal downloads, filters, and classifies automatically
3. The ECG draws itself progressively on the chart
4. Click **▶ Play** to start live scrolling playback with beat annotations
5. Click **⏸ Pause** to pause, **↺ Reset** to go back to the full view

---

## How It Works

### Signal Processing
Raw ECG is filtered with a 5th-order Butterworth bandpass filter (0.5–40 Hz) to remove baseline drift and high-frequency noise. R-peaks are detected using `scipy.find_peaks` with a dynamic height threshold and minimum distance constraint.

### Feature Extraction (per beat)
| Feature | Description |
|---|---|
| RR pre | Interval before the beat (s) |
| RR post | Interval after the beat (s) |
| RR ratio | `RR_post / RR_pre` |
| Amplitude | Peak voltage (mV) |
| Segment std | Std deviation of ±100 ms window |
| Segment mean | Mean of ±100 ms window |

### Classification
A `RandomForestClassifier` (100 trees) is trained on records 100–109 from the MIT-BIH database using the 6 features above. Training uses an 80/20 stratified split and reports per-class precision, recall, and F1.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `scipy` | Signal filtering & peak detection |
| `matplotlib` | ECG plotting |
| `wfdb` | MIT-BIH database access |
| `scikit-learn` | Random Forest classifier |
| `joblib` | Model serialization |

---

## Data Source

ECG records are streamed on demand from [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) via the `wfdb` library. No local dataset download is required.

---

## License

MIT License. See `LICENSE` for details.