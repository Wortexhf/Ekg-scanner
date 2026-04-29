# 🫀 EKG Analyzer
A desktop application for real-time ECG signal visualization and automatic arrhythmia classification, built with Python and Tkinter.
<img width="947" height="1026" alt="image" src="https://github.com/user-attachments/assets/9576152f-e194-4628-8815-4f33f38485f6" />

---

## Features

- **Load ECG records** from the MIT-BIH Arrhythmia Database (records 100–234)
- **Bandpass filtering** — 5th-order Butterworth filter (0.5–40 Hz) to remove noise and baseline wander
- **R-peak detection** using adaptive thresholding with prominence and local window constraints
- **Beat classification** via a Random Forest model trained on MIT-BIH records 100–109:
  - Normal sinus rhythm
  - Premature Ventricular Contraction (PVC)
  - Atrial Premature Contraction (APC)
  - Left Bundle Branch Block (LBBB)`
  - Right Bundle Branch Block (RBBB)
  - Paced beat
- **Paginated signal view** — 60-second pages with Prev/Next scrub navigation
- **Live playback mode** — scrolls through the signal in a 6-second window at 0.1 s/step with beat labels
- **Stats panel** — R-peak count, mean heart rate, average amplitude
- **Rhythm breakdown** — color-coded beat-type counts in the left panel
---

## Project Structure

```
ekg_signals/
├── main.py              # Application entry point & UI (ECGApp)
├── dataset.py           # MIT-BIH record loading via wfdb (1 minute per load)
├── signal_processing.py # Bandpass filter & R-peak / feature extraction
├── classifier.py        # Random Forest training, model save/load & beat classification
├── visualizer.py        # Matplotlib-based ECG visualizer with paging & playback
├── requirements.txt     # Python dependencies
└── README.md
```

After the first run, two model files are created automatically:
- `ecg_model.pkl` — trained Random Forest classifier
- `label_encoder.pkl` — label encoder for class names

---

## Installation

**Python 3.9 or higher is required.**

```bash
# Clone the repository
git clone https://github.com/Wortexhf/Ekg-scanner
cd Ekg-scanner

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will automatically download training records from PhysioNet and train the model (~1–2 minutes depending on your connection). Subsequent launches use the saved model (`ecg_model.pkl`).

---

## Usage

```bash
python main.py
```

1. Select a **Record ID** (100–234) from the dropdown in the left panel
2. Click **⬇ Load** — the signal is downloaded (first 1 minute), filtered, and classified automatically; playback starts immediately
3. Use **◀ / ▶** buttons to scrub through the signal in 6-second steps (only when paused)
4. Click **▶ Play** / **⏸ Pause** to toggle live scrolling playback with beat annotations
5. Click **↺ Reset** to return to position 0

---

## How It Works

### Signal Loading
Only the **first minute** of each record is loaded to keep startup fast. The record is streamed on demand from PhysioNet via `wfdb` — no local dataset download required.

### Signal Processing
Raw ECG is filtered with a 5th-order Butterworth bandpass filter (0.5–40 Hz) in `signal_processing.py`. R-peaks are detected using `scipy.signal.find_peaks` with:
- **Adaptive height threshold** — mean + 0.3 × std of the positive signal portion
- **Minimum distance** — 0.4 s between peaks (max ~150 BPM)
- **Prominence** — 0.5 mV minimum to reject T-waves and noise
- **Local window** — 0.6 s for prominence calculation

### Feature Extraction (per beat)

| Feature | Description |
|---|---|
| RR pre | Interval before the beat (s) |
| RR post | Interval after the beat (s) |
| RR ratio | `RR_post / RR_pre` |
| Amplitude | Peak voltage at the R-peak (mV) |
| Segment std | Std deviation of ±100 ms window around the peak |
| Segment mean | Mean of ±100 ms window around the peak |

### Classification
A `RandomForestClassifier` (100 trees) is trained on records 100–109 from the MIT-BIH database using the 6 features above. Training uses an 80/20 stratified split and reports per-class precision, recall, and F1. The model is saved to `ecg_model.pkl` and loaded automatically on subsequent runs.

### Beat Colors

| Type | Color |
|---|---|
| Normal | 🟢 Green (`#a6e3a1`) |
| PVC | 🔴 Red (`#f38ba8`) |
| APC | 🟠 Orange (`#fab387`) |
| LBBB | 🔵 Blue (`#89b4fa`) |
| RBBB | 🟣 Purple (`#cba6f7`) |
| Paced | 🟡 Yellow (`#f9e2af`) |

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

`tkinter` is bundled with the standard Python installation — no separate install required.

---

## Data Source

ECG records are streamed on demand from [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) via the `wfdb` library. No local dataset download is required.

---

## License

MIT License. See `LICENSE` for details.