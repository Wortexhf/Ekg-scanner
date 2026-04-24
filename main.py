import tkinter as tk
from tkinter import messagebox
from collections import Counter
import numpy as np
import threading

from dataset import ecg_record
from signal_processing import filter_ecg, extracted_features
from classifier import classify_signal
from visualizer import ECGVisualizer


class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EKG Analyzer")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e2e")
        self._build_ui()

    def _build_ui(self):
        left = tk.Frame(self.root, bg="#1e1e2e", width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(left, text="EKG Analyzer", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 16, "bold")).pack(pady=(10, 20))

        tk.Label(left, text="Record ID (100-234):", bg="#1e1e2e",
                 fg="#a6adc8", font=("Helvetica", 10)).pack(anchor="w")

        self.record_entry = tk.Entry(left, font=("Helvetica", 12), width=10,
                                     bg="#313244", fg="#cdd6f4",
                                     insertbackground="white", relief="flat")
        self.record_entry.insert(0, "100")
        self.record_entry.pack(pady=5, ipady=4)

        self.load_btn = tk.Button(left, text="⬇ Load", font=("Helvetica", 11),
                                   bg="#89b4fa", fg="#1e1e2e", relief="flat",
                                   cursor="hand2", command=self._load_record)
        self.load_btn.pack(fill=tk.X, pady=5)

        self.play_btn = tk.Button(left, text="▶ Play", font=("Helvetica", 11),
                                   bg="#a6e3a1", fg="#1e1e2e", relief="flat",
                                   cursor="hand2", command=self._toggle_play,
                                   state=tk.DISABLED)
        self.play_btn.pack(fill=tk.X, pady=5)

        self.reset_btn = tk.Button(left, text="↺ Reset", font=("Helvetica", 11),
                                    bg="#f38ba8", fg="#1e1e2e", relief="flat",
                                    cursor="hand2", command=self._reset,
                                    state=tk.DISABLED)
        self.reset_btn.pack(fill=tk.X, pady=5)

        tk.Label(left, text="─── Stats ───", bg="#1e1e2e",
                 fg="#6c7086", font=("Helvetica", 9)).pack(pady=(20, 5))

        self.stat_peaks = tk.Label(left, text="R-peaks: —", bg="#1e1e2e",
                                    fg="#cdd6f4", font=("Helvetica", 10))
        self.stat_peaks.pack(anchor="w")

        self.stat_hr = tk.Label(left, text="Heart rate: —", bg="#1e1e2e",
                                 fg="#cdd6f4", font=("Helvetica", 10))
        self.stat_hr.pack(anchor="w")

        self.stat_amp = tk.Label(left, text="Amplitude: —", bg="#1e1e2e",
                                  fg="#cdd6f4", font=("Helvetica", 10))
        self.stat_amp.pack(anchor="w")

        tk.Label(left, text="─── Rhythm ───", bg="#1e1e2e",
                 fg="#6c7086", font=("Helvetica", 9)).pack(pady=(20, 5))

        self.rhythm_frame = tk.Frame(left, bg="#1e1e2e")
        self.rhythm_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(left, textvariable=self.status_var, bg="#1e1e2e",
                 fg="#f9e2af", font=("Helvetica", 9),
                 wraplength=180).pack(pady=(20, 0), anchor="w")

        right = tk.Frame(self.root, bg="#1e1e2e")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        self.viz = ECGVisualizer(right)

    def _load_record(self):
        rec_id = self.record_entry.get().strip()
        if not rec_id:
            messagebox.showerror("Error", "Enter record ID!")
            return

        self.status_var.set("Loading...")
        self.load_btn.config(state=tk.DISABLED)

        def _worker():
            record, ann = ecg_record(rec_id)
            if record is None:
                self.root.after(0, lambda: self.status_var.set("Failed to load!"))
                self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
                return

            raw = record.p_signal[:, 0]
            fs = record.fs
            clean = filter_ecg(raw, fs)
            data = extracted_features(clean, fs)

            self.root.after(0, lambda: self.status_var.set("Classifying..."))
            classifications = classify_signal(clean, fs, data['peaks_indices'])

            self.root.after(0, lambda: self.viz.load(
                clean, fs, data['peaks_indices'], classifications))

            self.root.after(0, lambda: self.stat_peaks.config(
                text=f"R-peaks: {len(data['peaks_indices'])}"))
            self.root.after(0, lambda: self.stat_hr.config(
                text=f"Heart rate: {data['mean_hr']:.1f} BPM"))
            self.root.after(0, lambda: self.stat_amp.config(
                text=f"Amplitude: {np.mean(data['amplitudes']):.2f} mV"))

            self.root.after(0, lambda: self._update_rhythm(classifications))
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL, text="⏸ Pause"))
            self.root.after(0, lambda: self.reset_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set(f"Loaded: {rec_id}"))
            # запускаємо play тільки після того як всі root.after виконались
            self.root.after(100, self._autostart_play)

        threading.Thread(target=_worker, daemon=True).start()

    def _autostart_play(self):
        # малюємо всі pending події перед запуском
        self.root.update_idletasks()
        if not self.viz.is_playing:
            self.viz.toggle_play(on_stop_callback=lambda: self.play_btn.config(text="▶ Play"))

    def _update_rhythm(self, classifications):
        for w in self.rhythm_frame.winfo_children():
            w.destroy()

        colors = {
            'Normal': '#a6e3a1', 'PVC': '#f38ba8',
            'APC': '#fab387', 'LBBB': '#89b4fa',
            'RBBB': '#cba6f7', 'Paced': '#f9e2af'
        }
        counts = Counter(label for _, label in classifications)
        for label, count in counts.most_common():
            tk.Label(self.rhythm_frame,
                     text=f"● {label}: {count}",
                     bg="#1e1e2e", fg=colors.get(label, '#cdd6f4'),
                     font=("Helvetica", 10)).pack(anchor="w")

    def _toggle_play(self):
        if self.viz.is_playing:
            self.viz.toggle_play()
            self.play_btn.config(text="▶ Play")
        else:
            self.viz.toggle_play(on_stop_callback=lambda: self.play_btn.config(text="▶ Play"))
            self.play_btn.config(text="⏸ Pause")

    def _reset(self):
        self.viz.reset()
        self.play_btn.config(text="▶ Play")


if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()