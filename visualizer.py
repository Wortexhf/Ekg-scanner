import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time


class ECGVisualizer:
    def __init__(self, parent_frame):
        self.signal = None
        self.fs = None
        self.peaks = None
        self.classifications = None
        self.is_playing = False
        self.current_pos = 0

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.patch.set_facecolor("#1e1e2e")
        self._style_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _style_ax(self):
        self.ax.set_facecolor("#181825")
        self.ax.tick_params(colors="#a6adc8")
        self.ax.spines[:].set_color("#313244")

    def load(self, signal, fs, peaks, classifications):
        self.signal = signal
        self.fs = fs
        self.peaks = peaks
        self.classifications = dict(classifications)
        self.current_pos = 0
        self.draw_full()

    def draw_full(self):
        self.ax.clear()
        self._style_ax()

        t = np.arange(len(self.signal)) / self.fs
        self.ax.plot(t, self.signal, color="#89b4fa", linewidth=0.6, label="ECG")
        self.ax.scatter(self.peaks / self.fs, self.signal[self.peaks],
                        color="#f38ba8", s=15, zorder=5, label="R-peaks")

        self.ax.set_xlabel("Time (s)", color="#a6adc8")
        self.ax.set_ylabel("Amplitude (mV)", color="#a6adc8")
        self.ax.set_title("ECG Signal", color="#cdd6f4")
        self.ax.legend(facecolor="#313244", labelcolor="#cdd6f4", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def toggle_play(self, on_stop_callback=None):
        if self.is_playing:
            self.is_playing = False
        else:
            self.is_playing = True
            t = threading.Thread(
                target=self._play_loop,
                args=(on_stop_callback,),
                daemon=True
            )
            t.start()

    def _play_loop(self, on_stop_callback=None):
        window = int(self.fs * 4)
        step = int(self.fs * 0.1)

        while self.is_playing and self.current_pos + window < len(self.signal):
            start = self.current_pos
            end = start + window

            sig_slice = self.signal[start:end]
            t = np.arange(start, end) / self.fs
            peaks_in = self.peaks[(self.peaks >= start) & (self.peaks < end)]

            self.canvas.get_tk_widget().after(
                0, lambda s=sig_slice, tm=t, pk=peaks_in: self._draw_window(s, tm, pk)
            )

            self.current_pos += step
            time.sleep(0.05)

        self.is_playing = False
        if on_stop_callback:
            self.canvas.get_tk_widget().after(0, on_stop_callback)

    def _draw_window(self, sig, t, peaks):
        self.ax.clear()
        self._style_ax()

        self.ax.plot(t, sig, color="#89b4fa", linewidth=1.2)

        if len(peaks) > 0:
            self.ax.scatter(peaks / self.fs, self.signal[peaks],
                            color="#f38ba8", s=30, zorder=5)

            for p in peaks:
                label = self.classifications.get(p, "")
                self.ax.annotate(label, (p / self.fs, self.signal[p]),
                                 textcoords="offset points", xytext=(0, 8),
                                 color="#f9e2af", fontsize=7, ha='center')

        self.ax.set_xlabel("Time (s)", color="#a6adc8")
        self.ax.set_ylabel("Amplitude (mV)", color="#a6adc8")
        self.ax.set_title("ECG Signal — Live", color="#cdd6f4")
        self.fig.tight_layout()
        self.canvas.draw()

    def reset(self):
        self.is_playing = False
        self.current_pos = 0
        self.draw_full()