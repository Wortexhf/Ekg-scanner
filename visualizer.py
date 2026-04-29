import matplotlib
matplotlib.rcParams["toolbar"] = "None"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import threading
import time



BEAT_COLORS = {
    'Normal': '#a6e3a1',
    'PVC':    '#f38ba8',
    'APC':    '#fab387',
    'LBBB':   '#89b4fa',
    'RBBB':   '#cba6f7',
    'Paced':  '#f9e2af',
}

PAGE_SEC = 60  


class ECGVisualizer:
    def __init__(self, parent_frame):
        self.signal = None
        self.fs = None
        self.peaks = None
        self.classifications = None
        self.page = 0
        self.is_playing = False
        self.play_pos = 0

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.patch.set_facecolor("#1e1e2e")
        self._style_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)





    def _style_ax(self):
        self.ax.set_facecolor("#181825")
        self.ax.tick_params(colors="#a6adc8")
        self.ax.spines[:].set_color("#313244")

    def _total_pages(self):
        if self.signal is None:
            return 1
        return max(1, int(np.ceil(len(self.signal) / self.fs / PAGE_SEC)))

    def _update_nav(self):
        pass  

    def _prev_page(self):
        self.page -= 1
        self._draw_page()

    def _next_page(self):
        self.page += 1
        self._draw_page()

    def load(self, signal, fs, peaks, classifications):
        self.is_playing = False
        self.signal = signal
        self.fs = fs
        self.peaks = peaks
        self.classifications = dict(classifications)
        self.page = 0
        self.play_pos = 0
        self._update_nav()
        self._draw_page()

    def _draw_page(self):
        start_i  = int(self.page * PAGE_SEC * self.fs)
        end_i    = min(int((self.page + 1) * PAGE_SEC * self.fs), len(self.signal))
        sig      = self.signal[start_i:end_i]
        t        = np.arange(start_i, end_i) / self.fs
        peaks_in = self.peaks[(self.peaks >= start_i) & (self.peaks < end_i)]

        self.ax.clear()
        self._style_ax()
        self.ax.plot(t, sig, color="#89b4fa", linewidth=0.8)

        groups = defaultdict(list)
        for p in peaks_in:
            groups[self.classifications.get(p, "Normal")].append(p)

        handles = [plt.Line2D([0], [0], color="#89b4fa", linewidth=1.5, label="ECG")]
        for label, idxs in groups.items():
            idxs  = np.array(idxs)
            color = BEAT_COLORS.get(label, "#f9e2af")
            self.ax.scatter(idxs / self.fs, self.signal[idxs],
                            color=color, s=25, zorder=5)
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=6, label=label))

        self.ax.set_xlim(t[0], t[-1])
        self.ax.set_ylim(self.signal.min() - 0.1, self.signal.max() + 0.1)
        self.ax.set_xlabel("Time (s)", color="#a6adc8")
        self.ax.set_ylabel("Amplitude (mV)", color="#a6adc8")
        self.ax.set_title("ECG Signal", color="#cdd6f4")
        self.ax.legend(handles=handles, facecolor="#313244",
                       labelcolor="#cdd6f4", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def toggle_play(self, on_stop_callback=None):
        if self.is_playing:
            self.is_playing = False
        else:
            self.is_playing = True
            threading.Thread(target=self._play_loop,
                             args=(on_stop_callback,), daemon=True).start()

    def _play_loop(self, on_stop_callback=None):
        window = int(self.fs * 6)
        step   = int(self.fs * 0.1)
        while self.is_playing and self.play_pos + window < len(self.signal):
            s  = self.play_pos
            e  = s + window
            pk = self.peaks[(self.peaks >= s) & (self.peaks < e)]
            self.canvas.get_tk_widget().after(
                0, lambda s=s, e=e, pk=pk: self._draw_window(s, e, pk))
            self.play_pos += step
            time.sleep(0.05)
        self.is_playing = False
        if on_stop_callback:
            self.canvas.get_tk_widget().after(0, on_stop_callback)

    def _draw_window(self, start, end, peaks):
        self.ax.clear()
        self._style_ax()
        t = np.arange(start, end) / self.fs
        self.ax.plot(t, self.signal[start:end], color="#89b4fa", linewidth=1.2)
        if len(peaks) > 0:
            groups = defaultdict(list)
            for p in peaks:
                groups[self.classifications.get(p, "Normal")].append(p)
            for label, idxs in groups.items():
                idxs  = np.array(idxs)
                color = BEAT_COLORS.get(label, "#f9e2af")
                self.ax.scatter(idxs / self.fs, self.signal[idxs],
                                color=color, s=30, zorder=5)
                for p in idxs:
                    self.ax.annotate(label, (p / self.fs, self.signal[p]),
                                     textcoords="offset points", xytext=(0, 8),
                                     color=color, fontsize=7, ha='center')
        self.ax.set_xlim(t[0], t[-1])
        self.ax.set_ylim(self.signal.min() - 0.1, self.signal.max() + 0.1)
        self.ax.set_xlabel("Time (s)", color="#a6adc8")
        self.ax.set_ylabel("Amplitude (mV)", color="#a6adc8")
        self.fig.tight_layout()
        self.canvas.draw()

    def reset(self):
        self.is_playing = False
        self.play_pos = 0
        self.page = 0
        if self.signal is not None:
            self._update_nav()
            self._draw_page()

    def scrub(self, direction):
        """Move play_pos by 6s forward or backward and redraw. Only when paused."""
        if self.signal is None or self.is_playing:
            return
        step = int(self.fs * 6)
        self.play_pos = max(0, min(self.play_pos + direction * step,
                                   len(self.signal) - int(self.fs * 6)))
        s = self.play_pos
        e = min(s + int(self.fs * 6), len(self.signal))
        pk = self.peaks[(self.peaks >= s) & (self.peaks < e)]
        self._draw_window(s, e, pk)