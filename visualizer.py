import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


class ECGVisualizer:
    def __init__(self, parent_frame):
        self.signal = None
        self.fs = None
        self.peaks = None
        self.classifications = None

        self.is_playing = False
        self.current_pos = 0

        self._draw_job = None       
        self._draw_pos = 0          
        self._draw_step = None    

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.patch.set_facecolor("#1e1e2e")
        self._style_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _style_ax(self):
        self.ax.set_facecolor("#181825")
        self.ax.tick_params(colors="#a6adc8")
        self.ax.spines[:].set_color("#313244")

    def _cancel_draw(self):
        if self._draw_job is not None:
            self.canvas.get_tk_widget().after_cancel(self._draw_job)
            self._draw_job = None


    def load(self, signal, fs, peaks, classifications):
        self.signal = signal
        self.fs = fs
        self.peaks = peaks
        self.classifications = dict(classifications)
        self.current_pos = 0
        self.is_playing = False
        self.draw_full_progressive()

    def draw_full_progressive(self):
        self._cancel_draw()

        total_frames = 80
        self._draw_step = max(1, len(self.signal) // total_frames)
        self._draw_pos = 0

        self.ax.clear()
        self._style_ax()

        t_full = np.arange(len(self.signal)) / self.fs
        self.ax.set_xlim(t_full[0], t_full[-1])
        self.ax.set_ylim(self.signal.min() - 0.1, self.signal.max() + 0.1)
        self.ax.set_xlabel("Time (s)", color="#a6adc8")
        self.ax.set_ylabel("Amplitude (mV)", color="#a6adc8")
        self.ax.set_title("ECG Signal", color="#cdd6f4")

        self._line, = self.ax.plot([], [], color="#89b4fa", linewidth=0.6)
        self._scatter = self.ax.scatter([], [], color="#f38ba8", s=15, zorder=5)

        self.fig.tight_layout()
        self.canvas.draw()

        self._draw_job = self.canvas.get_tk_widget().after(
            16, self._progressive_frame
        )

    def _progressive_frame(self):
        if self.signal is None:
            return

        end = min(self._draw_pos + self._draw_step, len(self.signal))
        t = np.arange(end) / self.fs

        self._line.set_data(t, self.signal[:end])

        visible_peaks = self.peaks[self.peaks < end]
        if len(visible_peaks) > 0:
            self._scatter.set_offsets(
                np.c_[visible_peaks / self.fs, self.signal[visible_peaks]]
            )

        self.canvas.draw_idle()

        self._draw_pos = end

        if end < len(self.signal):
            self._draw_job = self.canvas.get_tk_widget().after(
                16, self._progressive_frame
            )
        else:
            # Анімація завершена — додаємо легенду
            self._draw_job = None
            self.ax.legend(
                handles=[
                    plt.Line2D([0], [0], color="#89b4fa", linewidth=1.5, label="ECG"),
                    plt.scatter([], [], color="#f38ba8", s=15, label="R-peaks"),
                ],
                facecolor="#313244", labelcolor="#cdd6f4", fontsize=8,
            )
            self.canvas.draw_idle()

    def toggle_play(self, on_stop_callback=None):
        if self.is_playing:
            self.is_playing = False
        else:
            self._cancel_draw()          # зупиняємо прогресивний рендер
            self.is_playing = True
            threading.Thread(
                target=self._play_loop,
                args=(on_stop_callback,),
                daemon=True,
            ).start()

    def _play_loop(self, on_stop_callback=None):
        window = int(self.fs * 4)
        step   = int(self.fs * 0.1)

        while self.is_playing and self.current_pos + window < len(self.signal):
            start = self.current_pos
            end   = start + window
            sig_slice   = self.signal[start:end]
            t_slice     = np.arange(start, end) / self.fs
            peaks_in    = self.peaks[(self.peaks >= start) & (self.peaks < end)]

            self.canvas.get_tk_widget().after(
                0,
                lambda s=sig_slice, tm=t_slice, pk=peaks_in:
                    self._draw_window(s, tm, pk),
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
            self.ax.scatter(
                peaks / self.fs, self.signal[peaks],
                color="#f38ba8", s=30, zorder=5,
            )
            for p in peaks:
                label = self.classifications.get(p, "")
                color = BEAT_COLORS.get(label, "#f9e2af")
                self.ax.annotate(
                    label,
                    (p / self.fs, self.signal[p]),
                    textcoords="offset points", xytext=(0, 8),
                    color=color, fontsize=7, ha='center',
                )

        self.ax.set_xlabel("Time (s)", color="#a6adc8")
        self.ax.set_ylabel("Amplitude (mV)", color="#a6adc8")
        self.ax.set_title("ECG Signal — Live", color="#cdd6f4")
        self.fig.tight_layout()
        self.canvas.draw()

    def reset(self):
        self.is_playing = False
        self.current_pos = 0
        if self.signal is not None:
            self.draw_full_progressive()