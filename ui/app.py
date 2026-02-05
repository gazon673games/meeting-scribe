# --- File: D:\work\own\voice2textTest\ui\app.py ---
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundcard as sc
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QLineEdit,
    QDialog,
    QComboBox,
    QFormLayout,
    QDialogButtonBox,
)

from audio.engine import AudioEngine, AudioFormat
from audio.sources.wasapi_loopback import WasapiLoopbackSource
from audio.sources.microphone import MicrophoneSource

try:
    import soundfile as sf
except ImportError:
    sf = None


# ---------------- UI data ----------------

@dataclass
class SourceRow:
    name: str
    enabled: QCheckBox
    meter: QProgressBar
    status: QLabel
    delay_ms: QLineEdit


# ---------------- Writer ----------------

class WriterThread(threading.Thread):
    """
    Always drains engine output queue.
    Writes to WAV only when recording=True.
    """

    def __init__(self, out_q: "queue.Queue[np.ndarray]"):
        super().__init__(name="wav-writer", daemon=True)
        self._q = out_q
        self._stop = threading.Event()

        self._lock = threading.RLock()
        self._recording = False
        self._wav_file = None  # soundfile.SoundFile
        self._target_path: Optional[Path] = None

        self._last_error: Optional[str] = None
        self._written_blocks: int = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                rec = self._recording
                wf = self._wav_file

            if rec and wf is not None:
                try:
                    wf.write(frame)
                    with self._lock:
                        self._written_blocks += 1
                except Exception as e:
                    with self._lock:
                        self._last_error = f"{type(e).__name__}: {e}"
                        self._recording = False
                        try:
                            wf.close()
                        except Exception:
                            pass
                        self._wav_file = None
                        self._target_path = None

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.0)
        with self._lock:
            self._recording = False
            if self._wav_file is not None:
                try:
                    self._wav_file.close()
                except Exception:
                    pass
            self._wav_file = None
            self._target_path = None

    def is_recording(self) -> bool:
        with self._lock:
            return bool(self._recording)

    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def written_blocks(self) -> int:
        with self._lock:
            return int(self._written_blocks)

    def start_recording(self, path: Path, fmt: AudioFormat) -> None:
        if sf is None:
            raise RuntimeError("soundfile is not installed")

        with self._lock:
            if self._recording:
                return

            self._last_error = None
            self._written_blocks = 0

            path.parent.mkdir(parents=True, exist_ok=True)
            wf = sf.SoundFile(
                str(path),
                mode="w",
                samplerate=fmt.sample_rate,
                channels=fmt.channels,
                subtype="PCM_16",
            )
            self._wav_file = wf
            self._target_path = path
            self._recording = True

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            if self._wav_file is not None:
                try:
                    self._wav_file.close()
                except Exception:
                    pass
            self._wav_file = None
            self._target_path = None


# ---------------- Device picker ----------------

class DevicePickerDialog(QDialog):
    TYPE_LOOPBACK = "System audio (WASAPI loopback)"
    TYPE_MIC = "Microphone (input device)"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add device")
        self.resize(560, 160)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems([self.TYPE_LOOPBACK, self.TYPE_MIC])

        self.cmb_device = QComboBox()
        self.cmb_device.setEditable(False)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Source type:", self.cmb_type)
        form.addRow("Device:", self.cmb_device)
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.cmb_type.currentIndexChanged.connect(self._reload_devices)

        self._reload_devices()

    def _reload_devices(self) -> None:
        self.cmb_device.clear()
        t = self.cmb_type.currentText()

        if t == self.TYPE_LOOPBACK:
            # soundcard: loopback microphones represent speaker loopback
            devs = self._list_loopback_devices()
            if not devs:
                self.cmb_device.addItem("(no loopback devices found)", None)
                self.cmb_device.setEnabled(False)
            else:
                self.cmb_device.setEnabled(True)
                for label, token in devs:
                    self.cmb_device.addItem(label, token)

        else:
            devs = self._list_input_devices()
            if not devs:
                self.cmb_device.addItem("(no input devices found)", None)
                self.cmb_device.setEnabled(False)
            else:
                self.cmb_device.setEnabled(True)
                for label, token in devs:
                    self.cmb_device.addItem(label, token)

    @staticmethod
    def _list_loopback_devices() -> List[Tuple[str, str]]:
        """
        Returns list of (label, token) where token is a string we pass into WasapiLoopbackSource(device=token).
        We use the device name substring/token.
        """
        out: List[Tuple[str, str]] = []
        try:
            mics = sc.all_microphones(include_loopback=True)
        except Exception:
            mics = []

        # Filter to loopback-y entries. soundcard doesn't always flag them, so keep all but prioritize likely ones.
        for m in mics:
            name = getattr(m, "name", "")
            if not name:
                continue
            out.append((name, name))

        # Put default speaker loopback first
        try:
            sp = sc.default_speaker()
            if sp is not None:
                default_token = sp.name
                # soundcard uses get_microphone(sp.name, include_loopback=True) internally; token can be substring.
                out.sort(key=lambda x: 0 if default_token.lower() in x[0].lower() else 1)
        except Exception:
            pass

        # de-dup
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for label, token in out:
            k = label.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append((label, token))
        return uniq

    @staticmethod
    def _list_input_devices() -> List[Tuple[str, int]]:
        """
        Returns list of (label, device_index) for sounddevice InputStream(device=...).
        """
        out: List[Tuple[str, int]] = []
        try:
            devs = sd.query_devices()
        except Exception:
            devs = []

        for idx, d in enumerate(devs):
            try:
                if int(d.get("max_input_channels", 0)) <= 0:
                    continue
                name = str(d.get("name", f"device-{idx}"))
                host = str(d.get("hostapi", ""))
                sr = d.get("default_samplerate", None)
                ch = d.get("max_input_channels", None)
                label = f"[{idx}] {name} (in={ch}, sr={sr})"
                out.append((label, idx))
            except Exception:
                continue

        return out

    def selected(self) -> Tuple[str, object | None]:
        """
        Returns (type, token) where token:
          - for loopback: str device token (name substring)
          - for mic: int device index
        """
        t = self.cmb_type.currentText()
        token = self.cmb_device.currentData()
        return t, token


# ---------------- Main window ----------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice2TextTest — Audio Mixer")
        self.resize(1040, 420)

        self.project_root = Path(__file__).resolve().parents[1]

        self.fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)
        self.out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.engine = AudioEngine(format=self.fmt, output_queue=self.out_q)

        self.rows: dict[str, SourceRow] = {}

        self.writer = WriterThread(self.out_q)
        self.writer.start()

        self.output_name = "capture_mix.wav"

        root = QVBoxLayout(self)

        # --- Add device ---
        top = QHBoxLayout()
        self.btn_add = QPushButton("Add device…")
        top.addWidget(self.btn_add)
        top.addStretch(1)
        root.addLayout(top)

        # --- Auto-sync controls (optional) ---
        sync_row = QHBoxLayout()
        self.chk_autosync = QCheckBox("Auto-sync mic to desktop_audio (GCC-PHAT)")
        self.chk_autosync.setChecked(False)
        self.chk_autosync.setEnabled(False)  # will enable if both exist
        sync_row.addWidget(self.chk_autosync, 1)
        root.addLayout(sync_row)

        # --- Output filename ---
        outrow = QHBoxLayout()
        outrow.addWidget(QLabel("Output file (saved in project root):"))
        self.txt_output = QLineEdit(self.output_name)
        self.txt_output.setPlaceholderText("capture_mix.wav")
        outrow.addWidget(self.txt_output, 1)
        root.addLayout(outrow)

        # --- Recording controls (only two buttons) ---
        ctrl = QHBoxLayout()
        self.btn_rec = QPushButton("Start Recording")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_rec)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        # --- Sources meters group ---
        self.grp = QGroupBox("Sources")
        self.grp_layout = QVBoxLayout(self.grp)
        root.addWidget(self.grp)

        # --- Master meter ---
        master = QHBoxLayout()
        master.addWidget(QLabel("MASTER"))
        self.master_meter = QProgressBar()
        self.master_meter.setRange(0, 100)
        self.master_meter.setTextVisible(True)
        self.master_status = QLabel("idle")
        self.master_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        master.addWidget(self.master_meter, 1)
        master.addWidget(self.master_status)
        root.addLayout(master)

        # --- Drops line ---
        drops_row = QHBoxLayout()
        self.lbl_drops = QLabel("drops: 0")
        self.lbl_drops.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        drops_row.addWidget(self.lbl_drops, 1)
        root.addLayout(drops_row)

        # --- Status line ---
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("ready")
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_row.addWidget(self.lbl_status, 1)
        root.addLayout(status_row)

        # --- UI timer ---
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self._tick_ui)

        # Wiring
        self.btn_add.clicked.connect(self._add_device_dialog)
        self.txt_output.textChanged.connect(self._on_output_changed)

        self.btn_rec.clicked.connect(self._start_recording)
        self.btn_stop.clicked.connect(self._stop_all)

        self.chk_autosync.stateChanged.connect(self._on_autosync_changed)

        if sf is None:
            self._set_status("soundfile not installed: recording disabled (pip install soundfile)")
            self.btn_rec.setEnabled(False)

    # ---------------- helpers ----------------

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def _current_output_path(self) -> Path:
        name = (self.txt_output.text() or "").strip()
        if not name:
            name = "capture_mix.wav"
        name = Path(name).name
        if not name.lower().endswith(".wav"):
            name += ".wav"
        return self.project_root / name

    def _on_output_changed(self, _text: str) -> None:
        if self.writer.is_recording():
            self.txt_output.blockSignals(True)
            self.txt_output.setText(self.output_name)
            self.txt_output.blockSignals(False)
            return

        p = self._current_output_path()
        self.output_name = p.name
        if self.txt_output.text() != self.output_name:
            self.txt_output.blockSignals(True)
            self.txt_output.setText(self.output_name)
            self.txt_output.blockSignals(False)

    def _make_unique_name(self, base: str) -> str:
        if base not in self.rows:
            return base
        i = 2
        while f"{base}_{i}" in self.rows:
            i += 1
        return f"{base}_{i}"

    def _maybe_enable_autosync_checkbox(self) -> None:
        # Keep your current UX assumption: autosync only makes sense for desktop_audio + mic
        have_desktop = "desktop_audio" in self.rows
        have_mic = "mic" in self.rows
        self.chk_autosync.setEnabled(have_desktop and have_mic)

    def _add_row(self, name: str) -> None:
        if name in self.rows:
            return

        row = QHBoxLayout()

        cb = QCheckBox(name)
        cb.setChecked(True)

        delay = QLineEdit("0")
        delay.setFixedWidth(70)
        delay.setPlaceholderText("ms")
        delay.setToolTip("Delay compensation in milliseconds (>=0).")

        meter = QProgressBar()
        meter.setRange(0, 100)
        meter.setTextVisible(False)

        status = QLabel("idle")
        status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        cb.stateChanged.connect(lambda st, n=name: self.engine.set_source_enabled(n, st == Qt.Checked))
        delay.editingFinished.connect(lambda n=name: self._apply_delay_from_ui(n))

        row.addWidget(cb, 0)
        row.addWidget(QLabel("delay ms:"), 0)
        row.addWidget(delay, 0)
        row.addWidget(meter, 1)
        row.addWidget(status, 0)

        self.grp_layout.addLayout(row)
        self.rows[name] = SourceRow(name=name, enabled=cb, meter=meter, status=status, delay_ms=delay)

        self._maybe_enable_autosync_checkbox()

    def _apply_delay_from_ui(self, name: str) -> None:
        r = self.rows.get(name)
        if r is None:
            return
        txt = (r.delay_ms.text() or "").strip().replace(",", ".")
        try:
            v = float(txt) if txt else 0.0
        except ValueError:
            v = 0.0
        if v < 0:
            v = 0.0
        r.delay_ms.setText(str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.2f}")
        self.engine.set_source_delay_ms(name, v)

    # ---------------- actions ----------------

    def _add_device_dialog(self) -> None:
        # Do not allow adding while recording; engine would be running and engine.add_source is forbidden.
        if self.writer.is_recording() or self.engine.is_running():
            self._set_status("Stop recording before adding devices.")
            return

        dlg = DevicePickerDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return

        typ, token = dlg.selected()
        if token is None:
            self._set_status("No device selected.")
            return

        try:
            if typ == DevicePickerDialog.TYPE_LOOPBACK:
                # token is str device substring/name
                base_name = "desktop_audio"
                name = self._make_unique_name(base_name)
                src = WasapiLoopbackSource(name=name, format=self.fmt, device=str(token))
                self.engine.add_source(src)
                self._add_row(name)
                self._set_status(f"Added loopback: {token} -> source '{name}'")

            else:
                # token is sounddevice input index
                base_name = "mic"
                name = self._make_unique_name(base_name)
                mic_fmt = AudioFormat(sample_rate=48000, channels=1, dtype="float32", blocksize=1024)
                src = MicrophoneSource(name=name, format=mic_fmt, device=int(token))
                self.engine.add_source(src)
                self._add_row(name)
                self._set_status(f"Added mic: {token} -> source '{name}'")

        except Exception as e:
            self._set_status(f"Failed to add device: {e}")

    def _on_autosync_changed(self, state: int) -> None:
        enabled = state == Qt.Checked
        if enabled:
            try:
                self.engine.enable_auto_sync(reference_source="desktop_audio", target_source="mic")
                self._set_status("auto-sync enabled (mic <- desktop_audio)")
            except Exception as e:
                self.chk_autosync.blockSignals(True)
                self.chk_autosync.setChecked(False)
                self.chk_autosync.blockSignals(False)
                self._set_status(f"auto-sync enable failed: {e}")
        else:
            self.engine.disable_auto_sync()
            self._set_status("auto-sync disabled")

    def _start_recording(self) -> None:
        if sf is None:
            self._set_status("Recording unavailable: install soundfile.")
            return
        if self.writer.is_recording():
            return
        if len(self.rows) == 0:
            self._set_status("Add at least one device first.")
            return

        # apply current delays before starting
        for n in list(self.rows.keys()):
            self._apply_delay_from_ui(n)

        # start engine if not running
        try:
            if not self.engine.is_running():
                self.engine.start()
        except Exception as e:
            self._set_status(f"Engine start failed: {e}")
            return

        # start recording
        out_path = self._current_output_path()
        self.output_name = out_path.name
        try:
            self.writer.start_recording(out_path, self.fmt)
        except Exception as e:
            self._set_status(f"Recording start failed: {e}")
            try:
                self.engine.stop()
            except Exception:
                pass
            return

        # UI state
        self.btn_rec.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_add.setEnabled(False)
        self.txt_output.setEnabled(False)
        self.chk_autosync.setEnabled(False)  # keep stable during recording
        self.ui_timer.start()

        self._set_status(f"recording -> {out_path}")

    def _stop_all(self) -> None:
        if not self.writer.is_recording() and not self.engine.is_running():
            self.btn_stop.setEnabled(False)
            self.btn_rec.setEnabled(sf is not None)
            self.btn_add.setEnabled(True)
            self.txt_output.setEnabled(True)
            self._maybe_enable_autosync_checkbox()
            return

        # stop recording first
        if self.writer.is_recording():
            self.writer.stop_recording()

        # stop engine
        if self.engine.is_running():
            try:
                self.engine.stop()
            except Exception as e:
                self._set_status(f"Engine stop error: {e}")

        # UI reset
        self.ui_timer.stop()

        self.btn_stop.setEnabled(False)
        self.btn_rec.setEnabled(sf is not None)
        self.btn_add.setEnabled(True)
        self.txt_output.setEnabled(True)
        self._maybe_enable_autosync_checkbox()

        # meters reset (cosmetic)
        self.master_meter.setValue(0)
        self.master_status.setText("stopped")
        self.lbl_drops.setText("drops: 0")
        for r in self.rows.values():
            r.meter.setValue(0)
            r.status.setText("stopped")

        err = self.writer.last_error()
        if err:
            self._set_status(f"stopped (writer error: {err})")
        else:
            self._set_status(f"stopped (blocks: {self.writer.written_blocks()})")

    def _tick_ui(self) -> None:
        meters = self.engine.get_meters()
        now = time.monotonic()

        # master
        mrms = float(meters["master"]["rms"])
        mlast = float(meters["master"]["last_ts"])
        self.master_meter.setValue(self._rms_to_pct(mrms))
        self.master_status.setText("active" if (now - mlast) < 0.5 and mrms > 1e-4 else "silence")

        # drops summary
        dropped_out = int(meters.get("drops", {}).get("dropped_out_blocks", 0))
        self.lbl_drops.setText(f"drops: out={dropped_out}")

        # sources
        srcs = meters.get("sources", {})
        for name, info in srcs.items():
            if name not in self.rows:
                self._add_row(name)
            r = self.rows[name]

            r.enabled.blockSignals(True)
            r.enabled.setChecked(bool(info.get("enabled", True)))
            r.enabled.blockSignals(False)

            rms = float(info.get("rms", 0.0))
            last_ts = float(info.get("last_ts", 0.0))
            buf_frames = int(info.get("buffer_frames", 0))
            dropped_in = int(info.get("dropped_in_frames", 0))
            missing_out = int(info.get("missing_out_frames", 0))
            delay_ms = float(info.get("delay_ms", 0.0))
            src_rate = int(info.get("src_rate", 0))

            # keep UI delay field in sync (unless user editing)
            if not r.delay_ms.hasFocus():
                r.delay_ms.setText(str(int(round(delay_ms))))

            r.meter.setValue(self._rms_to_pct(rms))
            active = (now - last_ts) < 0.5 and rms > 1e-4

            rate_warn = ""
            if src_rate and src_rate != self.fmt.sample_rate:
                rate_warn = f" SR={src_rate}!"

            state = "active" if active else "silence"
            r.status.setText(
                f"{state} buf={buf_frames} miss={missing_out} drop_in={dropped_in} delay={int(round(delay_ms))}ms{rate_warn}"
            )

    @staticmethod
    def _rms_to_pct(rms: float) -> int:
        x = float(rms)
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        pct = int((x ** 0.5) * 100.0)
        if pct < 0:
            return 0
        if pct > 100:
            return 100
        return pct

    def closeEvent(self, event) -> None:
        try:
            self._stop_all()
        finally:
            try:
                self.writer.stop()
            except Exception:
                pass
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
