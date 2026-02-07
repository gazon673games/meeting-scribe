# --- File: D:\work\own\voice2textTest\ui\app.py ---
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from PySide6.QtGui import QTextCursor
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
    QTextEdit,
)

from audio.engine import AudioEngine, AudioFormat
from audio.sources.wasapi_loopback import WasapiLoopbackSource
from audio.sources.microphone import MicrophoneSource

from asr.pipeline import ASRPipeline

try:
    import soundfile as sf
except ImportError:
    sf = None


@dataclass
class SourceRow:
    name: str
    enabled: QCheckBox
    meter: QProgressBar
    status: QLabel
    delay_ms: QLineEdit


class WriterThread(threading.Thread):
    """
    Critical: ALWAYS drains engine output queue to avoid queue.Full -> dropped_out_blocks.

    - If recording=True -> write WAV
    - else -> discard frames
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
        self._drained_blocks: int = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                self._drained_blocks += 1
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

    def drained_blocks(self) -> int:
        with self._lock:
            return int(self._drained_blocks)

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


class DevicePickerDialog(QDialog):
    TYPE_LOOPBACK = "System audio (WASAPI loopback)"
    TYPE_MIC = "Microphone (input device)"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add device")
        self.resize(720, 190)

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
        out: List[Tuple[str, str]] = []
        try:
            mics = sc.all_microphones(include_loopback=True)
        except Exception:
            mics = []

        for m in mics:
            name = getattr(m, "name", "")
            if name:
                out.append((name, name))

        try:
            sp = sc.default_speaker()
            if sp is not None:
                default_token = sp.name.lower()
                out.sort(key=lambda x: 0 if default_token in x[0].lower() else 1)
        except Exception:
            pass

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
                sr = d.get("default_samplerate", None)
                ch = d.get("max_input_channels", None)
                label = f"[{idx}] {name} (in={ch}, sr={sr})"
                out.append((label, idx))
            except Exception:
                continue
        return out

    def selected(self) -> Tuple[str, object | None]:
        t = self.cmb_type.currentText()
        token = self.cmb_device.currentData()
        return t, token


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice2TextTest — Audio Mixer + ASR")
        self.resize(1120, 720)  # taller for transcript

        self.project_root = Path(__file__).resolve().parents[1]
        self.fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)

        self.out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.tap_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)

        # UI events from ASR
        self.asr_ui_q: "queue.Queue[dict]" = queue.Queue(maxsize=400)

        self.engine = AudioEngine(format=self.fmt, output_queue=self.out_q, tap_queue=self.tap_q)

        self.rows: dict[str, SourceRow] = {}

        self.writer = WriterThread(self.out_q)
        self.writer.start()

        self.asr: Optional[ASRPipeline] = None
        self.asr_running: bool = False

        self.output_name = "capture_mix.wav"

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_add = QPushButton("Add device…")
        top.addWidget(self.btn_add)
        top.addStretch(1)
        root.addLayout(top)

        asr_row = QHBoxLayout()
        self.chk_asr = QCheckBox("Enable ASR (Russian)")
        self.chk_asr.setChecked(True)
        asr_row.addWidget(self.chk_asr)

        asr_row.addWidget(QLabel("Mode:"))
        self.cmb_asr_mode = QComboBox()
        self.cmb_asr_mode.addItems(["MIX (master)", "SPLIT (all sources)"])
        self.cmb_asr_mode.setCurrentIndex(1)
        asr_row.addWidget(self.cmb_asr_mode)

        asr_row.addWidget(QLabel("Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["large-v3", "medium", "small"])
        self.cmb_model.setCurrentText("medium")
        asr_row.addWidget(self.cmb_model)

        asr_row.addStretch(1)
        root.addLayout(asr_row)

        wav_row = QHBoxLayout()
        self.chk_wav = QCheckBox("Write WAV (master mix)")
        self.chk_wav.setChecked(False)
        wav_row.addWidget(self.chk_wav)

        wav_row.addWidget(QLabel("Output file (project root):"))
        self.txt_output = QLineEdit(self.output_name)
        self.txt_output.setPlaceholderText("capture_mix.wav")
        wav_row.addWidget(self.txt_output, 1)
        root.addLayout(wav_row)

        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)

        # clear transcript
        self.btn_clear = QPushButton("Clear transcript")
        ctrl.addWidget(self.btn_clear)

        ctrl.addStretch(1)
        root.addLayout(ctrl)

        self.grp = QGroupBox("Sources")
        self.grp_layout = QVBoxLayout(self.grp)
        root.addWidget(self.grp)

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

        drops_row = QHBoxLayout()
        self.lbl_drops = QLabel("drops: 0")
        self.lbl_drops.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        drops_row.addWidget(self.lbl_drops, 1)
        root.addLayout(drops_row)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("ready")
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_row.addWidget(self.lbl_status, 1)
        root.addLayout(status_row)

        # transcript box (chat-like)
        self.grp_tr = QGroupBox("Transcript")
        tr_layout = QVBoxLayout(self.grp_tr)
        self.txt_tr = QTextEdit()
        self.txt_tr.setReadOnly(True)
        self.txt_tr.setPlaceholderText("ASR output will appear here…")
        tr_layout.addWidget(self.txt_tr)
        root.addWidget(self.grp_tr, 1)

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self._tick_ui)

        # Wiring
        self.btn_add.clicked.connect(self._add_device_dialog)
        self.txt_output.textChanged.connect(self._on_output_changed)
        self.btn_start.clicked.connect(self._start_all)
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_clear.clicked.connect(self._clear_transcript)

        if sf is None:
            self.chk_wav.setEnabled(False)
            self.chk_wav.setChecked(False)

    # ---------------- transcript ----------------

    def _clear_transcript(self) -> None:
        self.txt_tr.clear()

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        try:
            lt = time.localtime(ts)
            return time.strftime("%H:%M:%S", lt)
        except Exception:
            return "??:??:??"

    def _append_transcript_line(self, line: str) -> None:
        max_chars = 200_000
        if self.txt_tr.document().characterCount() > max_chars:
            self.txt_tr.clear()
            self.txt_tr.append("[transcript cleared: too large]")

        self.txt_tr.append(line)
        self.txt_tr.moveCursor(QTextCursor.End)
        self.txt_tr.ensureCursorVisible()

    def _drain_asr_ui_events(self, limit: int = 50) -> None:
        n = 0
        while n < limit:
            try:
                ev = self.asr_ui_q.get_nowait()
            except queue.Empty:
                break
            n += 1

            typ = str(ev.get("type", ""))
            ts = float(ev.get("ts", time.time()))
            tss = self._fmt_ts(ts)

            if typ == "segment":
                text = (ev.get("text") or "").strip()
                if not text:
                    continue
                stream = str(ev.get("stream", ""))
                spk = str(ev.get("speaker", "") or "")
                if spk == "S?":
                    spk = ""
                spk_part = f" {spk}" if spk else ""
                self._append_transcript_line(f"[{tss}] {stream}{spk_part}: {text}")

            elif typ == "speaker_estimate":
                stream = str(ev.get("stream", ""))
                nsp = ev.get("n_speakers", None)
                win = ev.get("window_s", None)
                if nsp is not None:
                    if win is not None:
                        self._append_transcript_line(f"[{tss}] {stream}: ~{int(nsp)} speaker(s) in last {int(win)}s")
                    else:
                        self._append_transcript_line(f"[{tss}] {stream}: ~{int(nsp)} speaker(s)")

            elif typ == "asr_init_start":
                model = ev.get("model", "")
                device = ev.get("device", "")
                self._append_transcript_line(f"[{tss}] ASR init start ({model}, {device})")

            elif typ == "segment_ready":
                stream = ev.get("stream", "")
                samples = ev.get("samples", 0)
                dur_s = ev.get("dur_s", None)
                if dur_s is not None:
                    self._append_transcript_line(f"[{tss}] segment ready ({stream}, {float(dur_s):.2f}s)")
                else:
                    self._append_transcript_line(f"[{tss}] segment ready ({stream}, samples={samples})")

            elif typ == "error":
                where = ev.get("where", "")
                err = ev.get("error", "")
                self._append_transcript_line(f"[{tss}] ERROR {where}: {err}")

            elif typ == "asr_started":
                model = ev.get("model", "")
                mode = ev.get("mode", "")
                diar = ev.get("diarization_enabled", None)
                diar_s = "on" if diar else "off"
                self._append_transcript_line(f"[{tss}] ASR started (mode={mode}, model={model}, diar={diar_s})")

            elif typ == "asr_init_ok":
                model = ev.get("model", "")
                self._append_transcript_line(f"[{tss}] ASR init OK ({model})")

            elif typ == "asr_stopped":
                self._append_transcript_line(f"[{tss}] ASR stopped")

    # ---------------- helpers ----------------

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def _is_running(self) -> bool:
        return self.engine.is_running()

    def _current_output_path(self) -> Path:
        name = (self.txt_output.text() or "").strip()
        if not name:
            name = "capture_mix.wav"
        name = Path(name).name
        if not name.lower().endswith(".wav"):
            name += ".wav"
        return self.project_root / name

    def _on_output_changed(self, _text: str) -> None:
        if self._is_running() and self.chk_wav.isChecked():
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
        if self._is_running():
            self._set_status("Stop before adding devices.")
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
                name = self._make_unique_name("desktop_audio")
                src = WasapiLoopbackSource(name=name, format=self.fmt, device=str(token))
                self.engine.add_source(src)
                self._add_row(name)
                self._set_status(f"Added loopback -> source '{name}'")
            else:
                name = self._make_unique_name("mic")
                mic_fmt = AudioFormat(sample_rate=48000, channels=1, dtype="float32", blocksize=1024)
                src = MicrophoneSource(name=name, format=mic_fmt, device=int(token))
                self.engine.add_source(src)
                self._add_row(name)
                self._set_status(f"Added mic -> source '{name}'")
        except Exception as e:
            self._set_status(f"Failed to add device: {e}")

    def _start_all(self) -> None:
        if self._is_running():
            return
        if len(self.rows) == 0:
            self._set_status("Add at least one device first.")
            return

        # clear stale UI events
        while True:
            try:
                self.asr_ui_q.get_nowait()
            except queue.Empty:
                break

        for n in list(self.rows.keys()):
            self._apply_delay_from_ui(n)

        try:
            self.engine.start()
        except Exception as e:
            self._set_status(f"Engine start failed: {e}")
            return

        self.asr_running = False
        self.asr = None
        if self.chk_asr.isChecked():
            mode = "split" if self.cmb_asr_mode.currentIndex() == 1 else "mix"
            model_name = self.cmb_model.currentText().strip() or "medium"
            try:
                self.asr = ASRPipeline(
                    tap_queue=self.tap_q,
                    project_root=self.project_root,
                    language="ru",
                    mode=mode,
                    asr_model_name=model_name,
                    device="cuda",
                    compute_type="float16",
                    beam_size=5,

                    endpoint_silence_ms=650.0,
                    max_segment_s=7.0,
                    overlap_ms=200.0,

                    vad_energy_threshold=0.0055,
                    vad_hangover_ms=350,
                    vad_min_speech_ms=350,

                    diarization_enabled=True,
                    diar_backend="online",  # <--- ДОБАВИТЬ ЯВНО
                    diar_chunk_s=30.0,  # <--- ДОБАВИТЬ
                    diar_step_s=10.0,  # <--- ДОБАВИТЬ

                    # эти параметры имеют смысл ТОЛЬКО для online backend;
                    # если используешь pyannote, можешь их убрать чтобы не путали
                    diar_sim_threshold=0.78,
                    diar_min_segment_s=1.6,
                    diar_window_s=120.0,

                    ui_queue=self.asr_ui_q,
                )
                self.asr.start()
                self.asr_running = True
            except Exception as e:
                self._set_status(f"ASR start failed: {e}")

        if self.chk_wav.isChecked():
            if sf is None:
                self._set_status("WAV disabled: install soundfile.")
            else:
                out_path = self._current_output_path()
                self.output_name = out_path.name
                try:
                    self.writer.start_recording(out_path, self.fmt)
                except Exception as e:
                    self._set_status(f"WAV start failed: {e}")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.btn_add.setEnabled(False)
        self.chk_asr.setEnabled(False)
        self.cmb_asr_mode.setEnabled(False)
        self.cmb_model.setEnabled(False)

        self.chk_wav.setEnabled(False)
        self.txt_output.setEnabled(False)

        self.ui_timer.start()
        self._set_status(
            f"running: ASR={'on' if self.asr_running else 'off'}, WAV={'on' if self.writer.is_recording() else 'off'}"
        )

    def _stop_all(self) -> None:
        if self.writer.is_recording():
            self.writer.stop_recording()

        if self.asr is not None:
            try:
                self.asr.stop()
            except Exception:
                pass
        self.asr = None
        self.asr_running = False

        if self.engine.is_running():
            try:
                self.engine.stop()
            except Exception as e:
                self._set_status(f"Engine stop error: {e}")

        self.ui_timer.stop()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.btn_add.setEnabled(True)
        self.chk_asr.setEnabled(True)
        self.cmb_asr_mode.setEnabled(True)
        self.cmb_model.setEnabled(True)

        self.chk_wav.setEnabled(sf is not None)
        self.txt_output.setEnabled(True)

        self.master_meter.setValue(0)
        self.master_status.setText("stopped")
        self.lbl_drops.setText("drops: 0")
        for r in self.rows.values():
            r.meter.setValue(0)
            r.status.setText("stopped")

        # final drain (show "ASR stopped" if it came)
        self._drain_asr_ui_events(limit=200)

        werr = self.writer.last_error()
        if werr:
            self._set_status(f"stopped (wav error: {werr})")
        else:
            self._set_status("stopped")

    def _tick_ui(self) -> None:
        # 1) update meters
        meters = self.engine.get_meters()
        now = time.monotonic()

        mrms = float(meters["master"]["rms"])
        mlast = float(meters["master"]["last_ts"])
        self.master_meter.setValue(self._rms_to_pct(mrms))
        self.master_status.setText("active" if (now - mlast) < 0.5 and mrms > 1e-4 else "silence")

        drops = meters.get("drops", {})
        dropped_out = int(drops.get("dropped_out_blocks", 0))
        dropped_tap = int(drops.get("dropped_tap_blocks", 0))
        drained = self.writer.drained_blocks()
        self.lbl_drops.setText(f"drops: out={dropped_out} tap={dropped_tap} drained={drained}")

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
            drop_in = int(info.get("dropped_in_frames", 0))
            miss_out = int(info.get("missing_out_frames", 0))
            delay_ms = float(info.get("delay_ms", 0.0))
            src_rate = int(info.get("src_rate", 0))

            if not r.delay_ms.hasFocus():
                r.delay_ms.setText(str(int(round(delay_ms))))

            r.meter.setValue(self._rms_to_pct(rms))
            active = (now - last_ts) < 0.5 and rms > 1e-4

            rate_warn = ""
            if src_rate and src_rate != self.fmt.sample_rate:
                rate_warn = f" SR={src_rate}!"

            state = "active" if active else "silence"
            r.status.setText(
                f"{state} buf={buf_frames} miss={miss_out} drop_in={drop_in} delay={int(round(delay_ms))}ms{rate_warn}"
            )

        # 2) drain ASR events to transcript (limit per tick to keep UI smooth)
        self._drain_asr_ui_events(limit=50)

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
