# --- File: D:\work\own\voice2textTest\ui\app.py ---
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PySide6.QtGui import QTextCursor
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
    QSizePolicy,
    QSplitter,  # NEW
)

import numpy as np
import sounddevice as sd
import soundcard as sc

from audio.engine import AudioEngine, AudioFormat
from audio.sources.wasapi_loopback import WasapiLoopbackSource
from audio.sources.microphone import MicrophoneSource
from asr.pipeline import ASRPipeline

try:
    import soundfile as sf
except ImportError:
    sf = None


CONFIG_VERSION = 2  # bumped


@dataclass
class SourceRow:
    name: str
    enabled: QCheckBox
    meter: QProgressBar
    status: QLabel
    delay_ms: QLineEdit


class WriterThread(threading.Thread):
    def __init__(self, out_q: "queue.Queue[np.ndarray]"):
        super().__init__(name="wav-writer", daemon=True)
        self._q = out_q
        self._stop = threading.Event()

        self._lock = threading.RLock()
        self._recording = False
        self._wav_file = None
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
    PROFILE_REALTIME = "Realtime"
    PROFILE_BALANCED = "Balanced"
    PROFILE_QUALITY = "Quality"
    PROFILE_CUSTOM = "Custom"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice2TextTest — Audio Mixer + ASR")
        self.resize(1180, 820)

        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / "config.json"

        self.fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)

        self.out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.tap_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        self.asr_ui_q: "queue.Queue[dict]" = queue.Queue(maxsize=600)

        self.engine = AudioEngine(format=self.fmt, output_queue=self.out_q, tap_queue=self.tap_q)
        self.rows: dict[str, SourceRow] = {}

        self.writer = WriterThread(self.out_q)
        self.writer.start()

        self.asr: Optional[ASRPipeline] = None
        self.asr_running: bool = False

        self.output_name = "capture_mix.wav"

        self._asr_overload_active: bool = False
        self._last_warn_ts: float = 0.0

        # ===== session metrics mirror (UI side) =====
        self._tap_dropped_total: int = 0
        self._seg_dropped_total: int = 0
        self._seg_skipped_total: int = 0
        self._avg_latency_s: float = 0.0
        self._p95_latency_s: float = 0.0
        self._lag_s: float = 0.0

        # ===== silence alert tracking =====
        self._silence_eps = 1e-4
        self._silence_alert_s = 15.0
        self._desktop_silence_since_mono: Optional[float] = None

        # ===== transcript file (optional) =====
        self._tr_fh = None
        self._tr_path: Optional[Path] = None

        # ===== offline pass (optional) =====
        self._offline_thread: Optional[threading.Thread] = None
        self._offline_running: bool = False
        self._offline_last_msg: str = ""

        # ===== UI =====
        root = QVBoxLayout(self)

        # NEW: splitter for resizing transcript vs controls
        self.splitter = QSplitter(Qt.Vertical)
        root.addWidget(self.splitter, 1)

        # -------- top pane --------
        top_pane = QWidget()
        top_layout = QVBoxLayout(top_pane)
        top_layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self.btn_add = QPushButton("Add device…")
        top.addWidget(self.btn_add)

        # NEW: long-run mode toggle (lighter UI)
        self.chk_longrun = QCheckBox("Long-run mode (lighter UI)")
        self.chk_longrun.setChecked(False)
        top.addWidget(self.chk_longrun)

        top.addStretch(1)
        top_layout.addLayout(top)

        # ===== ASR row =====
        asr_row = QHBoxLayout()
        self.chk_asr = QCheckBox("Enable ASR")
        self.chk_asr.setChecked(True)
        asr_row.addWidget(self.chk_asr)

        asr_row.addWidget(QLabel("Profile:"))
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems([self.PROFILE_REALTIME, self.PROFILE_BALANCED, self.PROFILE_QUALITY, self.PROFILE_CUSTOM])
        self.cmb_profile.setCurrentText(self.PROFILE_BALANCED)
        asr_row.addWidget(self.cmb_profile)

        asr_row.addWidget(QLabel("Lang:"))
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(["ru", "en", "auto"])
        self.cmb_lang.setCurrentText("ru")
        asr_row.addWidget(self.cmb_lang)

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
        top_layout.addLayout(asr_row)

        # ===== profile settings group =====
        self.grp_asr_cfg = QGroupBox("ASR settings (Profile)")
        cfg_layout = QVBoxLayout(self.grp_asr_cfg)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.cmb_compute = QComboBox()
        self.cmb_compute.addItems(["int8_float16", "float16", "int8", "int8_float32", "float32"])
        self.cmb_compute.setCurrentText("float16")

        self.txt_beam = QLineEdit("5")
        self.txt_endpoint = QLineEdit("650.0")
        self.txt_maxseg = QLineEdit("7.0")
        self.txt_overlap = QLineEdit("200.0")
        self.txt_vad_thr = QLineEdit("0.0055")

        self.cmb_overload_strategy = QComboBox()
        self.cmb_overload_strategy.addItems(["drop_old", "keep_all"])
        self.cmb_overload_strategy.setCurrentText("drop_old")

        self.txt_over_enter = QLineEdit("18")
        self.txt_over_exit = QLineEdit("6")
        self.txt_over_hard = QLineEdit("28")
        self.txt_over_beamcap = QLineEdit("2")
        self.txt_over_maxseg = QLineEdit("5.0")
        self.txt_over_overlap = QLineEdit("120.0")

        form.addRow("compute_type:", self.cmb_compute)
        form.addRow("beam_size:", self.txt_beam)
        form.addRow("endpoint_silence_ms:", self.txt_endpoint)
        form.addRow("max_segment_s:", self.txt_maxseg)
        form.addRow("overlap_ms:", self.txt_overlap)
        form.addRow("vad_energy_threshold:", self.txt_vad_thr)
        form.addRow("overload_strategy:", self.cmb_overload_strategy)
        form.addRow("overload_enter_qsize:", self.txt_over_enter)
        form.addRow("overload_exit_qsize:", self.txt_over_exit)
        form.addRow("overload_hard_qsize:", self.txt_over_hard)
        form.addRow("overload_beam_cap:", self.txt_over_beamcap)
        form.addRow("overload_max_segment_s:", self.txt_over_maxseg)
        form.addRow("overload_overlap_ms:", self.txt_over_overlap)

        cfg_layout.addLayout(form)
        top_layout.addWidget(self.grp_asr_cfg)

        # ===== Step 6 options row =====
        opt_row = QHBoxLayout()
        self.chk_offline_on_stop = QCheckBox("Offline pass on Stop (quality)")
        self.chk_offline_on_stop.setChecked(False)
        opt_row.addWidget(self.chk_offline_on_stop)

        self.chk_tr_to_file = QCheckBox("Also write realtime transcript to file")
        self.chk_tr_to_file.setChecked(False)
        opt_row.addWidget(self.chk_tr_to_file)

        opt_row.addStretch(1)
        top_layout.addLayout(opt_row)

        # ===== WAV row =====
        wav_row = QHBoxLayout()
        self.chk_wav = QCheckBox("Write WAV (master mix)")
        self.chk_wav.setChecked(False)
        wav_row.addWidget(self.chk_wav)

        wav_row.addWidget(QLabel("Output file (project root):"))
        self.txt_output = QLineEdit(self.output_name)
        self.txt_output.setPlaceholderText("capture_mix.wav")
        wav_row.addWidget(self.txt_output, 1)
        top_layout.addLayout(wav_row)

        # ===== control row =====
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Clear transcript")
        ctrl.addWidget(self.btn_clear)

        ctrl.addStretch(1)
        top_layout.addLayout(ctrl)

        # ===== sources group =====
        self.grp = QGroupBox("Sources")
        self.grp_layout = QVBoxLayout(self.grp)
        top_layout.addWidget(self.grp)

        master = QHBoxLayout()
        master.addWidget(QLabel("MASTER"))
        self.master_meter = QProgressBar()
        self.master_meter.setRange(0, 100)
        self.master_meter.setTextVisible(True)
        self.master_status = QLabel("idle")
        self.master_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        master.addWidget(self.master_meter, 1)
        master.addWidget(self.master_status)
        top_layout.addLayout(master)

        drops_row = QHBoxLayout()
        self.lbl_drops = QLabel("drops: 0")
        self.lbl_drops.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        drops_row.addWidget(self.lbl_drops, 1)
        top_layout.addLayout(drops_row)

        comp_row = QHBoxLayout()
        self.lbl_completeness = QLabel("Completeness: OK")
        self.lbl_completeness.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        comp_row.addWidget(self.lbl_completeness, 1)
        top_layout.addLayout(comp_row)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("ready")
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_row.addWidget(self.lbl_status, 1)
        top_layout.addLayout(status_row)

        self.splitter.addWidget(top_pane)

        # -------- transcript pane (resizable) --------
        self.grp_tr = QGroupBox("Transcript")
        tr_layout = QVBoxLayout(self.grp_tr)
        self.txt_tr = QTextEdit()
        self.txt_tr.setReadOnly(True)
        self.txt_tr.setPlaceholderText("ASR output will appear here…")
        self.txt_tr.setLineWrapMode(QTextEdit.WidgetWidth)  # easier reading
        tr_layout.addWidget(self.txt_tr)
        self.splitter.addWidget(self.grp_tr)

        # make transcript pane grow
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        try:
            self.splitter.setSizes([520, 300])
        except Exception:
            pass

        # ===== timers =====
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(120)  # will be adjusted by long-run mode
        self.ui_timer.timeout.connect(self._tick_ui)

        # autosave debounce
        self._cfg_dirty = False
        self._cfg_save_timer = QTimer(self)
        self._cfg_save_timer.setInterval(350)
        self._cfg_save_timer.timeout.connect(self._flush_config_if_dirty)

        # ===== signals =====
        self.btn_add.clicked.connect(self._add_device_dialog)
        self.txt_output.textChanged.connect(self._on_output_changed)
        self.btn_start.clicked.connect(self._start_all)
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_clear.clicked.connect(self._clear_transcript)

        self.cmb_profile.currentIndexChanged.connect(self._on_profile_changed)

        self.chk_longrun.stateChanged.connect(lambda _s: self._apply_ui_mode_settings())
        self.chk_tr_to_file.stateChanged.connect(lambda _s: self._mark_config_dirty())
        self.chk_offline_on_stop.stateChanged.connect(lambda _s: self._mark_config_dirty())

        # config widgets -> autosave
        for w in [
            self.chk_asr,
            self.cmb_lang,
            self.cmb_asr_mode,
            self.cmb_model,
            self.cmb_compute,
            self.txt_beam,
            self.txt_endpoint,
            self.txt_maxseg,
            self.txt_overlap,
            self.txt_vad_thr,
            self.cmb_overload_strategy,
            self.txt_over_enter,
            self.txt_over_exit,
            self.txt_over_hard,
            self.txt_over_beamcap,
            self.txt_over_maxseg,
            self.txt_over_overlap,
            self.chk_wav,
            self.txt_output,
            self.chk_longrun,
            self.chk_tr_to_file,
            self.chk_offline_on_stop,
        ]:
            self._wire_config_change(w)

        if sf is None:
            self.chk_wav.setEnabled(False)
            self.chk_wav.setChecked(False)

        # ===== load config + apply profile + apply UI mode =====
        self._load_config_into_ui()
        self._apply_profile_to_fields(self.cmb_profile.currentText(), force=True)
        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)
        self._apply_ui_mode_settings()

    # ---------------- config ----------------

    def _wire_config_change(self, w) -> None:
        try:
            if isinstance(w, QLineEdit):
                w.textChanged.connect(lambda _t: self._mark_config_dirty())
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(lambda _s: self._mark_config_dirty())
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(lambda _i: self._mark_config_dirty())
        except Exception:
            pass

    def _mark_config_dirty(self) -> None:
        self._cfg_dirty = True
        if not self._cfg_save_timer.isActive():
            self._cfg_save_timer.start()

    def _flush_config_if_dirty(self) -> None:
        if not self._cfg_dirty:
            self._cfg_save_timer.stop()
            return
        self._cfg_dirty = False
        self._cfg_save_timer.stop()
        try:
            cfg = self._build_config_from_ui()
            tmp = self.config_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.config_path)
        except Exception:
            pass

    def _build_config_from_ui(self) -> Dict[str, Any]:
        return {
            "version": CONFIG_VERSION,
            "ui": {
                "asr_enabled": bool(self.chk_asr.isChecked()),
                "lang": str(self.cmb_lang.currentText()),
                "asr_mode": int(self.cmb_asr_mode.currentIndex()),
                "model": str(self.cmb_model.currentText()),
                "profile": str(self.cmb_profile.currentText()),
                "wav_enabled": bool(self.chk_wav.isChecked()),
                "output_file": str(self.txt_output.text() or "").strip(),
                "longrun": bool(self.chk_longrun.isChecked()),
                "realtime_transcript_to_file": bool(self.chk_tr_to_file.isChecked()),
                "offline_pass_on_stop": bool(self.chk_offline_on_stop.isChecked()),
            },
            "asr": {
                "compute_type": str(self.cmb_compute.currentText()),
                "beam_size": self._safe_int(self.txt_beam.text(), 5, 1, 20),
                "endpoint_silence_ms": self._safe_float(self.txt_endpoint.text(), 650.0, 50.0, 5000.0),
                "max_segment_s": self._safe_float(self.txt_maxseg.text(), 7.0, 1.0, 60.0),
                "overlap_ms": self._safe_float(self.txt_overlap.text(), 200.0, 0.0, 2000.0),
                "vad_energy_threshold": self._safe_float(self.txt_vad_thr.text(), 0.0055, 1e-5, 1.0),
                "overload_strategy": str(self.cmb_overload_strategy.currentText()),
                "overload_enter_qsize": self._safe_int(self.txt_over_enter.text(), 18, 1, 999),
                "overload_exit_qsize": self._safe_int(self.txt_over_exit.text(), 6, 1, 999),
                "overload_hard_qsize": self._safe_int(self.txt_over_hard.text(), 28, 1, 999),
                "overload_beam_cap": self._safe_int(self.txt_over_beamcap.text(), 2, 1, 20),
                "overload_max_segment_s": self._safe_float(self.txt_over_maxseg.text(), 5.0, 0.5, 60.0),
                "overload_overlap_ms": self._safe_float(self.txt_over_overlap.text(), 120.0, 0.0, 2000.0),
            },
        }

    def _load_config_into_ui(self) -> None:
        if not self.config_path.exists():
            return
        try:
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        ui = cfg.get("ui", {}) if isinstance(cfg, dict) else {}
        asr = cfg.get("asr", {}) if isinstance(cfg, dict) else {}

        try:
            if "asr_enabled" in ui:
                self.chk_asr.setChecked(bool(ui.get("asr_enabled")))
            if "lang" in ui and str(ui.get("lang")) in ("ru", "en", "auto"):
                self.cmb_lang.setCurrentText(str(ui.get("lang")))
            if "asr_mode" in ui:
                idx = int(ui.get("asr_mode", 1))
                self.cmb_asr_mode.setCurrentIndex(1 if idx == 1 else 0)
            if "model" in ui and str(ui.get("model")) in ("large-v3", "medium", "small"):
                self.cmb_model.setCurrentText(str(ui.get("model")))
            if "profile" in ui and str(ui.get("profile")) in (
                self.PROFILE_REALTIME, self.PROFILE_BALANCED, self.PROFILE_QUALITY, self.PROFILE_CUSTOM
            ):
                self.cmb_profile.setCurrentText(str(ui.get("profile")))
            if "wav_enabled" in ui and sf is not None:
                self.chk_wav.setChecked(bool(ui.get("wav_enabled")))
            if "output_file" in ui:
                val = str(ui.get("output_file") or "").strip()
                if val:
                    self.txt_output.setText(val)

            if "longrun" in ui:
                self.chk_longrun.setChecked(bool(ui.get("longrun")))
            if "realtime_transcript_to_file" in ui:
                self.chk_tr_to_file.setChecked(bool(ui.get("realtime_transcript_to_file")))
            if "offline_pass_on_stop" in ui:
                self.chk_offline_on_stop.setChecked(bool(ui.get("offline_pass_on_stop")))
        except Exception:
            pass

        try:
            if "compute_type" in asr:
                v = str(asr.get("compute_type"))
                if v:
                    self.cmb_compute.setCurrentText(v)
            if "beam_size" in asr:
                self.txt_beam.setText(str(int(asr.get("beam_size"))))
            if "endpoint_silence_ms" in asr:
                self.txt_endpoint.setText(str(float(asr.get("endpoint_silence_ms"))))
            if "max_segment_s" in asr:
                self.txt_maxseg.setText(str(float(asr.get("max_segment_s"))))
            if "overlap_ms" in asr:
                self.txt_overlap.setText(str(float(asr.get("overlap_ms"))))
            if "vad_energy_threshold" in asr:
                self.txt_vad_thr.setText(str(float(asr.get("vad_energy_threshold"))))
            if "overload_strategy" in asr:
                v = str(asr.get("overload_strategy")).strip().lower()
                self.cmb_overload_strategy.setCurrentText("keep_all" if v == "keep_all" else "drop_old")
            if "overload_enter_qsize" in asr:
                self.txt_over_enter.setText(str(int(asr.get("overload_enter_qsize"))))
            if "overload_exit_qsize" in asr:
                self.txt_over_exit.setText(str(int(asr.get("overload_exit_qsize"))))
            if "overload_hard_qsize" in asr:
                self.txt_over_hard.setText(str(int(asr.get("overload_hard_qsize"))))
            if "overload_beam_cap" in asr:
                self.txt_over_beamcap.setText(str(int(asr.get("overload_beam_cap"))))
            if "overload_max_segment_s" in asr:
                self.txt_over_maxseg.setText(str(float(asr.get("overload_max_segment_s"))))
            if "overload_overlap_ms" in asr:
                self.txt_over_overlap.setText(str(float(asr.get("overload_overlap_ms"))))
        except Exception:
            pass

    # ---------------- profiles ----------------

    def _profile_defaults(self, profile: str) -> Dict[str, Any]:
        p = (profile or "").strip().lower()

        if p == self.PROFILE_REALTIME.lower():
            return {
                "compute_type": "int8_float16",
                "beam_size": 2,
                "endpoint_silence_ms": 450.0,
                "max_segment_s": 5.0,
                "overlap_ms": 120.0,
                "vad_energy_threshold": 0.0055,
                "overload_strategy": "drop_old",
                "overload_enter_qsize": 14,
                "overload_exit_qsize": 5,
                "overload_hard_qsize": 22,
                "overload_beam_cap": 1,
                "overload_max_segment_s": 3.5,
                "overload_overlap_ms": 80.0,
            }

        if p == self.PROFILE_QUALITY.lower():
            return {
                "compute_type": "float16",
                "beam_size": 6,
                "endpoint_silence_ms": 900.0,
                "max_segment_s": 12.0,
                "overlap_ms": 320.0,
                "vad_energy_threshold": 0.0052,
                "overload_strategy": "keep_all",
                "overload_enter_qsize": 22,
                "overload_exit_qsize": 8,
                "overload_hard_qsize": 40,
                "overload_beam_cap": 3,
                "overload_max_segment_s": 6.0,
                "overload_overlap_ms": 160.0,
            }

        return {
            "compute_type": "float16",
            "beam_size": 5,
            "endpoint_silence_ms": 650.0,
            "max_segment_s": 7.0,
            "overlap_ms": 200.0,
            "vad_energy_threshold": 0.0055,
            "overload_strategy": "drop_old",
            "overload_enter_qsize": 18,
            "overload_exit_qsize": 6,
            "overload_hard_qsize": 28,
            "overload_beam_cap": 2,
            "overload_max_segment_s": 5.0,
            "overload_overlap_ms": 120.0,
        }

    def _apply_profile_to_fields(self, profile: str, *, force: bool = False) -> None:
        if (profile or "") != self.PROFILE_CUSTOM:
            d = self._profile_defaults(profile)
            self.cmb_compute.setCurrentText(str(d["compute_type"]))
            self.txt_beam.setText(str(int(d["beam_size"])))
            self.txt_endpoint.setText(str(float(d["endpoint_silence_ms"])))
            self.txt_maxseg.setText(str(float(d["max_segment_s"])))
            self.txt_overlap.setText(str(float(d["overlap_ms"])))
            self.txt_vad_thr.setText(str(float(d["vad_energy_threshold"])))

            self.cmb_overload_strategy.setCurrentText(str(d["overload_strategy"]))
            self.txt_over_enter.setText(str(int(d["overload_enter_qsize"])))
            self.txt_over_exit.setText(str(int(d["overload_exit_qsize"])))
            self.txt_over_hard.setText(str(int(d["overload_hard_qsize"])))
            self.txt_over_beamcap.setText(str(int(d["overload_beam_cap"])))
            self.txt_over_maxseg.setText(str(float(d["overload_max_segment_s"])))
            self.txt_over_overlap.setText(str(float(d["overload_overlap_ms"])))

            self._set_custom_enabled(False)
            self._mark_config_dirty()
            return

        self._set_custom_enabled(True)
        if force:
            self._mark_config_dirty()

    def _set_custom_enabled(self, enabled: bool) -> None:
        for w in [
            self.cmb_compute,
            self.txt_beam,
            self.txt_endpoint,
            self.txt_maxseg,
            self.txt_overlap,
            self.txt_vad_thr,
            self.cmb_overload_strategy,
            self.txt_over_enter,
            self.txt_over_exit,
            self.txt_over_hard,
            self.txt_over_beamcap,
            self.txt_over_maxseg,
            self.txt_over_overlap,
        ]:
            try:
                w.setEnabled(bool(enabled))
            except Exception:
                pass

    def _on_profile_changed(self) -> None:
        prof = self.cmb_profile.currentText()
        self._apply_profile_to_fields(prof, force=True)

    # ---------------- UI mode ----------------

    def _apply_ui_mode_settings(self) -> None:
        # Step 5: long-run mode => slower UI tick (200–300ms) and reduced transcript spam
        if bool(self.chk_longrun.isChecked()):
            self.ui_timer.setInterval(250)
        else:
            self.ui_timer.setInterval(120)

    # ---------------- transcript/UI helpers ----------------

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
        # Do not force-scroll unless user is already at bottom
        max_chars = 260_000
        if self.txt_tr.document().characterCount() > max_chars:
            self.txt_tr.clear()
            self.txt_tr.append("[transcript cleared: too large]")

        sb = self.txt_tr.verticalScrollBar()
        at_bottom = (sb.value() >= (sb.maximum() - 2))

        self.txt_tr.append(line)

        if at_bottom:
            sb.setValue(sb.maximum())

        # optional: also write to transcript file
        if self._tr_fh is not None:
            try:
                self._tr_fh.write(line + "\n")
            except Exception:
                pass

    def _warn_throttle(self, msg: str, *, min_interval_s: float = 1.2) -> None:
        now = time.time()
        if (now - float(self._last_warn_ts)) < float(min_interval_s):
            return
        self._last_warn_ts = now
        tss = self._fmt_ts(now)
        self._append_transcript_line(f"[{tss}] WARNING: {msg}")

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

    def _transcript_output_path(self) -> Path:
        out_dir = self.project_root / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        return out_dir / f"transcript_{ts}.txt"

    def _open_transcript_file_if_enabled(self) -> None:
        self._close_transcript_file()
        if not bool(self.chk_tr_to_file.isChecked()):
            return
        try:
            p = self._transcript_output_path()
            self._tr_path = p
            self._tr_fh = p.open("a", encoding="utf-8")
            self._tr_fh.write(f"# transcript start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._tr_fh.flush()
        except Exception:
            self._tr_fh = None
            self._tr_path = None

    def _close_transcript_file(self) -> None:
        try:
            if self._tr_fh is not None:
                try:
                    self._tr_fh.write(f"# transcript stop {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                except Exception:
                    pass
                try:
                    self._tr_fh.flush()
                except Exception:
                    pass
                try:
                    self._tr_fh.close()
                except Exception:
                    pass
        finally:
            self._tr_fh = None

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

    def _get_asr_lang(self) -> str:
        v = (self.cmb_lang.currentText() or "ru").strip().lower()
        if v not in ("ru", "en", "auto"):
            v = "ru"
        return v

    def _get_asr_prompt(self, lang: str) -> Optional[str]:
        if lang == "ru":
            return "Транскрибируй разговорную русскую речь. Сохраняй числа, имена, термины. Ставь пунктуацию."
        if lang == "en":
            return "Transcribe conversational English. Keep numbers, names, and technical terms. Add punctuation."
        return None

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

    # ---------------- ASR/UI event drain ----------------

    def _drain_asr_ui_events(self, limit: int = 120) -> None:
        # Step 5: in long-run mode, drain less per tick (less UI work)
        if bool(self.chk_longrun.isChecked()):
            limit = min(limit, 60)

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

            # Step 5: show only utterance + important warnings in transcript in long-run mode
            longrun = bool(self.chk_longrun.isChecked())

            if typ == "utterance":
                text = (ev.get("text") or "").strip()
                if not text:
                    continue
                stream = str(ev.get("stream", ""))
                overload = bool(ev.get("overload", False))
                if overload:
                    self._asr_overload_active = True
                self._append_transcript_line(f"[{tss}] {stream}: {text}")

            elif typ == "asr_overload":
                active = bool(ev.get("active", False))
                reason = str(ev.get("reason", ""))
                qsz = ev.get("seg_qsize", None)
                beam = ev.get("beam_cur", None)
                lag = ev.get("lag_s", None)
                if bool(active):
                    self._asr_overload_active = True
                    if not longrun:
                        self._append_transcript_line(f"[{tss}] OVERLOAD: {reason} q={qsz} beam={beam} lag={lag}")
                else:
                    self._asr_overload_active = False
                    if not longrun:
                        self._append_transcript_line(f"[{tss}] OVERLOAD OFF: {reason} q={qsz}")

            elif typ == "segment_dropped":
                stream = str(ev.get("stream", ""))
                reason = str(ev.get("reason", ""))
                qsz = ev.get("seg_qsize", None)
                self._warn_throttle(f"ASR dropped segment ({stream}) reason={reason} q={qsz}")

            elif typ == "segment_skipped_overload":
                cnt = int(ev.get("count", 0))
                qsz = ev.get("seg_qsize", None)
                self._warn_throttle(f"ASR skipped {cnt} old segments due to overload (q={qsz})")

            elif typ == "asr_metrics":
                self._seg_dropped_total = int(ev.get("seg_dropped_total", self._seg_dropped_total))
                self._seg_skipped_total = int(ev.get("seg_skipped_total", self._seg_skipped_total))
                self._avg_latency_s = float(ev.get("avg_latency_s", self._avg_latency_s))
                self._p95_latency_s = float(ev.get("p95_latency_s", self._p95_latency_s))
                self._lag_s = float(ev.get("lag_s", self._lag_s))

            elif typ == "segment":
                # never show raw segments in UI
                continue

            elif typ in ("asr_init_start", "asr_init_ok", "asr_started", "asr_stopped"):
                if longrun:
                    continue
                if typ == "asr_init_start":
                    model = ev.get("model", "")
                    device = ev.get("device", "")
                    self._append_transcript_line(f"[{tss}] ASR init start ({model}, {device})")
                elif typ == "asr_started":
                    model = ev.get("model", "")
                    mode = ev.get("mode", "")
                    lang = ev.get("language", "")
                    osx = ev.get("overload_strategy", "")
                    self._append_transcript_line(f"[{tss}] ASR started (lang={lang}, mode={mode}, model={model}, overload={osx})")
                elif typ == "asr_init_ok":
                    model = ev.get("model", "")
                    self._append_transcript_line(f"[{tss}] ASR init OK ({model})")
                elif typ == "asr_stopped":
                    self._append_transcript_line(f"[{tss}] ASR stopped")

            elif typ == "error":
                where = ev.get("where", "")
                err = ev.get("error", "")
                self._append_transcript_line(f"[{tss}] ERROR {where}: {err}")

            else:
                continue

    # ---------------- start/stop ----------------

    def _start_all(self) -> None:
        if self._is_running():
            return
        if len(self.rows) == 0:
            self._set_status("Add at least one device first.")
            return

        self._apply_ui_mode_settings()

        self._asr_overload_active = False
        self._last_warn_ts = 0.0

        self._tap_dropped_total = 0
        self._seg_dropped_total = 0
        self._seg_skipped_total = 0
        self._avg_latency_s = 0.0
        self._p95_latency_s = 0.0
        self._lag_s = 0.0

        self._desktop_silence_since_mono = None

        while True:
            try:
                self.asr_ui_q.get_nowait()
            except queue.Empty:
                break

        # apply delays from UI
        for n in list(self.rows.keys()):
            self._apply_delay_from_ui(n)

        enabled_sources: List[str] = [n for n, r in self.rows.items() if r.enabled.isChecked()]

        if self.chk_asr.isChecked():
            self.engine.set_tap_queue(self.tap_q)

            mode = "split" if self.cmb_asr_mode.currentIndex() == 1 else "mix"
            if mode == "mix":
                self.engine.set_tap_config(mode="mix", sources=None, drop_threshold=0.85)
            else:
                self.engine.set_tap_config(mode="sources", sources=enabled_sources, drop_threshold=0.85)
        else:
            self.engine.set_tap_queue(None)

        # Step 6: master WAV should be available for offline pass (best effort)
        if self.chk_offline_on_stop.isChecked() and sf is not None:
            self.chk_wav.setChecked(True)

        try:
            self.engine.start()
        except Exception as e:
            self._set_status(f"Engine start failed: {e}")
            return

        self.asr_running = False
        self.asr = None

        # open transcript file if enabled
        self._open_transcript_file_if_enabled()

        if self.chk_asr.isChecked():
            mode = "split" if self.cmb_asr_mode.currentIndex() == 1 else "mix"
            model_name = self.cmb_model.currentText().strip() or "medium"

            lang_ui = self._get_asr_lang()
            asr_lang = None if lang_ui == "auto" else lang_ui
            prompt = self._get_asr_prompt(lang_ui)

            compute_type = str(self.cmb_compute.currentText() or "float16")
            beam_size = self._safe_int(self.txt_beam.text(), 5, 1, 20)
            endpoint_silence_ms = self._safe_float(self.txt_endpoint.text(), 650.0, 50.0, 5000.0)
            max_segment_s = self._safe_float(self.txt_maxseg.text(), 7.0, 1.0, 60.0)
            overlap_ms = self._safe_float(self.txt_overlap.text(), 200.0, 0.0, 2000.0)
            vad_thr = self._safe_float(self.txt_vad_thr.text(), 0.0055, 1e-5, 1.0)

            overload_strategy = str(self.cmb_overload_strategy.currentText() or "drop_old").strip().lower()
            overload_enter = self._safe_int(self.txt_over_enter.text(), 18, 1, 999)
            overload_exit = self._safe_int(self.txt_over_exit.text(), 6, 1, 999)
            overload_hard = self._safe_int(self.txt_over_hard.text(), 28, 1, 999)
            overload_beamcap = self._safe_int(self.txt_over_beamcap.text(), 2, 1, 20)
            overload_maxseg = self._safe_float(self.txt_over_maxseg.text(), 5.0, 0.5, 60.0)
            overload_overlap = self._safe_float(self.txt_over_overlap.text(), 120.0, 0.0, 2000.0)

            try:
                self.asr = ASRPipeline(
                    tap_queue=self.tap_q,
                    project_root=self.project_root,
                    language=lang_ui,
                    mode=mode,
                    asr_model_name=model_name,
                    device="cuda",
                    compute_type=compute_type,
                    beam_size=beam_size,
                    endpoint_silence_ms=endpoint_silence_ms,
                    max_segment_s=max_segment_s,
                    overlap_ms=overlap_ms,
                    vad_energy_threshold=vad_thr,
                    vad_hangover_ms=350,
                    vad_min_speech_ms=350,

                    diarization_enabled=False,
                    log_speaker_labels=False,

                    overload_strategy="keep_all" if overload_strategy == "keep_all" else "drop_old",
                    overload_enter_qsize=overload_enter,
                    overload_exit_qsize=overload_exit,
                    overload_hard_drop_qsize=overload_hard,
                    overload_hold_s=2.5,
                    overload_beam_cap=overload_beamcap,
                    overload_overlap_ms=overload_overlap,
                    overload_max_segment_s=overload_maxseg,

                    utterance_enabled=True,
                    utterance_gap_s=0.85,
                    utterance_max_s=18.0,
                    utterance_flush_s=2.5,
                    log_max_bytes=25 * 1024 * 1024,
                    log_backup_count=5,
                    ui_queue=self.asr_ui_q,

                    asr_language=asr_lang,
                    asr_initial_prompt=prompt,

                    metrics_emit_interval_s=1.0,
                    metrics_latency_window=200,
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
        self.chk_longrun.setEnabled(False)

        self.cmb_profile.setEnabled(False)
        self.cmb_lang.setEnabled(False)
        self.cmb_asr_mode.setEnabled(False)
        self.cmb_model.setEnabled(False)
        self.grp_asr_cfg.setEnabled(False)

        self.chk_offline_on_stop.setEnabled(False)
        self.chk_tr_to_file.setEnabled(False)

        self.chk_wav.setEnabled(False)
        self.txt_output.setEnabled(False)

        self.ui_timer.start()
        wav_on = "on" if self.writer.is_recording() else "off"
        self._set_status(f"running: ASR={'on' if self.asr_running else 'off'}, masterWAV={wav_on}")

    def _run_offline_pass_async(self, wav_path: Path) -> None:
        if self._offline_running:
            return

        def _job():
            self._offline_running = True
            try:
                # Lazy import so app works even if module not present yet
                from asr.offline_runner import OfflineRunner  # type: ignore

                out_dir = self.project_root / "logs"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_txt = out_dir / f"transcript_offline_{ts}.txt"

                runner = OfflineRunner(project_root=self.project_root)
                runner.run_quality(
                    wav_path=wav_path,
                    out_path=out_txt,
                    lang=str(self.cmb_lang.currentText() or "ru"),
                    model=str(self.cmb_model.currentText() or "large-v3"),
                )
                self._offline_last_msg = f"offline transcript saved: {out_txt.name}"
            except Exception as e:
                self._offline_last_msg = f"offline pass failed: {type(e).__name__}: {e}"
            finally:
                self._offline_running = False

        self._offline_thread = threading.Thread(target=_job, name="offline-pass", daemon=True)
        self._offline_thread.start()

    def _stop_all(self) -> None:
        # capture wav path for offline pass (before disabling UI)
        wav_path = self._current_output_path()

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

        try:
            self.engine.set_tap_queue(None)
        except Exception:
            pass

        self.ui_timer.stop()

        # close transcript file
        self._close_transcript_file()

        # Step 6: offline pass on stop (quality)
        if bool(self.chk_offline_on_stop.isChecked()) and wav_path.exists():
            self._run_offline_pass_async(wav_path)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.btn_add.setEnabled(True)
        self.chk_asr.setEnabled(True)
        self.chk_longrun.setEnabled(True)

        self.cmb_profile.setEnabled(True)
        self.cmb_lang.setEnabled(True)
        self.cmb_asr_mode.setEnabled(True)
        self.cmb_model.setEnabled(True)
        self.grp_asr_cfg.setEnabled(True)
        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)

        self.chk_offline_on_stop.setEnabled(True)
        self.chk_tr_to_file.setEnabled(True)

        self.chk_wav.setEnabled(sf is not None)
        self.txt_output.setEnabled(True)

        self.master_meter.setValue(0)
        self.master_status.setText("stopped")
        self.lbl_drops.setText("drops: 0")
        for r in self.rows.values():
            r.meter.setValue(0)
            r.status.setText("stopped")

        self._drain_asr_ui_events(limit=300)

        werr = self.writer.last_error()
        if werr:
            self._set_status(f"stopped (wav error: {werr})")
        else:
            msg = "stopped"
            if self._offline_last_msg:
                msg += f" | {self._offline_last_msg}"
            self._set_status(msg)

        self._flush_config_if_dirty()

    # ---------------- tick UI ----------------

    def _tick_ui(self) -> None:
        meters = self.engine.get_meters()
        now_mono = time.monotonic()

        # master
        mrms = float(meters["master"]["rms"])
        mlast = float(meters["master"]["last_ts"])
        self.master_meter.setValue(self._rms_to_pct(mrms))
        self.master_status.setText("active" if (now_mono - mlast) < 0.6 and mrms > 1e-4 else "silence")

        drops = meters.get("drops", {})
        dropped_out = int(drops.get("dropped_out_blocks", 0))
        dropped_tap = int(drops.get("dropped_tap_blocks", 0))
        self._tap_dropped_total = dropped_tap

        drained = self.writer.drained_blocks()
        self.lbl_drops.setText(f"drops: out={dropped_out} tap={dropped_tap} drained={drained}")

        if dropped_out > 0 or dropped_tap > 0:
            self._warn_throttle(f"Engine drops detected: out={dropped_out} tap={dropped_tap}", min_interval_s=2.0)

        # sources
        srcs = meters.get("sources", {})
        desktop_any_active = False
        desktop_any_present = False

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
            active = (now_mono - last_ts) < 0.6 and rms > 1e-4

            rate_warn = ""
            if src_rate and src_rate != self.fmt.sample_rate:
                rate_warn = f" SR={src_rate}!"

            state = "active" if active else "silence"
            r.status.setText(f"{state} buf={buf_frames} miss={miss_out} drop_in={drop_in} delay={int(round(delay_ms))}ms{rate_warn}")

            if str(name).startswith("desktop_audio"):
                desktop_any_present = True
                if rms > float(self._silence_eps):
                    desktop_any_active = True

        # drain ASR events and update metrics
        self._drain_asr_ui_events(limit=140)

        # completeness label
        ok = (self._tap_dropped_total <= 0) and (self._seg_dropped_total <= 0) and (self._seg_skipped_total <= 0)
        status = "OK" if ok else "DROPS"
        self.lbl_completeness.setText(
            f"Completeness: {status} | tap_drop={self._tap_dropped_total} seg_drop={self._seg_dropped_total} "
            f"seg_skip={self._seg_skipped_total} | avg_lat={self._avg_latency_s:.2f}s p95={self._p95_latency_s:.2f}s "
            f"lag={self._lag_s:.2f}s"
        )

        # desktop silence alert
        if self._is_running() and desktop_any_present:
            if desktop_any_active:
                self._desktop_silence_since_mono = None
            else:
                if self._desktop_silence_since_mono is None:
                    self._desktop_silence_since_mono = now_mono
                else:
                    dur = float(now_mono - float(self._desktop_silence_since_mono))
                    if dur >= float(self._silence_alert_s):
                        self._warn_throttle(f"desktop_audio silence for {dur:.1f}s (rms<{self._silence_eps})", min_interval_s=4.0)

        if self._is_running() and self._asr_overload_active:
            self.lbl_status.setText("running (ASR overload: degraded mode)")

        # show offline pass status (if running)
        if self._offline_running:
            self.lbl_status.setText("running offline pass (quality) ...")

    # ---------------- utils ----------------

    @staticmethod
    def _rms_to_pct(rms: float) -> int:
        x = float(rms)
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        pct = int((x ** 0.5) * 100.0)
        return max(0, min(100, pct))

    @staticmethod
    def _safe_int(s: str, default: int, lo: int, hi: int) -> int:
        try:
            v = int(str(s).strip())
        except Exception:
            v = int(default)
        v = max(int(lo), min(int(hi), int(v)))
        return int(v)

    @staticmethod
    def _safe_float(s: str, default: float, lo: float, hi: float) -> float:
        try:
            v = float(str(s).strip().replace(",", "."))
        except Exception:
            v = float(default)
        v = max(float(lo), min(float(hi), float(v)))
        return float(v)

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
