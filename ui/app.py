# --- File: D:\work\own\voice2textTest\ui\app.py ---
from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QLineEdit,
    QComboBox,
    QFormLayout,
    QTextEdit,
    QSizePolicy,
    QSplitter,
    QScrollArea,
)

from audio.engine import AudioEngine
from audio.types import AudioFormat
from application.asr_profiles import (
    PROFILE_BALANCED as ASR_PROFILE_BALANCED,
    PROFILE_CUSTOM as ASR_PROFILE_CUSTOM,
    PROFILE_QUALITY as ASR_PROFILE_QUALITY,
    PROFILE_REALTIME as ASR_PROFILE_REALTIME,
)
from application.asr_session import ASRRuntimeFactory
from application.audio_sources import AudioSourceFactory
from application.codex_use_case import CodexRequestUseCase
from application.device_catalog import DeviceCatalog
from application.offline_pass import OfflineAsrRunnerPort
from application.recording import WavRecorderFactory
from ui.asr_events_mixin import AsrEventsMixin
from ui.config_mixin import MainWindowConfigMixin
from ui.codex_integration import CodexIntegrationMixin
from ui.device_picker import DevicePickerDialog
from ui.session_mixin import SessionMixin
from ui.telemetry_mixin import TelemetryMixin
from ui.transcript_mixin import TranscriptMixin

# Optional resource telemetry
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


@dataclass
class SourceRow:
    name: str
    enabled: QCheckBox
    meter: QProgressBar
    status: QLabel
    delay_ms: QLineEdit


class MainWindow(
    SessionMixin,
    TelemetryMixin,
    AsrEventsMixin,
    TranscriptMixin,
    MainWindowConfigMixin,
    CodexIntegrationMixin,
    QWidget,
):
    background_event = Signal(dict)

    PROFILE_REALTIME = ASR_PROFILE_REALTIME
    PROFILE_BALANCED = ASR_PROFILE_BALANCED
    PROFILE_QUALITY = ASR_PROFILE_QUALITY
    PROFILE_CUSTOM = ASR_PROFILE_CUSTOM

    def __init__(
        self,
        *,
        asr_runtime_factory: ASRRuntimeFactory,
        audio_source_factory: AudioSourceFactory,
        device_catalog: DeviceCatalog,
        wav_recorder_factory: WavRecorderFactory,
        codex_request_use_case: CodexRequestUseCase,
        offline_asr_runner: OfflineAsrRunnerPort,
    ):
        super().__init__()
        self.setWindowTitle("Meeting Scribe — Audio Mixer + ASR")
        self.resize(1180, 820)

        if getattr(sys, "frozen", False):
            self.project_root = Path(sys.executable).resolve().parent
        else:
            self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / "config.json"

        self.fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)

        self.out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.tap_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        self.asr_ui_q: "queue.Queue[dict]" = queue.Queue(maxsize=600)

        self.engine = AudioEngine(format=self.fmt, output_queue=self.out_q, tap_queue=self.tap_q)
        self.rows: dict[str, SourceRow] = {}
        self.source_objs: Dict[str, Any] = {}

        self.asr_runtime_factory = asr_runtime_factory
        self.audio_source_factory = audio_source_factory
        self.device_catalog = device_catalog
        self.wav_recorder_factory = wav_recorder_factory
        self.codex_request_use_case = codex_request_use_case
        self.offline_asr_runner = offline_asr_runner

        self.writer = self.wav_recorder_factory.create(self.out_q)
        self.writer.start()

        self.asr: Any = None
        self.asr_running: bool = False

        self.output_name = "capture_mix.wav"

        self._asr_overload_active: bool = False
        self._last_warn_ts: float = 0.0

        # session metrics mirror (UI side)
        self._tap_dropped_total: int = 0
        self._seg_dropped_total: int = 0
        self._seg_skipped_total: int = 0
        self._avg_latency_s: float = 0.0
        self._p95_latency_s: float = 0.0
        self._lag_s: float = 0.0

        # silence alert tracking
        self._silence_eps = 1e-4
        self._silence_alert_s = 15.0
        self._desktop_silence_since_mono: float | None = None

        # UI modes
        self._long_run_mode: bool = False
        self._ui_interval_normal_ms: int = 120
        self._ui_interval_long_ms: int = 260

        # transcript file logging (optional)
        self._rt_tr_to_file: bool = False
        self._rt_tr_path: Path | None = None
        self._rt_tr_fh = None

        # human-readable transcript logging (always on during ASR session)
        self._human_log_path: Path | None = None
        self._human_log_fh = None

        self._init_codex_state()
        self._closing: bool = False
        self._offline_pass_active: bool = False
        self._offline_thread: Any = None
        self._asr_stop_active: bool = False
        self._asr_stop_thread: Any = None
        self.background_event.connect(self._handle_background_event)

        # resource telemetry (optional)
        self._proc = psutil.Process() if psutil is not None else None
        self._last_cpu_poll_mono: float = 0.0
        self._cpu_pct: float = 0.0
        self._rss_mb: float = 0.0

        # ===================== UI ROOT =====================
        outer = QVBoxLayout(self)

        # Scroll area so you can scroll the whole window when ASR settings are expanded
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)

        content = QWidget()
        self.scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)

        # top row
        top = QHBoxLayout()
        self.btn_add = QPushButton("Add device…")
        top.addWidget(self.btn_add)

        self.chk_longrun = QCheckBox("Long-run mode (lighter UI)")
        self.chk_longrun.setChecked(False)
        top.addWidget(self.chk_longrun)

        top.addStretch(1)
        root.addLayout(top)

        # ASR row
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
        root.addLayout(asr_row)

        # ASR settings: collapsible header + body
        hdr = QHBoxLayout()
        self.btn_asr_toggle = QPushButton("Hide ASR settings")
        self.btn_asr_toggle.setCheckable(True)
        self.btn_asr_toggle.setChecked(True)  # checked => expanded
        hdr.addWidget(self.btn_asr_toggle)

        hdr.addStretch(1)
        root.addLayout(hdr)

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
        root.addWidget(self.grp_asr_cfg)

        # Step 6 / transcript options row
        opt_row = QHBoxLayout()
        self.chk_offline_on_stop = QCheckBox("Offline pass on Stop (quality)")
        self.chk_rt_transcript_file = QCheckBox("Also write realtime transcript to file")
        opt_row.addWidget(self.chk_offline_on_stop)
        opt_row.addWidget(self.chk_rt_transcript_file)
        opt_row.addStretch(1)
        root.addLayout(opt_row)

        # WAV row
        wav_row = QHBoxLayout()
        self.chk_wav = QCheckBox("Write WAV (master mix)")
        self.chk_wav.setChecked(False)
        wav_row.addWidget(self.chk_wav)

        wav_row.addWidget(QLabel("Output file (project root):"))
        self.txt_output = QLineEdit(self.output_name)
        self.txt_output.setPlaceholderText("capture_mix.wav")
        wav_row.addWidget(self.txt_output, 1)
        root.addLayout(wav_row)

        # control row
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Clear transcript")
        ctrl.addWidget(self.btn_clear)

        ctrl.addStretch(1)
        root.addLayout(ctrl)

        # Sources group
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

        # drops/completeness + resource usage
        drops_row = QHBoxLayout()
        self.lbl_drops = QLabel("drops: 0")
        self.lbl_drops.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        drops_row.addWidget(self.lbl_drops, 2)

        self.lbl_resources = QLabel("resources: n/a")
        self.lbl_resources.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        drops_row.addWidget(self.lbl_resources, 1)

        root.addLayout(drops_row)

        comp_row = QHBoxLayout()
        self.lbl_completeness = QLabel("Completeness: OK")
        self.lbl_completeness.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        comp_row.addWidget(self.lbl_completeness, 1)
        root.addLayout(comp_row)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("ready")
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_row.addWidget(self.lbl_status, 1)
        root.addLayout(status_row)

        self._build_codex_header(root)

        # Splitter: transcript should be resizable vertically without breaking scrolling
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)

        # transcript panel
        self.grp_tr = QGroupBox("Transcript (utterances + important events)")
        tr_layout = QVBoxLayout(self.grp_tr)
        self.txt_tr = QTextEdit()
        self.txt_tr.setReadOnly(True)
        self.txt_tr.setPlaceholderText("ASR output will appear here…")
        self.txt_tr.setLineWrapMode(QTextEdit.WidgetWidth)
        self.txt_tr.document().setMaximumBlockCount(2500)

        # Ensure it can grow and has its own scrollbar
        self.txt_tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grp_tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        tr_layout.addWidget(self.txt_tr, 1)
        self.splitter.addWidget(self.grp_tr)

        self._build_codex_panel(self.splitter)

        root.addWidget(self.splitter, 1)

        # timers
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(self._ui_interval_normal_ms)
        self.ui_timer.timeout.connect(self._tick_ui)

        self._start_codex_timer()

        # autosave debounce
        self._cfg_dirty = False
        self._cfg_save_timer = QTimer(self)
        self._cfg_save_timer.setInterval(350)
        self._cfg_save_timer.timeout.connect(self._flush_config_if_dirty)

        # signals
        self.btn_add.clicked.connect(self._add_device_dialog)
        self.txt_output.textChanged.connect(self._on_output_changed)
        self.btn_start.clicked.connect(self._start_all)
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_clear.clicked.connect(self._clear_transcript)

        self.cmb_profile.currentIndexChanged.connect(self._on_profile_changed)
        self.chk_longrun.stateChanged.connect(self._on_longrun_changed)

        self.btn_asr_toggle.clicked.connect(self._toggle_asr_settings)
        self._connect_codex_signals()

        # config widgets -> autosave
        for w in [
            self.chk_asr,
            self.cmb_lang,
            self.cmb_asr_mode,
            self.cmb_model,
            self.cmb_profile,
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
            self.chk_rt_transcript_file,
            self.chk_offline_on_stop,
        ]:
            self._wire_config_change(w)

        if not self._wav_recording_available():
            self.chk_wav.setEnabled(False)
            self.chk_wav.setChecked(False)

        if not self._offline_asr_available():
            self.chk_offline_on_stop.setEnabled(False)
            self.chk_offline_on_stop.setToolTip("Offline ASR runner is unavailable.")
        else:
            self.chk_offline_on_stop.setEnabled(True)

        self._set_codex_inputs_enabled(False)

        # load config + apply profile
        self._load_config_into_ui()
        self._apply_profile_to_fields(self.cmb_profile.currentText(), force=True)
        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)

        # initial asr settings visibility
        self._apply_asr_settings_visibility(expanded=True)

    def _set_status(self, text: str) -> None:
        self._set_label_text_if_changed(self.lbl_status, text)

    @staticmethod
    def _set_label_text_if_changed(label: QLabel, text: str) -> None:
        if label.text() != text:
            label.setText(text)

    @staticmethod
    def _set_progress_if_changed(bar: QProgressBar, value: int) -> None:
        if int(bar.value()) != int(value):
            bar.setValue(int(value))

    @staticmethod
    def _set_line_edit_if_changed(edit: QLineEdit, text: str) -> None:
        if edit.text() != text:
            edit.setText(text)

    def _on_source_error(self, source: str, error: str) -> None:
        ev = {
            "type": "source_error",
            "source": str(source),
            "error": str(error),
            "ts": time.time(),
        }
        try:
            self.asr_ui_q.put_nowait(ev)
        except Exception:
            pass

    def _is_running(self) -> bool:
        return self.engine.is_running()

    def _wav_recording_available(self) -> bool:
        return bool(self.wav_recorder_factory.available())

    def _offline_asr_available(self) -> bool:
        return bool(self.offline_asr_runner.available())

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
        cb.setTristate(False)
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

        # ВАЖНО: clicked(bool) — стабильнее для интерактивного mute/unmute
        cb.clicked.connect(lambda checked, n=name: self._on_source_toggle(n, checked))
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

    def _on_source_toggle(self, name: str, checked: bool) -> None:
        # 1) реально переключаем движок
        self.engine.set_source_enabled(name, bool(checked))

        # 2) логируем в transcript, чтобы было видно, что клик дошёл
        self._append_transcript_line(
            f"[{self._fmt_ts(time.time())}] UI toggle: {name} -> {'ON' if checked else 'MUTED'}"
        )

        # 3) Если ASR включен и режим SPLIT — обновляем tap sources filter (чтобы ASR точно видел текущий набор)
        if self._is_running() and self.chk_asr.isChecked() and self.cmb_asr_mode.currentIndex() == 1:
            enabled_sources = [n for n, r in self.rows.items() if r.enabled.isChecked()]
            self.engine.set_tap_config(mode="sources", sources=enabled_sources, drop_threshold=0.85)

    def _add_device_dialog(self) -> None:
        if self._is_running():
            self._set_status("Stop before adding devices.")
            return

        dlg = DevicePickerDialog(self, catalog=self.device_catalog)
        if dlg.exec() != DevicePickerDialog.Accepted:
            return

        typ, token = dlg.selected()
        if token is None:
            self._set_status("No device selected.")
            return

        try:
            if typ == DevicePickerDialog.TYPE_LOOPBACK:
                name = self._make_unique_name("desktop_audio")
                src = self.audio_source_factory.create_loopback_source(
                    name=name,
                    engine_format=self.fmt,
                    device=token,
                    error_callback=self._on_source_error,
                )
                self.engine.add_source(src)
                self.source_objs[name] = src
                self._add_row(name)
                self._set_status(f"Added loopback -> source '{name}'")
            else:
                name = self._make_unique_name("mic")
                src = self.audio_source_factory.create_microphone_source(name=name, device=token)
                self.engine.add_source(src)
                self.source_objs[name] = src
                self._add_row(name)
                self._set_status(f"Added mic -> source '{name}'")
        except Exception as e:
            self._set_status(f"Failed to add device: {e}")

    # ---------------- utils ----------------

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
        self._closing = True
        try:
            self._stop_all(run_offline_pass=False, wait=True)
        finally:
            self._stop_codex_timer()
            try:
                self.writer.stop()
            except Exception:
                pass
            self._rt_close()
            self._human_log_close()
        event.accept()
