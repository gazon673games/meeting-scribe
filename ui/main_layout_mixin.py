from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QScrollArea, QSizePolicy,
    QSplitter, QTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from application.model_policy import ASR_MODEL_NAMES
from ui.asr_field_defs import _ASR_ALL_FIELDS, _ASR_CUSTOM_WIDGET_ATTRS


class MainWindowLayoutMixin:
    def _build_main_layout(self) -> None:
        content = QWidget()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.addWidget(self.scroll, 1)

        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._build_controls_row())
        root.addLayout(self._build_asr_row())
        root.addLayout(self._build_asr_toggle_row())
        root.addWidget(self._build_asr_settings_group())
        root.addLayout(self._build_options_row())
        root.addLayout(self._build_wav_row())
        root.addLayout(self._build_transport_row())
        root.addWidget(self._build_sources_group())
        root.addLayout(self._build_master_meter_row())
        root.addLayout(self._build_status_rows())
        root.addWidget(self._build_transcript_splitter(), 1)

        self._init_timers()
        self._connect_main_signals()
        self._wire_autosaved_widgets()
        self._apply_initial_availability()
        self._set_codex_inputs_enabled(False)
        self._load_config_into_ui()
        self._apply_profile_to_fields(self.cmb_profile.currentText(), force=True)
        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)
        self._apply_asr_settings_visibility(expanded=bool(self.btn_asr_toggle.isChecked()))

    # ── section builders ───────────────────────────────────────────────

    def _build_controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_add = QPushButton("Add device...")
        self.chk_longrun = QCheckBox("Long-run mode (lighter UI)")
        row.addWidget(self.btn_add)
        row.addWidget(self.chk_longrun)
        row.addStretch(1)
        return row

    def _build_asr_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.chk_asr = QCheckBox("Enable ASR")
        self.chk_asr.setChecked(True)
        row.addWidget(self.chk_asr)

        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems([self.PROFILE_REALTIME, self.PROFILE_BALANCED, self.PROFILE_QUALITY, self.PROFILE_CUSTOM])
        self.cmb_profile.setCurrentText(self.PROFILE_BALANCED)

        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(["ru", "en", "auto"])

        self.cmb_asr_mode = QComboBox()
        self.cmb_asr_mode.addItems(["MIX (master)", "SPLIT (all sources)"])
        self.cmb_asr_mode.setCurrentIndex(1)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(list(ASR_MODEL_NAMES))
        self.cmb_model.setCurrentText("medium")

        for label, widget in [("Profile:", self.cmb_profile), ("Lang:", self.cmb_lang),
                               ("Mode:", self.cmb_asr_mode), ("Model:", self.cmb_model)]:
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addStretch(1)
        return row

    def _build_asr_toggle_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_asr_toggle = QPushButton("Show ASR settings")
        self.btn_asr_toggle.setCheckable(True)
        row.addWidget(self.btn_asr_toggle)
        row.addStretch(1)
        return row

    def _build_asr_settings_group(self) -> QGroupBox:
        self.grp_asr_cfg = QGroupBox("ASR settings (Profile)")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.cmb_compute = QComboBox()
        self.cmb_compute.addItems(["int8_float16", "float16", "int8", "int8_float32", "float32"])
        self.cmb_compute.setCurrentText("float16")
        form.addRow("compute_type:", self.cmb_compute)

        for key, attr, default, *_ in _ASR_ALL_FIELDS[:5]:   # base ASR params
            w = QLineEdit(str(default))
            setattr(self, attr, w)
            form.addRow(f"{key}:", w)

        self.cmb_overload_strategy = QComboBox()
        self.cmb_overload_strategy.addItems(["drop_old", "keep_all"])
        form.addRow("overload_strategy:", self.cmb_overload_strategy)

        for key, attr, default, *_ in _ASR_ALL_FIELDS[5:]:   # overload params
            w = QLineEdit(str(default))
            setattr(self, attr, w)
            form.addRow(f"{key}:", w)

        QVBoxLayout(self.grp_asr_cfg).addLayout(form)
        return self.grp_asr_cfg

    def _build_options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.chk_offline_on_stop = QCheckBox("Offline pass on Stop (quality)")
        self.chk_rt_transcript_file = QCheckBox("Also write realtime transcript to file")
        row.addWidget(self.chk_offline_on_stop)
        row.addWidget(self.chk_rt_transcript_file)
        row.addStretch(1)
        return row

    def _build_wav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.chk_wav = QCheckBox("Write WAV (master mix)")
        self.txt_output = QLineEdit(self.output_name)
        self.txt_output.setPlaceholderText("capture_mix.wav")
        row.addWidget(self.chk_wav)
        row.addWidget(QLabel("Output file (project root):"))
        row.addWidget(self.txt_output, 1)
        return row

    def _build_transport_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        self.btn_clear = QPushButton("Clear transcript")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        return row

    def _build_sources_group(self) -> QGroupBox:
        self.grp = QGroupBox("Sources")
        self.grp_layout = QVBoxLayout(self.grp)
        return self.grp

    def _build_master_meter_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.master_meter = QProgressBar()
        self.master_meter.setRange(0, 100)
        self.master_status = QLabel("idle")
        self.master_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(QLabel("MASTER"))
        row.addWidget(self.master_meter, 1)
        row.addWidget(self.master_status)
        return row

    def _build_status_rows(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self.lbl_drops = QLabel("drops: 0")
        self.lbl_resources = QLabel("resources: n/a")
        self.lbl_completeness = QLabel("Completeness: OK")
        self.lbl_status = QLabel("ready")
        for lbl in (self.lbl_drops, self.lbl_resources, self.lbl_completeness, self.lbl_status):
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top = QHBoxLayout()
        top.addWidget(self.lbl_drops, 2)
        top.addWidget(self.lbl_resources, 1)
        col.addLayout(top)
        col.addWidget(self.lbl_completeness)
        col.addWidget(self.lbl_status)
        return col

    def _build_transcript_splitter(self) -> QSplitter:
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)

        self.grp_tr = QGroupBox("Transcript (utterances + important events)")
        self.grp_tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.txt_tr = QTextEdit()
        self.txt_tr.setReadOnly(True)
        self.txt_tr.setPlaceholderText("ASR output will appear here...")
        self.txt_tr.setLineWrapMode(QTextEdit.WidgetWidth)
        self.txt_tr.document().setMaximumBlockCount(2500)
        self.txt_tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tr_layout = QVBoxLayout(self.grp_tr)
        tr_layout.addWidget(self.txt_tr, 1)
        self._build_codex_header(tr_layout)
        self.splitter.addWidget(self.grp_tr)
        self._build_codex_panel(self.splitter)
        return self.splitter

    # ── init helpers ───────────────────────────────────────────────────

    def _init_timers(self) -> None:
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(self._ui_interval_normal_ms)
        self.ui_timer.timeout.connect(self._tick_ui)
        self._start_codex_timer()
        self._cfg_dirty = False
        self._cfg_save_timer = QTimer(self)
        self._cfg_save_timer.setInterval(350)
        self._cfg_save_timer.timeout.connect(self._flush_config_if_dirty)

    def _connect_main_signals(self) -> None:
        self.btn_add.clicked.connect(self._add_device_dialog)
        self.txt_output.textChanged.connect(self._on_output_changed)
        self.btn_start.clicked.connect(self._start_all)
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_clear.clicked.connect(self._clear_transcript)
        self.cmb_profile.currentIndexChanged.connect(self._on_profile_changed)
        self.cmb_lang.currentIndexChanged.connect(self._on_policy_input_changed)
        self.chk_longrun.stateChanged.connect(self._on_longrun_changed)
        self.btn_asr_toggle.clicked.connect(self._toggle_asr_settings)
        self._connect_codex_signals()

    def _wire_autosaved_widgets(self) -> None:
        asr = [getattr(self, attr) for attr in _ASR_CUSTOM_WIDGET_ATTRS]
        ui  = [self.chk_asr, self.cmb_lang, self.cmb_asr_mode, self.cmb_model,
               self.cmb_profile, self.chk_wav, self.txt_output,
               self.chk_longrun, self.chk_rt_transcript_file, self.chk_offline_on_stop]
        for w in asr + ui:
            self._wire_config_change(w)

    def _apply_initial_availability(self) -> None:
        if not self._wav_recording_available():
            self.chk_wav.setEnabled(False)
            self.chk_wav.setChecked(False)
        if not self._offline_asr_available():
            self.chk_offline_on_stop.setEnabled(False)
            self.chk_offline_on_stop.setToolTip("Offline ASR runner is unavailable.")
