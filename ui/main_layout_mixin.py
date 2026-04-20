from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from application.model_policy import ASR_MODEL_NAMES


class MainWindowLayoutMixin:
    def _build_main_layout(self) -> None:
        outer = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)

        content = QWidget()
        self.scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self.btn_add = QPushButton("Add device...")
        top.addWidget(self.btn_add)

        self.chk_longrun = QCheckBox("Long-run mode (lighter UI)")
        self.chk_longrun.setChecked(False)
        top.addWidget(self.chk_longrun)

        top.addStretch(1)
        root.addLayout(top)

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
        self.cmb_model.addItems(list(ASR_MODEL_NAMES))
        self.cmb_model.setCurrentText("medium")
        asr_row.addWidget(self.cmb_model)

        asr_row.addStretch(1)
        root.addLayout(asr_row)

        hdr = QHBoxLayout()
        self.btn_asr_toggle = QPushButton("Show ASR settings")
        self.btn_asr_toggle.setCheckable(True)
        self.btn_asr_toggle.setChecked(False)
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

        opt_row = QHBoxLayout()
        self.chk_offline_on_stop = QCheckBox("Offline pass on Stop (quality)")
        self.chk_rt_transcript_file = QCheckBox("Also write realtime transcript to file")
        opt_row.addWidget(self.chk_offline_on_stop)
        opt_row.addWidget(self.chk_rt_transcript_file)
        opt_row.addStretch(1)
        root.addLayout(opt_row)

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

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)

        self.grp_tr = QGroupBox("Transcript (utterances + important events)")
        tr_layout = QVBoxLayout(self.grp_tr)
        self.txt_tr = QTextEdit()
        self.txt_tr.setReadOnly(True)
        self.txt_tr.setPlaceholderText("ASR output will appear here...")
        self.txt_tr.setLineWrapMode(QTextEdit.WidgetWidth)
        self.txt_tr.document().setMaximumBlockCount(2500)

        self.txt_tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grp_tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        tr_layout.addWidget(self.txt_tr, 1)
        self._build_codex_header(tr_layout)
        self.splitter.addWidget(self.grp_tr)

        self._build_codex_panel(self.splitter)

        root.addWidget(self.splitter, 1)

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(self._ui_interval_normal_ms)
        self.ui_timer.timeout.connect(self._tick_ui)

        self._start_codex_timer()

        self._cfg_dirty = False
        self._cfg_save_timer = QTimer(self)
        self._cfg_save_timer.setInterval(350)
        self._cfg_save_timer.timeout.connect(self._flush_config_if_dirty)

        self._connect_main_signals()
        self._wire_autosaved_widgets()
        self._apply_initial_availability()

        self._set_codex_inputs_enabled(False)
        self._load_config_into_ui()
        self._apply_profile_to_fields(self.cmb_profile.currentText(), force=True)
        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)
        self._apply_asr_settings_visibility(expanded=bool(self.btn_asr_toggle.isChecked()))

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
        for widget in [
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
            self._wire_config_change(widget)

    def _apply_initial_availability(self) -> None:
        if not self._wav_recording_available():
            self.chk_wav.setEnabled(False)
            self.chk_wav.setChecked(False)

        if not self._offline_asr_available():
            self.chk_offline_on_stop.setEnabled(False)
            self.chk_offline_on_stop.setToolTip("Offline ASR runner is unavailable.")
        else:
            self.chk_offline_on_stop.setEnabled(True)
