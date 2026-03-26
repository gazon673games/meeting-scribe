from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)


@dataclass
class CodexProfile:
    id: str
    label: str
    prompt: str
    model: str = ""
    reasoning_effort: str = ""
    codex_profile: str = ""
    answer_prompt: str = ""
    extra_args: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra_args is None:
            self.extra_args = []


class CodexIntegrationMixin:
    def _init_codex_state(self) -> None:
        self._codex_enabled: bool = False
        self._codex_proxy: str = "http://127.0.0.1:10808"
        self._codex_answer_keyword: str = "ANSWER"
        self._codex_timeout_s: int = 90
        self._codex_max_log_chars: int = 24000
        self._codex_command_tokens: List[str] = ["codex"]
        self._codex_path_hints: List[str] = []
        self._codex_profiles: List[CodexProfile] = []
        self._codex_selected_profile_id: str = ""
        self._codex_profile_buttons: Dict[str, QPushButton] = {}
        self._codex_busy: bool = False
        self._codex_ui_q: "queue.Queue[dict]" = queue.Queue(maxsize=120)
        self.codex_timer: Optional[QTimer] = None

    def _build_codex_header(self, root: QVBoxLayout) -> None:
        codex_hdr = QHBoxLayout()
        self.btn_codex_toggle = QPushButton("Show Codex helper")
        self.btn_codex_toggle.setCheckable(True)
        self.btn_codex_toggle.setChecked(False)
        self.btn_codex_toggle.setVisible(False)
        codex_hdr.addWidget(self.btn_codex_toggle)
        codex_hdr.addStretch(1)
        root.addLayout(codex_hdr)

    def _build_codex_panel(self, splitter: QSplitter) -> None:
        self.grp_codex = QGroupBox("Codex helper console")
        codex_layout = QVBoxLayout(self.grp_codex)

        self.lbl_codex_status = QLabel("idle")
        self.lbl_codex_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        codex_layout.addWidget(self.lbl_codex_status)

        self.codex_profiles_row = QHBoxLayout()
        codex_layout.addLayout(self.codex_profiles_row)

        self.txt_codex = QTextEdit()
        self.txt_codex.setReadOnly(True)
        self.txt_codex.setPlaceholderText("Codex output will appear here")
        self.txt_codex.setLineWrapMode(QTextEdit.WidgetWidth)
        self.txt_codex.document().setMaximumBlockCount(1200)
        self.txt_codex.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        codex_layout.addWidget(self.txt_codex, 1)

        codex_input_row = QHBoxLayout()
        self.txt_codex_input = QLineEdit()
        self.txt_codex_input.setPlaceholderText("Type request or ANSWER")
        codex_input_row.addWidget(self.txt_codex_input, 1)
        self.btn_codex_send = QPushButton("Send")
        codex_input_row.addWidget(self.btn_codex_send, 0)
        codex_layout.addLayout(codex_input_row)

        self.grp_codex.setVisible(False)
        splitter.addWidget(self.grp_codex)
        splitter.setSizes([820, 0])

    def _connect_codex_signals(self) -> None:
        self.btn_codex_toggle.clicked.connect(self._toggle_codex_console)
        self.btn_codex_send.clicked.connect(self._on_codex_send_clicked)
        self.txt_codex_input.returnPressed.connect(self._on_codex_send_clicked)

    def _set_codex_inputs_enabled(self, enabled: bool) -> None:
        self.txt_codex_input.setEnabled(bool(enabled))
        self.btn_codex_send.setEnabled(bool(enabled))

    def _start_codex_timer(self) -> None:
        self.codex_timer = QTimer(self)
        self.codex_timer.setInterval(140)
        self.codex_timer.timeout.connect(lambda: self._drain_codex_ui_events(limit=10))
        self.codex_timer.start()

    def _stop_codex_timer(self) -> None:
        try:
            if self.codex_timer is not None and self.codex_timer.isActive():
                self.codex_timer.stop()
        except Exception:
            pass

    def _profile_to_dict(self, p: CodexProfile) -> Dict[str, Any]:
        return {
            "id": str(p.id),
            "label": str(p.label),
            "prompt": str(p.prompt),
            "model": str(p.model),
            "reasoning_effort": str(p.reasoning_effort),
            "codex_profile": str(p.codex_profile),
            "answer_prompt": str(p.answer_prompt),
            "extra_args": [str(x) for x in list(p.extra_args or []) if str(x).strip()],
        }

    def _default_codex_profiles(self) -> List[CodexProfile]:
        return [
            CodexProfile(
                id="default",
                label="Default",
                model="",
                reasoning_effort="low",
                prompt=(
                    "Support a realtime interview. Give short, practical, accurate responses from the session log."
                ),
                answer_prompt=(
                    "Command ANSWER: provide a quick candidate response for the latest question. "
                    "Format: 1) short answer 2) key points."
                ),
            )
        ]

    def _parse_codex_profiles(self, raw_profiles: Any) -> List[CodexProfile]:
        out: List[CodexProfile] = []
        if isinstance(raw_profiles, list):
            for i, item in enumerate(raw_profiles):
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("id") or f"profile_{i+1}").strip() or f"profile_{i+1}"
                label = str(item.get("label") or pid).strip() or pid
                prompt = str(item.get("prompt") or "").strip()
                model = str(item.get("model") or "").strip()
                reasoning_effort = str(
                    item.get("reasoning_effort")
                    or item.get("reasoning")
                    or item.get("reasoning_level")
                    or ""
                ).strip()
                codex_profile = str(item.get("codex_profile") or "").strip()
                answer_prompt = str(item.get("answer_prompt") or "").strip()
                extra_args_raw = item.get("extra_args", [])
                extra_args: List[str] = []
                if isinstance(extra_args_raw, list):
                    extra_args = [str(x).strip() for x in extra_args_raw if str(x).strip()]
                out.append(
                    CodexProfile(
                        id=pid,
                        label=label,
                        prompt=prompt,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        codex_profile=codex_profile,
                        answer_prompt=answer_prompt,
                        extra_args=extra_args,
                    )
                )
        if not out:
            out = self._default_codex_profiles()
        return out

    def _build_codex_config(self) -> Dict[str, Any]:
        cmd_out: Any = "codex"
        if self._codex_command_tokens:
            if len(self._codex_command_tokens) == 1:
                cmd_out = str(self._codex_command_tokens[0])
            else:
                cmd_out = [str(x) for x in self._codex_command_tokens]
        return {
            "enabled": bool(self._codex_enabled),
            "proxy": str(self._codex_proxy),
            "answer_keyword": str(self._codex_answer_keyword),
            "timeout_s": int(self._codex_timeout_s),
            "max_log_chars": int(self._codex_max_log_chars),
            "command": cmd_out,
            "path_hints": [str(x) for x in self._codex_path_hints if str(x).strip()],
            "console_expanded": bool(self.btn_codex_toggle.isChecked()),
            "selected_profile": str(self._codex_selected_profile_id or ""),
            "profiles": [self._profile_to_dict(p) for p in self._codex_profiles],
        }

    def _set_codex_enabled_ui(self, enabled: bool) -> None:
        self._codex_enabled = bool(enabled)
        self.btn_codex_toggle.setVisible(self._codex_enabled)
        if not self._codex_enabled:
            self.btn_codex_toggle.setChecked(False)
            self._apply_codex_console_visibility(expanded=False)
            self._set_codex_inputs_enabled(False)
            self.lbl_codex_status.setText("disabled by config")
            return
        self.lbl_codex_status.setText("idle")
        has_profiles = len(self._codex_profiles) > 0
        can_send = has_profiles and (not self._codex_busy)
        self._set_codex_inputs_enabled(can_send)

    def _clear_codex_profile_buttons(self) -> None:
        while self.codex_profiles_row.count() > 0:
            item = self.codex_profiles_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._codex_profile_buttons.clear()

    def _refresh_codex_profile_buttons(self) -> None:
        self._clear_codex_profile_buttons()
        if not self._codex_profiles:
            self.codex_profiles_row.addWidget(QLabel("No codex profiles in config"), 0)
            self.codex_profiles_row.addStretch(1)
            return

        for p in self._codex_profiles:
            btn = QPushButton(str(p.label))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, pid=p.id: self._set_codex_selected_profile(pid, mark_dirty=checked))
            self.codex_profiles_row.addWidget(btn, 0)
            self._codex_profile_buttons[p.id] = btn
        self.codex_profiles_row.addStretch(1)

    def _set_codex_selected_profile(self, profile_id: str, *, mark_dirty: bool) -> None:
        pid = str(profile_id or "").strip()
        ids = {p.id for p in self._codex_profiles}
        if pid not in ids:
            pid = self._codex_profiles[0].id if self._codex_profiles else ""
        self._codex_selected_profile_id = pid

        for bid, btn in self._codex_profile_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(bid == pid)
            btn.blockSignals(False)

        if mark_dirty:
            self._mark_config_dirty()

    def _apply_codex_console_visibility(self, *, expanded: bool) -> None:
        show = bool(self._codex_enabled and expanded)
        self.grp_codex.setVisible(show)
        self.btn_codex_toggle.setText("Hide Codex helper" if show else "Show Codex helper")
        if show:
            self.splitter.setSizes([620, 260])
        else:
            self.splitter.setSizes([860, 0])

    def _toggle_codex_console(self) -> None:
        expanded = bool(self.btn_codex_toggle.isChecked())
        self._apply_codex_console_visibility(expanded=expanded)
        self._mark_config_dirty()

    def _load_codex_from_config(self, codex: Any) -> None:
        try:
            enabled = bool(codex.get("enabled", False))
            self._codex_proxy = str(codex.get("proxy", "http://127.0.0.1:10808") or "http://127.0.0.1:10808").strip()
            self._codex_answer_keyword = str(codex.get("answer_keyword", "ANSWER") or "ANSWER").strip() or "ANSWER"
            self._codex_timeout_s = self._safe_int(str(codex.get("timeout_s", 90)), 90, 10, 1200)
            self._codex_max_log_chars = self._safe_int(str(codex.get("max_log_chars", 24000)), 24000, 2000, 200000)
            cmd_raw = codex.get("command", "codex")
            cmd_tokens: List[str] = []
            if isinstance(cmd_raw, list):
                cmd_tokens = [str(x).strip() for x in cmd_raw if str(x).strip()]
            elif isinstance(cmd_raw, str):
                s = cmd_raw.strip()
                if s:
                    cmd_tokens = [s]
            if not cmd_tokens:
                cmd_tokens = ["codex"]
            self._codex_command_tokens = cmd_tokens

            hints_raw = codex.get("path_hints", [])
            hints: List[str] = []
            if isinstance(hints_raw, list):
                hints = [str(x).strip() for x in hints_raw if str(x).strip()]
            self._codex_path_hints = hints

            self._codex_profiles = self._parse_codex_profiles(codex.get("profiles", []))

            self._refresh_codex_profile_buttons()
            selected = str(codex.get("selected_profile", "")).strip()
            self._set_codex_selected_profile(selected, mark_dirty=False)

            self._set_codex_enabled_ui(enabled)
            expanded = bool(codex.get("console_expanded", False))
            self.btn_codex_toggle.setChecked(expanded)
            self._apply_codex_console_visibility(expanded=expanded)
        except Exception:
            self._codex_command_tokens = ["codex"]
            self._codex_path_hints = []
            self._codex_profiles = self._default_codex_profiles()
            self._refresh_codex_profile_buttons()
            self._set_codex_selected_profile(self._codex_profiles[0].id, mark_dirty=False)
            self._set_codex_enabled_ui(False)
            self._apply_codex_console_visibility(expanded=False)

    def _append_codex_line(self, line: str) -> None:
        max_chars = 180_000
        if self.txt_codex.document().characterCount() > max_chars:
            self.txt_codex.clear()
            self.txt_codex.append("[codex console cleared: too large]")

        self.txt_codex.append(str(line))
        self.txt_codex.moveCursor(QTextCursor.End)
        self.txt_codex.ensureCursorVisible()

    def _get_selected_codex_profile(self) -> Optional[CodexProfile]:
        for p in self._codex_profiles:
            if p.id == self._codex_selected_profile_id:
                return p
        return self._codex_profiles[0] if self._codex_profiles else None

    def _read_human_log_for_codex(self) -> str:
        path: Optional[Path] = None
        if self._human_log_path is not None and Path(self._human_log_path).exists():
            path = Path(self._human_log_path)
        else:
            d = self.project_root / "human_logs"
            if d.exists():
                files = [x for x in d.glob("chat_*.txt") if x.is_file()]
                if files:
                    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    path = files[0]

        if path is None or not path.exists():
            return ""

        try:
            if self._human_log_fh is not None:
                self._human_log_fh.flush()
        except Exception:
            pass

        max_chars = max(2000, int(self._codex_max_log_chars))
        max_bytes = max_chars * 4 + 4096

        try:
            with path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                start = max(0, int(size) - int(max_bytes))
                fh.seek(start, os.SEEK_SET)
                raw = fh.read()
        except Exception:
            return ""

        txt = raw.decode("utf-8", errors="ignore")

        if len(txt) > max_chars:
            txt = txt[-max_chars:]
            txt = "[log tail]\n" + txt
        return txt.strip()

    def _build_codex_prompt(self, user_text: str, profile: CodexProfile, log_text: str) -> str:
        cmd = str(user_text or "").strip()
        is_answer = cmd.upper() == str(self._codex_answer_keyword).upper()
        model_hint = self._normalize_model_name(profile.model) or "default"
        effort_hint = self._normalize_reasoning_effort(profile.reasoning_effort) or "default"

        if is_answer:
            task = (profile.answer_prompt or "").strip()
            if not task:
                task = (
                    "Command ANSWER: provide a fast answer for the latest question from context.\n"
                    "Format:\n"
                    "1) Short answer (1-3 sentences)\n"
                    "2) Key points (up to 5)\n"
                    "3) Optional clarification question"
                )
        else:
            task = cmd

        base = (
            "You are an assistant for realtime interview support. Be concise and practical.\n"
            f"Profile: {profile.label}\n"
            f"Model hint: {model_hint}\n\n"
            f"Reasoning effort hint: {effort_hint}\n\n"
            "Profile instructions:\n"
            f"{profile.prompt or '(empty)'}\n\n"
            "Task:\n"
            f"{task}\n\n"
            "Current session human-readable log:\n"
        )
        if log_text.strip():
            base += log_text.strip()
        else:
            base += "(log is empty)"
        return base

    @staticmethod
    def _normalize_model_name(raw: str) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        sl = s.lower()
        # These are effort labels from /models UI, not actual model ids.
        effort_aliases = {
            "low",
            "medium",
            "high",
            "xhigh",
            "extra high",
            "extra_high",
            "extra-high",
        }
        if sl in effort_aliases:
            return ""
        return s

    @staticmethod
    def _normalize_reasoning_effort(raw: str) -> str:
        s = str(raw or "").strip().lower()
        if not s:
            return ""
        s = s.replace("(current)", "").strip()
        s_norm = s.replace("_", " ").replace("-", " ")
        s_compact = " ".join([x for x in s_norm.split() if x])
        mapping = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "extra": "xhigh",
            "extra high": "xhigh",
            "very high": "xhigh",
        }
        return mapping.get(s_compact, "")

    def _set_codex_busy(self, busy: bool) -> None:
        self._codex_busy = bool(busy)
        can_send = bool(self._codex_enabled and (not self._codex_busy) and len(self._codex_profiles) > 0)
        self._set_codex_inputs_enabled(can_send)
        if self._codex_enabled:
            self.lbl_codex_status.setText("busy..." if self._codex_busy else "idle")

    def _on_codex_send_clicked(self) -> None:
        if not self._codex_enabled:
            return
        if self._codex_busy:
            self._append_codex_line(f"[{self._fmt_ts(time.time())}] busy: wait for current request")
            return

        req = (self.txt_codex_input.text() or "").strip()
        if not req:
            return

        profile = self._get_selected_codex_profile()
        if profile is None:
            self._append_codex_line(f"[{self._fmt_ts(time.time())}] no codex profile configured")
            return

        log_text = self._read_human_log_for_codex()
        prompt = self._build_codex_prompt(req, profile, log_text)

        self._append_codex_line(f"[{self._fmt_ts(time.time())}] you ({profile.label}): {req}")
        self.txt_codex_input.clear()
        self._set_codex_busy(True)

        th = threading.Thread(
            target=self._run_codex_exec_worker,
            args=(prompt, profile, req),
            name="codex-helper-worker",
            daemon=True,
        )
        th.start()

    def _codex_push_event(self, ev: Dict[str, Any]) -> None:
        try:
            self._codex_ui_q.put_nowait(ev)
        except queue.Full:
            try:
                _ = self._codex_ui_q.get_nowait()
            except Exception:
                pass
            try:
                self._codex_ui_q.put_nowait(ev)
            except Exception:
                pass

    def _codex_common_search_dirs(self) -> List[Path]:
        out: List[Path] = []

        for raw in self._codex_path_hints:
            p = Path(str(raw).strip())
            if p and str(p).strip():
                out.append(p)

        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            out.append(Path(appdata) / "npm")

        home = Path.home()
        out.append(home / "AppData" / "Roaming" / "npm")

        ext_root = home / ".vscode" / "extensions"
        if ext_root.exists():
            try:
                exes = list(ext_root.glob("openai.chatgpt-*/bin/windows-x86_64"))
                exes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                out.extend(exes[:5])
            except Exception:
                pass

        uniq: List[Path] = []
        seen = set()
        for p in out:
            try:
                k = str(p.resolve())
            except Exception:
                k = str(p)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)
        return uniq

    @staticmethod
    def _wrap_cmd_for_windows(exe_path: str, tail: List[str]) -> List[str]:
        p = str(exe_path).strip()
        suffix = Path(p).suffix.lower()
        if suffix in (".cmd", ".bat"):
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [comspec, "/d", "/c", p, *tail]
        return [p, *tail]

    def _resolve_codex_base_command(self) -> Tuple[Optional[List[str]], str]:
        tokens = [str(x).strip() for x in self._codex_command_tokens if str(x).strip()]
        if not tokens:
            tokens = ["codex"]
        base = tokens[0]
        tail = tokens[1:]

        pbase = Path(base)
        if pbase.exists():
            return (self._wrap_cmd_for_windows(str(pbase), tail), "direct_path")

        found = shutil.which(base)
        if found:
            return (self._wrap_cmd_for_windows(found, tail), "path")

        names: List[str] = [base]
        if base.lower() == "codex":
            names = ["codex", "codex.exe", "codex.cmd", "codex.bat", "codex.CMD"]

        for name in names:
            w = shutil.which(name)
            if w:
                return (self._wrap_cmd_for_windows(w, tail), f"path:{name}")

        for d in self._codex_common_search_dirs():
            for name in names:
                cand = d / name
                if cand.exists():
                    return (self._wrap_cmd_for_windows(str(cand), tail), f"hint:{d}")

        try:
            probe = subprocess.run(
                ["where", "codex"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if probe.returncode == 0:
                for line in (probe.stdout or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    pp = Path(line)
                    if pp.exists():
                        return (self._wrap_cmd_for_windows(str(pp), tail), "where")
        except Exception:
            pass

        return (None, "")

    def _run_codex_exec_worker(self, prompt: str, profile: CodexProfile, original_cmd: str) -> None:
        t0 = time.time()
        out_path: Optional[Path] = None
        try:
            env = os.environ.copy()
            proxy = str(self._codex_proxy or "").strip()
            if proxy:
                env["HTTP_PROXY"] = proxy
                env["HTTPS_PROXY"] = proxy
                env["ALL_PROXY"] = proxy
                env["http_proxy"] = proxy
                env["https_proxy"] = proxy
                env["all_proxy"] = proxy

            base_cmd, src = self._resolve_codex_base_command()
            if base_cmd is None:
                self._codex_push_event(
                    {
                        "type": "codex_result",
                        "ok": False,
                        "profile": profile.label,
                        "cmd": original_cmd,
                        "text": (
                            "codex executable not found. "
                            "Set codex.command in config.json (e.g. "
                            "'C:/Users/<you>/AppData/Roaming/npm/codex.cmd' or full codex.exe path)."
                        ),
                        "dt_s": time.time() - t0,
                    }
                )
                return

            cmd: List[str] = list(base_cmd) + ["exec", "--color", "never", "--skip-git-repo-check"]
            model_name = self._normalize_model_name(profile.model)
            if model_name:
                cmd.extend(["-m", model_name])
            effort = self._normalize_reasoning_effort(profile.reasoning_effort)
            if effort:
                cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
            if profile.codex_profile:
                cmd.extend(["-p", str(profile.codex_profile)])
            if profile.extra_args:
                cmd.extend([str(x) for x in profile.extra_args if str(x).strip()])

            fd, tmp = tempfile.mkstemp(prefix="codex_last_", suffix=".txt")
            os.close(fd)
            out_path = Path(tmp)
            cmd.extend(["-o", str(out_path), "-"])

            prompt_safe = prompt.encode("utf-8", errors="replace").decode("utf-8")

            p = subprocess.run(
                cmd,
                input=prompt_safe,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=str(self.project_root),
                env=env,
                timeout=max(10, int(self._codex_timeout_s)),
            )

            out_text = ""
            if out_path.exists():
                try:
                    out_text = out_path.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    out_text = ""
            if not out_text:
                out_text = (p.stdout or "").strip()

            dt = time.time() - t0
            if p.returncode != 0:
                err = (p.stderr or "").strip() or (p.stdout or "").strip()
                if not err:
                    err = f"codex exec failed with code {p.returncode}"
                else:
                    err = f"{err}\n(source={src})"
                self._codex_push_event(
                    {
                        "type": "codex_result",
                        "ok": False,
                        "profile": profile.label,
                        "cmd": original_cmd,
                        "text": err,
                        "dt_s": dt,
                    }
                )
                return

            if not out_text:
                out_text = "(empty response)"
            self._codex_push_event(
                {
                    "type": "codex_result",
                    "ok": True,
                    "profile": profile.label,
                    "cmd": original_cmd,
                    "text": out_text,
                    "dt_s": dt,
                }
            )

        except subprocess.TimeoutExpired:
            self._codex_push_event(
                {
                    "type": "codex_result",
                    "ok": False,
                    "profile": profile.label,
                    "cmd": original_cmd,
                    "text": f"timeout after {int(self._codex_timeout_s)}s",
                    "dt_s": time.time() - t0,
                }
            )
        except FileNotFoundError:
            self._codex_push_event(
                {
                    "type": "codex_result",
                    "ok": False,
                    "profile": profile.label,
                    "cmd": original_cmd,
                    "text": (
                        "codex executable not found at runtime. "
                        "Try setting codex.command in config.json to an explicit path."
                    ),
                    "dt_s": time.time() - t0,
                }
            )
        except Exception as e:
            self._codex_push_event(
                {
                    "type": "codex_result",
                    "ok": False,
                    "profile": profile.label,
                    "cmd": original_cmd,
                    "text": f"{type(e).__name__}: {e}",
                    "dt_s": time.time() - t0,
                }
            )
        finally:
            if out_path is not None:
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _drain_codex_ui_events(self, limit: int = 8) -> None:
        n = 0
        while n < limit:
            try:
                ev = self._codex_ui_q.get_nowait()
            except queue.Empty:
                break
            n += 1

            if str(ev.get("type", "")) != "codex_result":
                continue

            ok = bool(ev.get("ok", False))
            profile = str(ev.get("profile", ""))
            dt_s = float(ev.get("dt_s", 0.0))
            text = str(ev.get("text", "")).strip()
            tss = self._fmt_ts(time.time())

            if ok:
                self._append_codex_line(f"[{tss}] codex ({profile}, {dt_s:.1f}s):")
                self._append_codex_line(text)
            else:
                self._append_codex_line(f"[{tss}] codex error ({profile}, {dt_s:.1f}s): {text}")

            self._set_codex_busy(False)
