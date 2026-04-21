from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

CODEX_CONTEXT_TRANSCRIPT = "transcript"
CODEX_CONTEXT_CURRENT_HUMAN_LOG = "current_human_log"
CODEX_CONTEXT_LATEST_HUMAN_LOG = "latest_human_log"
CODEX_CONTEXT_FILE_PREFIX = "human_log:"


@dataclass(frozen=True)
class AssistantContext:
    source: str
    label: str
    text: Optional[str] = None
    human_log_path: Optional[Path] = None
    human_log_fh: Any = None


class CodexContextMixin:
    def _refresh_codex_context_sources(self) -> None:
        selected = self._codex_context_source_from_ui(default=self._codex_context_source)
        self.cmb_codex_context.blockSignals(True)
        self.cmb_codex_context.clear()
        self.cmb_codex_context.addItem("Current transcript", CODEX_CONTEXT_TRANSCRIPT)
        self.cmb_codex_context.addItem("Current session human log", CODEX_CONTEXT_CURRENT_HUMAN_LOG)
        self.cmb_codex_context.addItem("Latest human log", CODEX_CONTEXT_LATEST_HUMAN_LOG)

        for path in self._recent_human_log_files(limit=30):
            self.cmb_codex_context.addItem(f"Log: {path.name}", f"{CODEX_CONTEXT_FILE_PREFIX}{path.name}")

        self._set_codex_context_source(selected, mark_dirty=False)
        self.cmb_codex_context.blockSignals(False)

    def _codex_context_source_from_ui(self, default: str = CODEX_CONTEXT_TRANSCRIPT) -> str:
        combo = getattr(self, "cmb_codex_context", None)
        if combo is None:
            return str(default or CODEX_CONTEXT_TRANSCRIPT)
        data = combo.currentData()
        return str(data or default or CODEX_CONTEXT_TRANSCRIPT)

    def _set_codex_context_source(self, source: str, *, mark_dirty: bool = False) -> None:
        wanted = str(source or CODEX_CONTEXT_TRANSCRIPT)
        idx = self.cmb_codex_context.findData(wanted)
        if idx < 0 and wanted.startswith(CODEX_CONTEXT_FILE_PREFIX):
            idx = self._append_missing_context_file(wanted)
        if idx < 0:
            idx = self.cmb_codex_context.findData(CODEX_CONTEXT_TRANSCRIPT)
        if idx >= 0:
            self.cmb_codex_context.setCurrentIndex(idx)
        self._codex_context_source = self._codex_context_source_from_ui(default=wanted)
        if mark_dirty:
            self._mark_config_dirty()

    def _append_missing_context_file(self, wanted: str) -> int:
        name = Path(wanted[len(CODEX_CONTEXT_FILE_PREFIX):]).name
        path = self.project_root / "human_logs" / name
        if path.exists():
            self.cmb_codex_context.addItem(f"Log: {path.name}", f"{CODEX_CONTEXT_FILE_PREFIX}{path.name}")
            return self.cmb_codex_context.findData(f"{CODEX_CONTEXT_FILE_PREFIX}{path.name}")
        return self.cmb_codex_context.findData(CODEX_CONTEXT_LATEST_HUMAN_LOG)

    def _on_codex_context_changed(self) -> None:
        self._codex_context_source = self._codex_context_source_from_ui()
        self._mark_config_dirty()

    def _resolve_codex_context(self) -> AssistantContext:
        source = self._codex_context_source_from_ui()
        self._codex_context_source = source

        if source == CODEX_CONTEXT_TRANSCRIPT:
            return AssistantContext(source=source, label="current transcript", text=self.txt_tr.toPlainText())

        if source == CODEX_CONTEXT_CURRENT_HUMAN_LOG:
            return self._current_human_log_context(source)

        if source.startswith(CODEX_CONTEXT_FILE_PREFIX):
            return self._human_log_file_context(source)

        return AssistantContext(source=source, label="latest human log")

    def _current_human_log_context(self, source: str) -> AssistantContext:
        self._sync_transcript_store_refs()
        if self._human_log_path is None:
            return AssistantContext(source=source, label="current human log (empty)", text="")
        self._flush_current_human_log()
        return AssistantContext(
            source=source,
            label="current human log",
            human_log_path=Path(self._human_log_path),
            human_log_fh=self._human_log_fh,
        )

    def _human_log_file_context(self, source: str) -> AssistantContext:
        name = Path(source[len(CODEX_CONTEXT_FILE_PREFIX):]).name
        path = self.project_root / "human_logs" / name
        if not path.exists():
            return AssistantContext(source=source, label=f"human log {name} (missing)", text="")
        return AssistantContext(source=source, label=f"human log {name}", human_log_path=path)

    def _recent_human_log_files(self, *, limit: int) -> List[Path]:
        logs_dir = self.project_root / "human_logs"
        try:
            files = [path for path in logs_dir.glob("chat_*.txt") if path.is_file()]
            files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return files[: int(limit)]
        except Exception:
            return []

    def _flush_current_human_log(self) -> None:
        try:
            if self._human_log_fh is not None:
                self._human_log_fh.flush()
        except Exception:
            pass
