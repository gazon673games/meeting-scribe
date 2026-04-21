from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtWidgets import QPushButton

from application.codex_config import CodexProfile
from application.model_policy import ModelOrchestrator


class CodexProfileState:
    """Manages codex profile list, active selection, and button widget sync."""

    def __init__(self) -> None:
        self.profiles: List[CodexProfile] = []
        self.selected_id: str = ""
        self.buttons: Dict[str, QPushButton] = {}

    def set_profiles(self, profiles: List[CodexProfile]) -> None:
        self.profiles = list(profiles)

    def select(self, profile_id: str) -> None:
        pid = str(profile_id or "").strip()
        if pid not in {p.id for p in self.profiles}:
            pid = self.profiles[0].id if self.profiles else ""
        self.selected_id = pid

    def selected_profile(self) -> Optional[CodexProfile]:
        for p in self.profiles:
            if p.id == self.selected_id:
                return p
        return self.profiles[0] if self.profiles else None

    def policy_profile(self, asr_profile: str) -> Optional[CodexProfile]:
        profile_id = ModelOrchestrator().recommend_codex_profile_id(
            asr_profile=asr_profile,
            profiles=list(self.profiles),
            current_profile_id=str(self.selected_id or ""),
        )
        for p in self.profiles:
            if p.id == profile_id:
                return p
        return self.selected_profile()

    def sync_buttons(self) -> None:
        for bid, btn in self.buttons.items():
            btn.blockSignals(True)
            btn.setChecked(bid == self.selected_id)
            btn.blockSignals(False)
