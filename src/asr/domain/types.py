from __future__ import annotations

from typing import Literal

Mode = Literal["mix", "split"]
DiarBackend = Literal["pyannote", "online", "nemo"]
OverloadStrategy = Literal["drop_old", "keep_all"]
