from __future__ import annotations

from collections.abc import Callable
from typing import Any

CommandHandler = Callable[[Any], Any]


class CommandDispatcher:
    def __init__(self) -> None:
        self._handlers: dict[type[Any], CommandHandler] = {}

    def register(self, command_cls: type[Any], handler: CommandHandler) -> None:
        self._handlers[command_cls] = handler

    def dispatch(self, command: Any) -> Any:
        handler = self._handlers.get(type(command))
        if handler is None:
            raise KeyError(
                f'No command handler registered for {type(command).__name__}'
            )
        return handler(command)
