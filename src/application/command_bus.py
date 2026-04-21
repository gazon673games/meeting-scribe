from __future__ import annotations

from typing import Any, Callable, Dict, Type


CommandHandler = Callable[[Any], Any]


class CommandDispatcher:
    def __init__(self) -> None:
        self._handlers: Dict[Type[Any], CommandHandler] = {}

    def register(self, command_cls: Type[Any], handler: CommandHandler) -> None:
        self._handlers[command_cls] = handler

    def dispatch(self, command: Any) -> Any:
        handler = self._handlers.get(type(command))
        if handler is None:
            raise KeyError(f"No command handler registered for {type(command).__name__}")
        return handler(command)
