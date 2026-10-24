from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendRequest:
    request_id: Any
    method: str
    params: dict[str, Any] | None

    @classmethod
    def from_message(cls, message: Any) -> BackendRequest:
        if not isinstance(message, dict):
            raise ValueError('request must be a JSON object')
        method = message.get('method')
        params = message.get('params', {})
        if not isinstance(method, str) or not method:
            raise ValueError('request.method must be a non-empty string')
        if params is not None and not isinstance(params, dict):
            raise ValueError('request.params must be an object')
        return cls(request_id=message.get('id'), method=method, params=params)

    def success(self, result: Any) -> dict[str, Any]:
        return {'id': self.request_id, 'ok': True, 'result': result}


def error_response(request_id: Any, error: Exception) -> dict[str, Any]:
    return {
        'id': request_id,
        'ok': False,
        'error': {'type': type(error).__name__, 'message': str(error)},
    }
