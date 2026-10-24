from __future__ import annotations

from typing import Any

from interface.backend_parts.system_utils import int_or_zero


def build_process_session_payload(
    owner: Any, raw_groups: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    counter = 0
    owner._clear_device_tokens_for_kinds({'process'})

    for group_index, group in enumerate(raw_groups):
        group_label = str(group.get('label') or f'Output {group_index + 1}')
        group_id = str(group.get('id') or f'process-output:{group_index}')
        group_sessions, counter = _build_group_sessions(
            owner, group, counter, group_id, group_label
        )
        if group_sessions:
            groups.append(
                {
                    'id': group_id,
                    'label': group_label,
                    'sessions': group_sessions,
                }
            )
            sessions.extend(group_sessions)

    return {'sessions': sessions, 'groups': groups}


def _build_group_sessions(
    owner: Any,
    group: dict[str, Any],
    counter: int,
    group_id: str,
    group_label: str,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    for session in group.get('sessions', []):
        record, counter = _build_session_record(
            session, counter, group_id, group_label
        )
        if record is None:
            continue
        owner._device_tokens[record['id']] = _device_token_record(
            record, session
        )
        records.append(record)
    return records, counter


def _build_session_record(
    session: dict[str, Any],
    counter: int,
    group_id: str,
    group_label: str,
) -> tuple[dict[str, Any] | None, int]:
    pid = int_or_zero(session.get('pid'))
    if pid <= 0:
        return None, counter
    label = str(session.get('label') or f'PID {pid}')
    counter += 1
    endpoint_id = str(session.get('endpointId') or group_id)
    endpoint_label = str(session.get('endpointLabel') or group_label)
    device_id = f'process:{pid}:{endpoint_id}:{session.get("index", 0)}'
    record = {
        **session,
        'id': device_id,
        'kind': 'process',
        'pid': pid,
        'label': label,
        'groupId': group_id,
        'groupLabel': group_label,
        'endpointId': endpoint_id,
        'endpointLabel': endpoint_label,
        'fullLabel': f'{group_label} / {label}',
    }
    return record, counter


def _device_token_record(
    record: dict[str, Any], session: dict[str, Any]
) -> tuple[str, dict[str, Any], str]:
    return (
        'process',
        {
            'pid': int(record['pid']),
            'index': session.get('index', 0),
            'endpointId': record['endpointId'],
            'endpointLabel': record['endpointLabel'],
        },
        str(record['fullLabel']),
    )
