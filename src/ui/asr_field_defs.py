from __future__ import annotations

from typing import Any, List, Tuple

# (config_key, widget_attr, default, lo, hi) — ordered for form display.
# Positions 0-4: base ASR params; positions 5-10: overload params.
# The combo widgets (cmb_compute, cmb_overload_strategy) sit between the two groups in the form.
_ASR_ALL_FIELDS: List[Tuple[str, str, Any, Any, Any]] = [
    ("beam_size",              "txt_beam",          5,      1,    20),
    ("endpoint_silence_ms",    "txt_endpoint",    650.0,  50.0, 5000.0),
    ("max_segment_s",          "txt_maxseg",        7.0,   1.0,   60.0),
    ("overlap_ms",             "txt_overlap",     200.0,   0.0, 2000.0),
    ("vad_energy_threshold",   "txt_vad_thr",    0.0055,  1e-5,    1.0),
    ("overload_enter_qsize",   "txt_over_enter",   18,     1,   999),
    ("overload_exit_qsize",    "txt_over_exit",     6,     1,   999),
    ("overload_hard_qsize",    "txt_over_hard",    28,     1,   999),
    ("overload_beam_cap",      "txt_over_beamcap",  2,     1,    20),
    ("overload_max_segment_s", "txt_over_maxseg",   5.0,  0.5,   60.0),
    ("overload_overlap_ms",    "txt_over_overlap", 120.0,  0.0, 2000.0),
]

_ASR_INT_FIELDS   = [(k, w, d, lo, hi) for k, w, d, lo, hi in _ASR_ALL_FIELDS if isinstance(d, int)]
_ASR_FLOAT_FIELDS = [(k, w, d, lo, hi) for k, w, d, lo, hi in _ASR_ALL_FIELDS if isinstance(d, float)]

# (config_key, widget_attr)
_ASR_COMBO_FIELDS: List[Tuple[str, str]] = [
    ("compute_type",      "cmb_compute"),
    ("overload_strategy", "cmb_overload_strategy"),
]

# All ASR widget attrs that can be enabled/disabled in Custom profile mode
_ASR_CUSTOM_WIDGET_ATTRS: List[str] = (
    [w for _, w in _ASR_COMBO_FIELDS]
    + [w for _, w, *_ in _ASR_ALL_FIELDS]
)
