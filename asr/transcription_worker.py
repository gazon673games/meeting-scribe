from __future__ import annotations

import queue
import time
from typing import Callable, Dict, Optional

from asr.domain import Segment
from asr.metrics import ASRMetrics
from asr.text import normalize_text, trim_overlap

LogEvent = Callable[[dict], None]


class TranscriptionWorkerRuntime:
    def __init__(
        self,
        *,
        segment_queue: "queue.Queue[Segment]",
        stop_event,
        log_event: LogEvent,
        metrics: ASRMetrics,
        overload,
        beam_controller,
        diarization,
        utterances,
        model_name: str,
        language: Optional[str],
        device: str,
        compute_type: str,
        beam_size: int,
        initial_prompt: Optional[str],
        text_dedup_enabled: bool,
        text_dedup_window: int,
        adaptive_beam_enabled: bool,
        log_speaker_labels: bool,
    ) -> None:
        self._seg_q = segment_queue
        self._stop = stop_event
        self._log_event = log_event
        self._metrics = metrics
        self._over = overload
        self._beam_ctl = beam_controller
        self._diar = diarization
        self._utt = utterances

        self._model_name = str(model_name)
        self._language = language
        self._device = str(device)
        self._compute_type = str(compute_type)
        self._beam_size = int(beam_size)
        self._initial_prompt = initial_prompt

        self._text_dedup_enabled = bool(text_dedup_enabled)
        self._text_dedup_window = int(text_dedup_window)
        self._adaptive_beam_enabled = bool(adaptive_beam_enabled)
        self._log_speaker_labels = bool(log_speaker_labels)

        self._last_text: Dict[str, str] = {}
        self._asr = None
        self.asr_init_error: Optional[str] = None

    def reset_runtime(self) -> None:
        self._last_text.clear()
        self.asr_init_error = None

    def run_safe(self) -> None:
        try:
            self.run()
        except Exception as e:
            self._log_event({"type": "error", "where": "worker", "error": str(e), "ts": time.time()})
        finally:
            # FasterWhisperASR owns native/CUDA resources; release them in the same
            # worker thread that created and used them to avoid cross-thread teardown.
            self._asr = None

    def run(self) -> None:
        self._log_event({"type": "asr_init_start", "model": self._model_name, "device": self._device, "ts": time.time()})

        from asr.worker_faster_whisper import FasterWhisperASR

        try:
            self._asr = FasterWhisperASR(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
                compute_type=self._compute_type,
                beam_size=self._beam_size,
                initial_prompt=self._initial_prompt,
            )
            self._log_event({"type": "asr_init_ok", "model": self._model_name, "ts": time.time()})
        except Exception as e:
            self.asr_init_error = str(e)
            self._log_event({"type": "error", "where": "asr_init", "error": str(e), "ts": time.time()})
            return

        self._diar.init_backend(self._log_event)

        while not self._stop.is_set():
            self._update_overload_state()
            self._drain_old_segments_if_hard_overload()

            try:
                seg = self._seg_q.get(timeout=0.2)
            except queue.Empty:
                if self._utt.enabled:
                    try:
                        self.flush_utterances(force=False)
                    except Exception:
                        pass
                self.emit_metrics(force=False)
                continue

            self._update_overload_state()
            self._transcribe_segment(seg)
            self.emit_metrics(force=False)

    def emit_metrics(self, *, force: bool = False) -> None:
        event = self._metrics.build_event(
            force=force,
            seg_qsize=int(self._seg_q.qsize()),
            overload_active=bool(self._over.active),
            overload_strategy=str(self._over.strategy),
            hard_overload=bool(self._over.hard_active),
        )
        if event is not None:
            self._log_event(event)

    def flush_utterances(self, *, force: bool) -> None:
        for event in self._utt.flush_all(now=time.time(), force=force, overload=bool(self._over.active)):
            self._log_event(event)

    def _update_overload_state(self) -> None:
        now = time.time()
        for event in self._over.update(
            seg_qsize=int(self._seg_q.qsize()),
            beam_cur=int(self._beam_ctl.cur_beam),
            lag_s=float(self._metrics.last_lag_s),
            now=now,
        ):
            self._log_event(event)

    def _drain_old_segments_if_hard_overload(self) -> None:
        qsz = int(self._seg_q.qsize())
        drop_cnt = self._over.drop_old_count(qsz)
        if drop_cnt <= 0:
            return

        dropped = 0
        for _ in range(drop_cnt):
            try:
                _ = self._seg_q.get_nowait()
                dropped += 1
            except queue.Empty:
                break

        if dropped <= 0:
            return

        self._metrics.record_segments_skipped(dropped)
        self._log_event(
            {
                "type": "segment_skipped_overload",
                "count": int(dropped),
                "seg_qsize": int(self._seg_q.qsize()),
                "ts": time.time(),
            }
        )

    def _transcribe_segment(self, seg: Segment) -> None:
        speaker = self._diar.speaker_for_segment(seg, self._log_event)
        seg_dur_s = max(1e-6, float(seg.audio_16k.shape[0]) / 16000.0)

        now0 = time.time()
        queue_wait_s = max(0.0, float(now0) - float(seg.enqueue_ts))

        beam_to_use = int(self._beam_ctl.cur_beam)
        beam_to_use = self._over.limit_beam(beam_to_use)

        t0 = time.time()
        try:
            res = self._asr.transcribe(seg.audio_16k, beam_size=beam_to_use)
            text = (res.get("text") or "").strip()
        except Exception as e:
            self._log_event(
                {
                    "type": "segment",
                    "stream": seg.stream,
                    "speaker": speaker if self._log_speaker_labels else "S?",
                    "t_start": seg.t_start,
                    "t_end": seg.t_end,
                    "text": "",
                    "error": str(e),
                    "overload": bool(self._over.active),
                    "ts": time.time(),
                }
            )
            return

        asr_latency_s = time.time() - t0
        total_lag_s = float(queue_wait_s) + float(asr_latency_s)

        removed = 0
        if self._text_dedup_enabled:
            prev = self._last_text.get(seg.stream, "")
            trimmed, removed = trim_overlap(prev, text, max_window=self._text_dedup_window, min_match=8)
            text = trimmed
            if text:
                self._last_text[seg.stream] = normalize_text(prev + " " + text).strip()
        else:
            if text:
                self._last_text[seg.stream] = normalize_text(text)

        self._metrics.record_latency(asr_latency_s=asr_latency_s, total_lag_s=total_lag_s)

        if self._adaptive_beam_enabled:
            now = time.time()
            qsz = int(self._seg_q.qsize())
            new_beam, reason = self._beam_ctl.maybe_update(
                seg_qsize=qsz,
                last_latency_s=float(asr_latency_s),
                last_dur_s=float(seg_dur_s),
                now=now,
            )
            if reason:
                self._log_event({"type": "asr_beam_update", "beam": int(new_beam), "reason": reason, "ts": now})

        self._log_event(
            {
                "type": "segment",
                "stream": seg.stream,
                "speaker": speaker if self._log_speaker_labels else "S?",
                "t_start": seg.t_start,
                "t_end": seg.t_end,
                "text": normalize_text(text),
                "latency_s": float(asr_latency_s),
                "seg_dur_s": float(seg_dur_s),
                "beam_used": int(beam_to_use),
                "dedup_removed_chars": int(removed),
                "seg_qsize": int(self._seg_q.qsize()),
                "overload": bool(self._over.active),
                "overload_strategy": self._over.strategy,
                "hard_overload": bool(self._over.hard_active),
                "lag_s": float(self._metrics.last_lag_s),
                "ts": time.time(),
            }
        )

        if self._utt.enabled and text.strip():
            for event in self._utt.update(
                stream=str(seg.stream),
                speaker=str(speaker),
                t_start=float(seg.t_start),
                t_end=float(seg.t_end),
                text=str(text),
                now=time.time(),
                overload=bool(self._over.active),
            ):
                self._log_event(event)
