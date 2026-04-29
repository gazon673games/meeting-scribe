from __future__ import annotations

import queue
import time
from typing import Callable, Optional, Tuple

from asr.application.metrics import ASRMetrics
from asr.application.ports import AsrBackendFactoryPort, AsrBackendPort
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.dedup import StreamDedupFilter
from asr.domain.segments import Segment
from asr.domain.text import normalize_text
from diarization.domain.speaker_labels import source_speaker_label

LogEvent = Callable[[dict], None]


class TranscriptionWorkerRuntime:
    def __init__(
        self,
        *,
        config: TranscriptionWorkerConfig,
        segment_queue: "queue.Queue[Segment]",
        stop_event,
        log_event: LogEvent,
        metrics: ASRMetrics,
        overload,
        beam_controller,
        diarization,
        utterances,
        asr_backend_factory: AsrBackendFactoryPort,
    ) -> None:
        self._seg_q = segment_queue
        self._stop = stop_event
        self._log_event = log_event
        self._metrics = metrics
        self._over = overload
        self._beam_ctl = beam_controller
        self._diar = diarization
        self._utt = utterances
        self._asr_backend_factory = asr_backend_factory

        self._model_name = str(config.model_name)
        self._language = config.language
        self._device = str(config.device)
        self._compute_type = str(config.compute_type)
        self._cpu_threads = int(config.cpu_threads)
        self._num_workers = int(config.num_workers)
        self._beam_size = int(config.beam_size)
        self._initial_prompt = config.initial_prompt
        self._adaptive_beam_enabled = bool(config.adaptive_beam_enabled)
        self._log_speaker_labels = bool(config.log_speaker_labels)
        self._init_diarization = bool(config.init_diarization)
        self._diarization_blocking_lookup = bool(config.diarization_blocking_lookup)
        self._source_speaker_labels = dict(config.source_speaker_labels or {})

        self._dedup = StreamDedupFilter(
            enabled=bool(config.text_dedup_enabled),
            window=int(config.text_dedup_window),
        )
        self._asr: Optional[AsrBackendPort] = None
        self.asr_init_error: Optional[str] = None

    def reset_runtime(self) -> None:
        self._dedup.reset()
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

        try:
            self._asr = self._asr_backend_factory(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,
                beam_size=self._beam_size,
                initial_prompt=self._initial_prompt,
            )
            self._log_event({"type": "asr_init_ok", "model": self._model_name, "ts": time.time()})
        except Exception as e:
            self.asr_init_error = str(e)
            self._log_event({"type": "error", "where": "asr_init", "error": str(e), "ts": time.time()})
            return

        if self._init_diarization:
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
                self._seg_q.get_nowait()
                dropped += 1
            except queue.Empty:
                break

        if dropped <= 0:
            return

        self._metrics.record_segments_skipped(dropped)
        self._log_event({
            "type": "segment_skipped_overload",
            "count": int(dropped),
            "seg_qsize": int(self._seg_q.qsize()),
            "ts": time.time(),
        })

    def _transcribe_segment(self, seg: Segment) -> None:
        speaker = self._speaker_for_segment(seg)
        seg_dur_s = max(1e-6, float(seg.duration_s))
        beam_to_use = self._over.limit_beam(int(self._beam_ctl.cur_beam))
        queue_wait_s = seg.queue_wait_s(time.time())

        self._log_event({
            "type": "asr_segment_processing",
            "stream": str(seg.stream),
            "t_start": float(seg.t_start),
            "t_end": float(seg.t_end),
            "seg_dur_s": float(seg_dur_s),
            "seg_qsize": int(self._seg_q.qsize()),
            "ts": time.time(),
        })
        asr_result = self._run_asr(seg, speaker, beam_to_use)
        if asr_result is None:
            return
        text, asr_latency_s = asr_result

        text, removed = self._dedup.filter(seg.stream, text)
        total_lag_s = queue_wait_s + asr_latency_s
        self._metrics.record_latency(asr_latency_s=asr_latency_s, total_lag_s=total_lag_s)
        self._maybe_update_beam(seg_dur_s, asr_latency_s)
        self._log_segment_event(seg, speaker, text, asr_latency_s, seg_dur_s, beam_to_use, removed)
        self._log_event({
            "type": "asr_segment_done",
            "stream": str(seg.stream),
            "speaker": str(speaker),
            "t_start": float(seg.t_start),
            "t_end": float(seg.t_end),
            "latency_s": float(asr_latency_s),
            "text_chars": len(text),
            "seg_qsize": int(self._seg_q.qsize()),
            "ts": time.time(),
        })
        if self._utt.enabled and text.strip():
            self._update_utterances(seg, speaker, text)

    def _speaker_for_segment(self, seg: Segment) -> str:
        if self._diarization_blocking_lookup:
            return self._diar.speaker_for_segment(seg, self._log_event)
        return source_speaker_label(self._source_speaker_labels, seg.stream)

    def _run_asr(
        self, seg: Segment, speaker: str, beam_to_use: int
    ) -> Optional[Tuple[str, float]]:
        t0 = time.time()
        try:
            res = self._asr.transcribe(seg.audio.samples, beam_size=beam_to_use)
            text = (res.get("text") or "").strip()
        except Exception as e:
            self._log_event({
                "type": "segment",
                "stream": seg.stream,
                "speaker": speaker if self._log_speaker_labels else "S?",
                "t_start": seg.t_start,
                "t_end": seg.t_end,
                "text": "",
                "error": str(e),
                "overload": bool(self._over.active),
                "ts": time.time(),
            })
            return None
        return text, time.time() - t0

    def _maybe_update_beam(self, seg_dur_s: float, asr_latency_s: float) -> None:
        if not self._adaptive_beam_enabled:
            return
        now = time.time()
        new_beam, reason = self._beam_ctl.maybe_update(
            seg_qsize=int(self._seg_q.qsize()),
            last_latency_s=float(asr_latency_s),
            last_dur_s=float(seg_dur_s),
            now=now,
        )
        if reason:
            self._log_event({"type": "asr_beam_update", "beam": int(new_beam), "reason": reason, "ts": now})

    def _log_segment_event(
        self,
        seg: Segment,
        speaker: str,
        text: str,
        asr_latency_s: float,
        seg_dur_s: float,
        beam_to_use: int,
        removed: int,
    ) -> None:
        self._log_event({
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
        })

    def _update_utterances(self, seg: Segment, speaker: str, text: str) -> None:
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
