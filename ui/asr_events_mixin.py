from __future__ import annotations

import queue
import time


class AsrEventsMixin:
    def _drain_asr_ui_events(self, limit: int = 140) -> None:
        n = 0
        while n < limit:
            try:
                ev = self.asr_ui_q.get_nowait()
            except queue.Empty:
                break
            n += 1

            typ = str(ev.get("type", ""))
            ts = float(ev.get("ts", time.time()))
            tss = self._fmt_ts(ts)

            lite = bool(self._long_run_mode)

            if typ == "utterance":
                text = (ev.get("text") or "").strip()
                if not text:
                    continue
                stream = str(ev.get("stream", ""))
                overload = bool(ev.get("overload", False))
                if overload:
                    self._asr_overload_active = True
                self._append_transcript_line(f"[{tss}] {stream}: {text}")

            elif typ == "source_error":
                source = str(ev.get("source", ""))
                error = str(ev.get("error", ""))
                self._append_transcript_line(f"[{tss}] SOURCE ERROR {source}: {error}")
                self._set_status(f"Audio source failed: {source}")

            elif typ == "asr_overload":
                active = bool(ev.get("active", False))
                reason = str(ev.get("reason", ""))
                qsz = ev.get("seg_qsize", None)
                beam = ev.get("beam_cur", None)
                lag = ev.get("lag_s", None)
                if bool(active):
                    self._asr_overload_active = True
                    self._append_transcript_line(f"[{tss}] OVERLOAD: {reason} q={qsz} beam={beam} lag={lag}")
                else:
                    self._asr_overload_active = False
                    self._append_transcript_line(f"[{tss}] OVERLOAD OFF: {reason} q={qsz}")

            elif typ == "segment_dropped":
                stream = str(ev.get("stream", ""))
                reason = str(ev.get("reason", ""))
                qsz = ev.get("seg_qsize", None)
                self._warn_throttle(f"ASR dropped segment ({stream}) reason={reason} q={qsz}")

            elif typ == "segment_skipped_overload":
                cnt = int(ev.get("count", 0))
                qsz = ev.get("seg_qsize", None)
                self._warn_throttle(f"ASR skipped {cnt} old segments due to overload (q={qsz})")

            elif typ == "asr_metrics":
                self._seg_dropped_total = int(ev.get("seg_dropped_total", self._seg_dropped_total))
                self._seg_skipped_total = int(ev.get("seg_skipped_total", self._seg_skipped_total))
                self._avg_latency_s = float(ev.get("avg_latency_s", self._avg_latency_s))
                self._p95_latency_s = float(ev.get("p95_latency_s", self._p95_latency_s))
                self._lag_s = float(ev.get("lag_s", self._lag_s))

            elif typ in ("segment", "segment_ready", "diar_debug", "asr_beam_update"):
                if lite:
                    continue

            elif typ == "asr_init_start":
                if lite:
                    continue
                model = ev.get("model", "")
                device = ev.get("device", "")
                self._append_transcript_line(f"[{tss}] ASR init start ({model}, {device})")

            elif typ == "asr_started":
                model = ev.get("model", "")
                mode = ev.get("mode", "")
                lang = ev.get("language", "")
                osx = ev.get("overload_strategy", "")
                self._append_transcript_line(f"[{tss}] ASR started (lang={lang}, mode={mode}, model={model}, overload={osx})")

            elif typ == "asr_init_ok":
                if lite:
                    continue
                model = ev.get("model", "")
                self._append_transcript_line(f"[{tss}] ASR init OK ({model})")

            elif typ == "error":
                where = ev.get("where", "")
                err = ev.get("error", "")
                self._append_transcript_line(f"[{tss}] ERROR {where}: {err}")

            elif typ == "asr_stopped":
                self._append_transcript_line(f"[{tss}] ASR stopped")

            else:
                continue
