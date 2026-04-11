from __future__ import annotations

import time


class TelemetryMixin:
    def _poll_resources(self) -> None:
        if self._proc is None:
            self._set_label_text_if_changed(self.lbl_resources, "resources: n/a")
            return
        now = time.monotonic()
        if (now - self._last_cpu_poll_mono) < 0.8:
            return
        self._last_cpu_poll_mono = now
        try:
            self._cpu_pct = float(self._proc.cpu_percent(interval=None))
            mem = self._proc.memory_info()
            self._rss_mb = float(mem.rss) / (1024.0 * 1024.0)
            self._set_label_text_if_changed(
                self.lbl_resources,
                f"resources: cpu={self._cpu_pct:.0f}% rss={self._rss_mb:.0f}MB",
            )
        except Exception:
            self._set_label_text_if_changed(self.lbl_resources, "resources: n/a")

    def _tick_ui(self) -> None:
        meters = self.engine.get_meters()
        now_mono = time.monotonic()

        self._poll_resources()

        master = meters["master"]
        mrms = float(master["rms"])
        mlast = float(master["last_ts"])
        self._set_progress_if_changed(self.master_meter, self._rms_to_pct(mrms))
        self._set_label_text_if_changed(
            self.master_status,
            "active" if (now_mono - mlast) < 0.6 and mrms > 1e-4 else "silence",
        )

        drops = meters.get("drops", {})
        dropped_out = int(drops.get("dropped_out_blocks", 0))
        dropped_tap = int(drops.get("dropped_tap_blocks", 0))
        self._tap_dropped_total = dropped_tap

        drained = self.writer.drained_blocks()
        self._set_label_text_if_changed(
            self.lbl_drops,
            f"drops: out={dropped_out} tap={dropped_tap} drained={drained}",
        )

        if dropped_out > 0 or dropped_tap > 0:
            self._warn_throttle(f"Engine drops detected: out={dropped_out} tap={dropped_tap}", min_interval_s=2.0)

        srcs = meters.get("sources", {})
        desktop_any_active = False
        desktop_any_present = False

        for name, info in srcs.items():
            if name not in self.rows:
                self._add_row(name)
            row = self.rows[name]

            enabled = bool(info.get("enabled", True))
            if not enabled:
                self._set_progress_if_changed(row.meter, 0)
                self._set_label_text_if_changed(row.status, "muted")

            rms = float(info.get("rms", 0.0))
            last_ts = float(info.get("last_ts", 0.0))
            buf_frames = int(info.get("buffer_frames", 0))
            drop_in = int(info.get("dropped_in_frames", 0))
            miss_out = int(info.get("missing_out_frames", 0))
            delay_ms = float(info.get("delay_ms", 0.0))
            src_rate = int(info.get("src_rate", 0))

            if not row.delay_ms.hasFocus():
                self._set_line_edit_if_changed(row.delay_ms, str(int(round(delay_ms))))

            self._set_progress_if_changed(row.meter, self._rms_to_pct(rms))
            active = (now_mono - last_ts) < 0.6 and rms > 1e-4

            rate_warn = ""
            if src_rate and src_rate != self.fmt.sample_rate:
                rate_warn = f" SR={src_rate}!"

            state = "active" if active else "silence"
            self._set_label_text_if_changed(
                row.status,
                f"{state} buf={buf_frames} miss={miss_out} drop_in={drop_in} "
                f"delay={int(round(delay_ms))}ms{rate_warn}",
            )

            if str(name).startswith("desktop_audio"):
                desktop_any_present = True
                if rms > float(self._silence_eps):
                    desktop_any_active = True

        self._drain_asr_ui_events(limit=160 if not self._long_run_mode else 120)

        ok = (self._tap_dropped_total <= 0) and (self._seg_dropped_total <= 0) and (self._seg_skipped_total <= 0)
        status = "OK" if ok else "DROPS"
        self._set_label_text_if_changed(
            self.lbl_completeness,
            f"Completeness: {status} | tap_drop={self._tap_dropped_total} seg_drop={self._seg_dropped_total} "
            f"seg_skip={self._seg_skipped_total} | avg_lat={self._avg_latency_s:.2f}s p95={self._p95_latency_s:.2f}s "
            f"lag={self._lag_s:.2f}s",
        )

        if self._is_running() and desktop_any_present:
            if desktop_any_active:
                self._desktop_silence_since_mono = None
            else:
                if self._desktop_silence_since_mono is None:
                    self._desktop_silence_since_mono = now_mono
                else:
                    dur = float(now_mono - float(self._desktop_silence_since_mono))
                    if dur >= float(self._silence_alert_s):
                        self._warn_throttle(
                            f"desktop_audio silence for {dur:.1f}s (rms<{self._silence_eps})",
                            min_interval_s=4.0,
                        )

        if self._is_running() and self._asr_overload_active:
            self._set_label_text_if_changed(self.lbl_status, "running (ASR overload: degraded mode)")

    @staticmethod
    def _rms_to_pct(rms: float) -> int:
        x = float(rms)
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        pct = int((x ** 0.5) * 100.0)
        return max(0, min(100, pct))
