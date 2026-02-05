# --- File: D:\work\own\voice2textTest\audio\engine.py ---
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    dtype: str = "float32"
    blocksize: int = 1024


class AudioFilter(Protocol):
    def process(self, x: np.ndarray, fmt: AudioFormat) -> np.ndarray:
        ...


class AudioSource(Protocol):
    name: str

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_format(self) -> AudioFormat:
        ...

    def get_filters(self) -> List[AudioFilter]:
        ...


# ------------------ DSP helpers ------------------


class _LinearResampler:
    """
    Simple linear resampler with state.

    ratio = src_rate / dst_rate
    We produce dst samples by sampling src at fractional positions.
    Keeps last sample to make interpolation continuous across blocks.

    Not audiophile-quality but stable and correct for ASR/logging/metering.
    """

    def __init__(self, src_rate: int, dst_rate: int, channels: int):
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.channels = int(channels)

        self._ratio = self.src_rate / float(self.dst_rate)
        self._pos = 0.0  # fractional position in "src sample index space"
        self._prev = np.zeros((1, self.channels), dtype=np.float32)  # last src sample

    def reset(self) -> None:
        self._pos = 0.0
        self._prev[...] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        x: (n, ch) float32 at src_rate
        returns y: (~n * dst/src, ch) float32 at dst_rate
        """
        if x.size == 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        x = x.astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.channels:
            # be strict; channel mapping should happen outside
            x = x[:, : self.channels]

        # prepend prev for continuous interpolation
        xin = np.vstack([self._prev, x])  # shape (n+1, ch)
        n_in = xin.shape[0]

        # estimate output length for this input chunk
        # we will generate while (pos + 1) < n_in
        # pos is in xin index space
        # pos advances by ratio per output sample
        # worst-case approximate
        max_out = int((n_in - 1 - self._pos) / self._ratio) + 2
        if max_out < 0:
            max_out = 0

        out = np.zeros((max_out, self.channels), dtype=np.float32)
        out_i = 0
        pos = self._pos

        while (pos + 1.0) < (n_in - 1e-9) and out_i < max_out:
            i0 = int(pos)
            frac = float(pos - i0)
            s0 = xin[i0]
            s1 = xin[i0 + 1]
            out[out_i] = s0 + (s1 - s0) * frac
            out_i += 1
            pos += self._ratio

        # keep remaining position relative to new "prev"
        # shift pos back by (n_in - 1) because we will keep last sample as prev
        pos -= (n_in - 1)
        self._pos = pos
        self._prev = xin[-1:, :].copy()  # last sample for next chunk

        return out[:out_i]


class _RingBuffer:
    """
    Ring buffer storing float32 samples as frames (n, ch), with exact reads.

    Writes append; reads pull exactly `n` frames; if insufficient => returns (n,ch) with zeros padding.
    """

    def __init__(self, channels: int, max_frames: int):
        self.channels = int(channels)
        self.max_frames = int(max_frames)

        self._buf = np.zeros((self.max_frames, self.channels), dtype=np.float32)
        self._size = 0  # number of valid frames in buffer
        self._r = 0  # read index
        self._w = 0  # write index

    def clear(self) -> None:
        self._size = 0
        self._r = 0
        self._w = 0

    def size(self) -> int:
        return int(self._size)

    def write(self, x: np.ndarray) -> int:
        """
        Write frames; if overflow => drops oldest frames.
        Returns dropped frames count (>=0).
        """
        if x.size == 0:
            return 0

        x = x.astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.channels:
            # strict: channel mapping must happen before writing
            x = x[:, : self.channels]

        n = int(x.shape[0])
        dropped = 0

        if n >= self.max_frames:
            # keep only tail
            x = x[-self.max_frames :]
            n = self.max_frames
            self.clear()
        else:
            # ensure space: drop oldest if needed
            overflow = max(0, (self._size + n) - self.max_frames)
            if overflow > 0:
                dropped = overflow
                self._r = (self._r + overflow) % self.max_frames
                self._size -= overflow

        # write in two chunks if wrap
        first = min(n, self.max_frames - self._w)
        self._buf[self._w : self._w + first] = x[:first]
        rest = n - first
        if rest > 0:
            self._buf[0:rest] = x[first:first + rest]
        self._w = (self._w + n) % self.max_frames
        self._size += n

        return dropped

    def read_exact(self, n: int) -> Tuple[np.ndarray, int]:
        """
        Read exactly n frames. If insufficient => pad with zeros.
        Returns (frames, missing_frames).
        """
        n = int(n)
        if n <= 0:
            return np.zeros((0, self.channels), dtype=np.float32), 0

        out = np.zeros((n, self.channels), dtype=np.float32)
        missing = 0

        take = min(n, self._size)
        if take > 0:
            first = min(take, self.max_frames - self._r)
            out[:first] = self._buf[self._r : self._r + first]
            rest = take - first
            if rest > 0:
                out[first:first + rest] = self._buf[0:rest]
            self._r = (self._r + take) % self.max_frames
            self._size -= take

        if take < n:
            missing = n - take

        return out, missing


def _soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """
    Soft clipper using tanh, keeps signal in [-1,1] smoothly.
    """
    x = x.astype(np.float32, copy=False)
    if drive <= 0.0:
        return np.clip(x, -1.0, 1.0)
    return np.tanh(x * float(drive)).astype(np.float32, copy=False)


def _gcc_phat_delay(a: np.ndarray, b: np.ndarray, sr: int, max_delay_ms: float = 200.0) -> int:
    """
    Estimate delay (in samples) between a and b using GCC-PHAT.
    Positive result means: b is delayed relative to a (b lags).
    Inputs should be 1D float32 arrays.
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size == 0 or b.size == 0:
        return 0

    n = int(1 << (int(np.ceil(np.log2(a.size + b.size))) ))
    A = np.fft.rfft(a, n=n)
    B = np.fft.rfft(b, n=n)

    R = A * np.conj(B)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom

    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-(n // 2) :], cc[: (n // 2)]))

    max_shift = int((max_delay_ms / 1000.0) * sr)
    mid = len(cc) // 2
    lo = max(0, mid - max_shift)
    hi = min(len(cc), mid + max_shift + 1)

    shift = int(np.argmax(cc[lo:hi]) + lo - mid)
    return shift


# ------------------ Engine state ------------------


@dataclass
class _SourceState:
    src: AudioSource
    enabled: bool = True

    # input pipeline config
    src_fmt: AudioFormat = None  # type: ignore[assignment]
    need_resample: bool = False

    # resampler (src_rate -> eng_rate) after per-source filters
    resampler: Optional[_LinearResampler] = None

    # sample ring buffer in ENGINE rate/channels
    ring: _RingBuffer = None  # type: ignore[assignment]
    ring_lock: threading.Lock = None  # type: ignore[assignment]

    # delay line in ENGINE samples (applied after ring read)
    delay_samples: int = 0
    delay_ring: _RingBuffer = None  # type: ignore[assignment]

    # meters + stats (owned by engine lock)
    rms: float = 0.0
    last_ts: float = 0.0
    dropped_in_frames: int = 0
    missing_out_frames: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ring_lock", threading.Lock())


class AudioEngine:
    """
    More robust real-time pipeline:
      - Source callback: copy float32, apply per-source filters, resample if needed,
        channel-map to engine channels, write into per-source sample ring buffer.
      - Mixer thread (master clock): every tick read_exact(blocksize) from each enabled source,
        apply optional delay line, sum, master filters, soft clip, output queue.

    Adds:
      - Sample-level ring buffers (no jitter/burst issues)
      - Resampling (linear) if source SR differs
      - Optional delay compensation and auto-sync (GCC-PHAT)
      - Thread-safe meters snapshot
    """

    def __init__(
        self,
        format: AudioFormat,
        output_queue: "queue.Queue[np.ndarray]",
        *,
        max_source_buffer_seconds: float = 2.0,
        soft_clip_drive: float = 1.5,
        headroom: float = 0.90,
    ):
        self._fmt = format
        self._out_q = output_queue

        self._max_buf_frames = max(1, int(max_source_buffer_seconds * self._fmt.sample_rate))
        self._soft_clip_drive = float(soft_clip_drive)
        self._headroom = float(headroom)

        self._sources: List[AudioSource] = []
        self._state: Dict[str, _SourceState] = {}
        self._master_filters: List[AudioFilter] = []

        self._lock = threading.RLock()
        self._running = False

        # mixer thread
        self._mix_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # master meters/stats
        self._master_rms: float = 0.0
        self._master_last_ts: float = 0.0
        self._dropped_out_blocks: int = 0

        # auto-sync
        self._autosync_enabled = False
        self._autosync_ref: Optional[str] = None
        self._autosync_target: Optional[str] = None
        self._autosync_window_s: float = 1.0
        self._autosync_interval_s: float = 2.0
        self._autosync_last_run: float = 0.0
        self._autosync_max_delay_ms: float = 200.0

    @property
    def format(self) -> AudioFormat:
        return self._fmt

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def add_source(self, src: AudioSource) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot add sources while engine is running (MVP).")
            if src.name in self._state:
                raise ValueError(f"Source '{src.name}' already exists")

            st = _SourceState(src=src)
            st.src_fmt = src.get_format()

            # resampler configured on source SR vs engine SR
            st.need_resample = (st.src_fmt.sample_rate != self._fmt.sample_rate)
            st.resampler = None
            if st.need_resample:
                st.resampler = _LinearResampler(
                    src_rate=st.src_fmt.sample_rate,
                    dst_rate=self._fmt.sample_rate,
                    channels=st.src_fmt.channels,
                )

            # ring buffers in engine channels (after channel-map)
            st.ring = _RingBuffer(channels=self._fmt.channels, max_frames=self._max_buf_frames)
            st.delay_ring = _RingBuffer(channels=self._fmt.channels, max_frames=self._max_buf_frames)

            self._sources.append(src)
            self._state[src.name] = st

    def add_master_filter(self, flt: AudioFilter) -> None:
        with self._lock:
            self._master_filters.append(flt)

    def set_source_enabled(self, name: str, enabled: bool) -> None:
        with self._lock:
            st = self._state.get(name)
            if st is None:
                return
            st.enabled = bool(enabled)

    def set_source_delay_ms(self, name: str, delay_ms: float) -> None:
        """
        Positive delay means: add extra delay to this source (it will be pushed later).
        Useful to align mic with desktop, etc.
        """
        with self._lock:
            st = self._state.get(name)
            if st is None:
                return
            ds = int(round((float(delay_ms) / 1000.0) * self._fmt.sample_rate))
            st.delay_samples = max(0, ds)
            st.delay_ring.clear()

    def enable_auto_sync(
        self,
        *,
        reference_source: str,
        target_source: str,
        window_seconds: float = 1.0,
        interval_seconds: float = 2.0,
        max_delay_ms: float = 200.0,
    ) -> None:
        """
        Periodically estimates delay between reference and target and updates target delay_samples.
        Typical usage:
          enable_auto_sync(reference_source="desktop_audio", target_source="mic")
        """
        with self._lock:
            if reference_source not in self._state:
                raise ValueError(f"Unknown reference source '{reference_source}'")
            if target_source not in self._state:
                raise ValueError(f"Unknown target source '{target_source}'")

            self._autosync_enabled = True
            self._autosync_ref = reference_source
            self._autosync_target = target_source
            self._autosync_window_s = float(window_seconds)
            self._autosync_interval_s = float(interval_seconds)
            self._autosync_max_delay_ms = float(max_delay_ms)
            self._autosync_last_run = 0.0

    def disable_auto_sync(self) -> None:
        with self._lock:
            self._autosync_enabled = False
            self._autosync_ref = None
            self._autosync_target = None

    def get_meters(self) -> dict:
        with self._lock:
            return {
                "sources": {
                    n: {
                        "enabled": bool(st.enabled),
                        "rms": float(st.rms),
                        "last_ts": float(st.last_ts),
                        "buffer_frames": int(st.ring.size()),
                        "dropped_in_frames": int(st.dropped_in_frames),
                        "missing_out_frames": int(st.missing_out_frames),
                        "delay_ms": float(st.delay_samples) * 1000.0 / float(self._fmt.sample_rate),
                        "src_rate": int(st.src_fmt.sample_rate),
                    }
                    for n, st in self._state.items()
                },
                "master": {
                    "rms": float(self._master_rms),
                    "last_ts": float(self._master_last_ts),
                },
                "drops": {
                    "dropped_out_blocks": int(self._dropped_out_blocks),
                },
                "autosync": {
                    "enabled": bool(self._autosync_enabled),
                    "ref": self._autosync_ref,
                    "target": self._autosync_target,
                },
            }

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._sources:
                raise RuntimeError("No sources added")
            self._running = True
            self._stop_evt.clear()

        for src in self._sources:
            src.start(self._on_audio_from_source)

        self._mix_thread = threading.Thread(
            target=self._mix_loop,
            name="audio-mixer",
            daemon=True,
        )
        self._mix_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_evt.set()

        for src in self._sources:
            try:
                src.stop()
            except Exception:
                pass

        if self._mix_thread is not None:
            self._mix_thread.join(timeout=2.0)
            self._mix_thread = None

        with self._lock:
            for st in self._state.values():
                with st.ring_lock:
                    st.ring.clear()
                    st.delay_ring.clear()
                if st.resampler is not None:
                    st.resampler.reset()
                st.rms = 0.0
                st.last_ts = 0.0
                st.dropped_in_frames = 0
                st.missing_out_frames = 0
            self._master_rms = 0.0
            self._master_last_ts = 0.0
            self._dropped_out_blocks = 0
            self._autosync_last_run = 0.0

    # ---------------- internals ----------------

    @staticmethod
    def _apply_filters(x: np.ndarray, fmt: AudioFormat, filters: List[AudioFilter]) -> np.ndarray:
        y = x
        for f in filters:
            y = f.process(y, fmt)
        return y

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        xf = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(xf * xf)))

    @staticmethod
    def _channel_map(x: np.ndarray, src_ch: int, dst_ch: int) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        if src_ch == dst_ch:
            return x.astype(np.float32, copy=False)

        if src_ch == 1 and dst_ch == 2:
            return np.repeat(x, 2, axis=1).astype(np.float32, copy=False)

        if src_ch == 2 and dst_ch == 1:
            return x.mean(axis=1, keepdims=True).astype(np.float32, copy=False)

        # crop/pad
        n = x.shape[0]
        if x.shape[1] > dst_ch:
            return x[:, :dst_ch].astype(np.float32, copy=False)
        out = np.zeros((n, dst_ch), dtype=np.float32)
        out[:, : x.shape[1]] = x.astype(np.float32, copy=False)
        return out

    def _on_audio_from_source(self, source_name: str, frame: np.ndarray) -> None:
        """
        Source callback thread:
          - minimal checks
          - copy to float32
          - per-source filters
          - resample if needed
          - channel map to engine channels
          - write into per-source sample ring buffer
        """
        with self._lock:
            if not self._running:
                return
            st = self._state.get(source_name)
            if st is None:
                return
            if not st.enabled:
                # still drain to keep ring stable? we skip writing => treated as silence
                return
            src_filters = st.src.get_filters()
            src_fmt = st.src_fmt
            resampler = st.resampler

        x = np.asarray(frame)
        if x.ndim == 1:
            x = x[:, None]
        x = np.array(x, dtype=np.float32, copy=True)

        # normalize source channels to src_fmt.channels if mismatch
        if x.shape[1] != src_fmt.channels:
            x = self._channel_map(x, x.shape[1], src_fmt.channels)

        # align length to something sane (no hard requirement here)
        try:
            if src_filters:
                x = self._apply_filters(x, src_fmt, src_filters)
        except Exception:
            # on filter failure treat as silence
            x = np.zeros((0, src_fmt.channels), dtype=np.float32)

        # resample to engine rate if needed
        if resampler is not None:
            try:
                x = resampler.process(x)
            except Exception:
                x = np.zeros((0, src_fmt.channels), dtype=np.float32)

        # channel-map to engine channels
        x = self._channel_map(x, x.shape[1], self._fmt.channels)

        # write to ring
        with st.ring_lock:
            dropped = st.ring.write(x)

        if dropped > 0:
            with self._lock:
                st.dropped_in_frames += int(dropped)

    def _read_with_delay(self, st: _SourceState, n: int) -> Tuple[np.ndarray, int]:
        """
        Read n frames from st.ring, then apply st.delay_samples via delay_ring.
        Returns (frames, missing_frames_from_source_ring).
        """
        with st.ring_lock:
            block, missing = st.ring.read_exact(n)

        # apply delay: push into delay_ring and read out delayed samples
        ds = int(st.delay_samples)
        if ds <= 0:
            return block, missing

        # write current block into delay line
        with st.ring_lock:
            st.delay_ring.write(block)
            # read out n frames, but delay means we should output older frames;
            # to implement delay, we ensure delay_ring starts with ds zeros.
            # easiest: if delay_ring size < ds + n, pad by writing zeros.
            need = (ds + n) - st.delay_ring.size()
            if need > 0:
                st.delay_ring.write(np.zeros((need, self._fmt.channels), dtype=np.float32))
            out, _ = st.delay_ring.read_exact(n)

        return out, missing

    def _maybe_autosync(self, now_mono_ts: float) -> None:
        with self._lock:
            if not self._autosync_enabled:
                return
            if self._autosync_ref is None or self._autosync_target is None:
                return
            if (now_mono_ts - self._autosync_last_run) < self._autosync_interval_s:
                return

            ref = self._state.get(self._autosync_ref)
            tgt = self._state.get(self._autosync_target)
            if ref is None or tgt is None:
                return

            window = int(max(0.2, self._autosync_window_s) * self._fmt.sample_rate)
            window = min(window, self._max_buf_frames)  # bounded by ring size
            self._autosync_last_run = now_mono_ts

        # Snapshot recent audio from rings (best-effort)
        # We can't "peek" easily without copying internals; simplest: read a window by temporarily reading then re-writing.
        # To keep it simple and safe: we just skip if buffer too small.
        def _snapshot_last(st: _SourceState, n: int) -> Optional[np.ndarray]:
            with st.ring_lock:
                if st.ring.size() < n:
                    return None
                # hack: read and immediately write back (may reorder slightly but acceptable at autosync interval)
                tmp, _ = st.ring.read_exact(n)
                st.ring.write(tmp)
            return tmp

        ref_blk = _snapshot_last(ref, window)
        tgt_blk = _snapshot_last(tgt, window)
        if ref_blk is None or tgt_blk is None:
            return

        # mono for correlation
        ref_mono = ref_blk.mean(axis=1)
        tgt_mono = tgt_blk.mean(axis=1)

        shift = _gcc_phat_delay(ref_mono, tgt_mono, sr=self._fmt.sample_rate, max_delay_ms=self._autosync_max_delay_ms)

        # Interpretation:
        # shift > 0 means tgt lags behind ref by shift samples -> we need to DELAY ref or ADVANCE tgt.
        # We can only add delay, not advance. So if tgt lags, do nothing (or increase ref delay).
        # If tgt leads (shift < 0), we can delay tgt by -shift.
        with self._lock:
            if shift < 0:
                tgt.delay_samples = max(0, int(-shift))
                tgt.delay_ring.clear()

    def _mix_loop(self) -> None:
        fmt = self._fmt
        period_s = fmt.blocksize / float(fmt.sample_rate)
        next_t = time.monotonic()

        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t += period_s

            # autosync (best-effort)
            self._maybe_autosync(now)

            with self._lock:
                if not self._running:
                    break
                items: List[Tuple[str, _SourceState]] = list(self._state.items())
                master_filters = list(self._master_filters)

            mix = np.zeros((fmt.blocksize, fmt.channels), dtype=np.float32)
            active = 0
            ts = time.monotonic()

            # mix
            for name, st in items:
                if not st.enabled:
                    continue

                block, missing = self._read_with_delay(st, fmt.blocksize)

                # meters/stats updated under lock for consistent snapshot
                with self._lock:
                    st.rms = self._rms(block)
                    st.last_ts = ts
                    st.missing_out_frames += int(missing)

                mix += block
                active += 1

            # headroom
            if active > 0:
                mix *= float(self._headroom)

            # master filters
            try:
                mixed = self._apply_filters(mix, fmt, master_filters)
            except Exception:
                mixed = mix

            # soft clip (safer than hard clip)
            mixed = _soft_clip(mixed, drive=self._soft_clip_drive)

            # master meters
            with self._lock:
                self._master_rms = self._rms(mixed)
                self._master_last_ts = ts

            # output
            try:
                self._out_q.put_nowait(mixed)
            except queue.Full:
                with self._lock:
                    self._dropped_out_blocks += 1
