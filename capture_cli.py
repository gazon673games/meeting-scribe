# --- File: D:\work\own\voice2textTest\capture_cli.py ---
from __future__ import annotations

import queue
import signal
from pathlib import Path

import numpy as np

from audio.engine import AudioEngine
from audio.types import AudioFormat
from audio.filters.volume import VolumeFilter
from audio.sources.microphone import MicrophoneSource
from audio.sources.wasapi_loopback import WasapiLoopbackSource

try:
    import soundfile as sf
except ImportError:
    sf = None


def main() -> None:
    fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)
    out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)

    engine = AudioEngine(format=fmt, output_queue=out_q)

    sys_src = WasapiLoopbackSource(name="desktop_audio", format=fmt, device=None)
    sys_src.add_filter(VolumeFilter(gain=1.0))
    engine.add_source(sys_src)

    # mic_src = MicrophoneSource(
    #     name="mic",
    #     format=AudioFormat(sample_rate=48000, channels=1, dtype="float32", blocksize=1024),
    #     device=None,
    # )
    # mic_src.add_filter(VolumeFilter(gain=1.2))
    # engine.add_source(mic_src)

    engine.add_master_filter(VolumeFilter(gain=1.0))

    stop = {"flag": False}

    def _sig(*_) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    try:
        signal.signal(signal.SIGTERM, _sig)
    except Exception:
        pass

    wav_path = Path("capture_mix.wav")
    wav_file = None
    if sf is not None:
        wav_file = sf.SoundFile(
            wav_path,
            mode="w",
            samplerate=fmt.sample_rate,
            channels=fmt.channels,
            subtype="PCM_16",
        )
        print(f"[capture_cli] Writing to {wav_path.resolve()}")
    else:
        print("[capture_cli] soundfile not installed: WAV writing disabled (pip install soundfile)")

    engine.start()
    print("[capture_cli] Capturing... Ctrl+C to stop")

    blocks_written = 0
    warned_drop = False

    try:
        while not stop["flag"]:
            try:
                frame = out_q.get(timeout=0.2)
            except queue.Empty:
                meters = engine.get_meters()
                dropped_out = int(meters.get("drops", {}).get("dropped_out_blocks", 0))
                if dropped_out > 0 and not warned_drop:
                    print(f"[capture_cli] WARNING: output blocks dropped: {dropped_out}")
                    warned_drop = True
                continue

            if wav_file is not None:
                wav_file.write(frame)
                blocks_written += 1

    except KeyboardInterrupt:
        print("[capture_cli] Ctrl+C received, stopping...")

    finally:
        engine.stop()
        if wav_file is not None:
            wav_file.close()
        print(f"[capture_cli] Stopped. Blocks written: {blocks_written}")


if __name__ == "__main__":
    main()
