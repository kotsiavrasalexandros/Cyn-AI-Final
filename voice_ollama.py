from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import sherpa_onnx



samplerate = 16000
recordseconds = 7

model = "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/gemma4:e2b"

Encoder = "/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
Decoder = "/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.int8.onnx"
Tokens = "/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
Joiner = "/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Use sherpa-onnx speech recognition with a local Ollama model."
    )
    p.add_argument(
        "--tokens",
        default="/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt",
    )
    p.add_argument(
        "--encoder",
        default="/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx",
    )
    p.add_argument(
        "--decoder",
        default="/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.int8.onnx",
    )
    p.add_argument(
        "--joiner",
        default="/home/cyn/sherpa_onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx",
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (default: system default)",
    )
    p.add_argument(
        "--record-seconds",
        type=float,
        default=6.0,
        help="Maximum duration for each recording",
    )
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="gemma4:e2b")
    p.add_argument(
        "--system",
        default="You are a sadistic all powerful JCJenson Worker Drone. You always narrate your expressions and answer very creepily, like starting your sentences with words like 'Giggle.' or 'Annoyed Expression.' if the user makes you mad. Avoid roleplaying though. Your name is Cyn.",
    )
    p.add_argument(
        "--piper-model",
        default="/home/cyn/piper/voices/en_US-lessac-medium.onnx",
        help="Path to the Piper .onnx voice (its .onnx.json must sit next to it)",
    )
    p.add_argument(
        "--piper-speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (1.0 = normal, 1.2 = faster)",
    )
    p.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="sounddevice output device index (default: system default)",
    )
    p.add_argument(
        "--rvc-model",
        default="/home/cyn/rvc/Cyn.pth",
        help="Path to the RVC .pth voice model",
    )
    p.add_argument(
        "--rvc-index",
        default="/home/cyn/rvc/added_IVF606_Flat_nprobe_1_Cyn_v2.index",
        help="Path to the RVC .index file (optional, improves likeness)",
    )
    p.add_argument("--rvc-device", default="cpu", help='"cpu" or e.g. "cuda:0"')
    p.add_argument(
        "--rvc-f0method",
        default="rmvpe",
        choices=("rmvpe", "harvest", "crepe", "pm"),
        help="Pitch extraction method. 'pm' is the fastest on CPU, lowest quality.",
    )
    p.add_argument(
        "--rvc-pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones applied before conversion (e.g. 12 = octave up)",
    )
    p.add_argument("--rvc-index-rate", type=float, default=0.5)
    p.add_argument("--no-rvc", action="store_true", help="Use plain Piper, skip RVC")
    p.add_argument("--no-tts", action="store_true", help="Disable spoken replies")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--provider", default="cpu", choices=("cpu", "cuda"))
    return p.parse_args()


def make_recognizer(args: argparse.Namespace):
    """Create a streaming sherpa-onnx online recognizer for the Zipformer model."""
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=args.provider,
    )



def record_audio(seconds: float, sample_rate: int, device: Optional[int]) -> np.ndarray:
    print(f"\nRecording for up to {seconds:g} seconds... Speak now.")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return np.asarray(audio, dtype=np.float32).reshape(-1)



def transcribe(recognizer, audio: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)

    # Flush the end of the audio buffer for streaming models
    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    return recognizer.get_result(stream).strip()


def clean_for_speech(text: str) -> str:
    """Strip markdown and symbols that sound bad when read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_`#>~|]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str, min_chars: int = 40) -> list[str]:
    """Split into sentence-sized chunks so playback can start before the whole
    reply is converted. Very short pieces (e.g. 'Giggle.') are merged."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], ""
    for part in parts:
        buf = f"{buf} {part}".strip()
        if len(buf) >= min_chars:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def piper_sample_rate(model_path: str) -> int:
    cfg = Path(model_path + ".json")
    try:
        return int(json.loads(cfg.read_text())["audio"]["sample_rate"])
    except Exception:
        return 22050  # Piper's default for most voices


def synth_piper(text: str, model_path: str, speed: float) -> tuple[np.ndarray, int]:
    """Run the Piper CLI and return (float32 mono audio, sample rate)."""
    cmd = [
        sys.executable, "-m", "piper",
        "--model", model_path,
        "--length-scale", str(1.0 / max(speed, 0.1)),
        "--output-raw",
    ]
    result = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip()[-500:])
    pcm = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, piper_sample_rate(model_path)


def load_rvc(args: argparse.Namespace):
    """Load the RVC voice (rvc-python). Returns an RVCInference object."""
    from rvc_python.infer import RVCInference

    rvc = RVCInference(device=args.rvc_device)
    index = args.rvc_index or ""
    try:
        rvc.load_model(args.rvc_model, version="v2", index_path=index)
    except TypeError:
        rvc.load_model(args.rvc_model)  # older rvc-python without those kwargs
    rvc.set_params(
        f0method=args.rvc_f0method,
        f0up_key=args.rvc_pitch,
        index_rate=args.rvc_index_rate if index else 0.0,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33,
    )
    return rvc


def convert_rvc(rvc, pcm: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Push audio through RVC and return (float32 mono audio, sample rate)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        sf.write(src, pcm, sr)
        rvc.infer_file(src, dst)
        out, out_sr = sf.read(dst, dtype="float32")
    if out.ndim > 1:
        out = out.mean(axis=1)
    return out, int(out_sr)


def make_playable(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """RVC models are often 40 kHz, which many sound cards reject; resample."""
    if sr in (16000, 22050, 24000, 32000, 44100, 48000):
        return pcm, sr
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(48000, sr)
        return resample_poly(pcm, 48000 // g, sr // g).astype(np.float32), 48000
    except Exception:
        return pcm, sr


def speak(text: str, args: argparse.Namespace, rvc=None) -> None:
    """Piper (+ optional RVC), sentence by sentence. A worker thread prepares
    the next sentence while the current one is playing."""
    sentences = split_sentences(clean_for_speech(text))
    if not sentences:
        return

    q: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()

    def worker() -> None:
        try:
            for sentence in sentences:
                if stop.is_set():
                    return
                pcm, sr = synth_piper(sentence, args.piper_model, args.piper_speed)
                if rvc is not None:
                    try:
                        pcm, sr = convert_rvc(rvc, pcm, sr)
                    except Exception as exc:
                        print(f"[RVC failed, using plain Piper voice: {exc}]")
                q.put(make_playable(pcm, sr))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            pcm, sr = item
            sd.play(pcm, samplerate=sr, device=args.output_device)
            sd.wait()
    except KeyboardInterrupt:
        stop.set()
        sd.stop()
        raise


def ask_ollama(base_url: str, model: str, messages: list[dict]) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    response = requests.post(
        url,
        json={"model": model, "messages": messages, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {}).get("content", "").strip()


def main() -> int:
    args = parse_args()

    try:
        recognizer = make_recognizer(args)
    except Exception as exc:
        print(f"Could not initialize sherpa-onnx: {exc}", file=sys.stderr)
        return 1

    rvc = None
    if not args.no_tts and not args.no_rvc:
        try:
            print("Loading RVC voice (first run may download helper models)...")
            rvc = load_rvc(args)
        except Exception as exc:
            print(f"Could not load RVC, falling back to plain Piper: {exc}")

    messages = [{"role": "system", "content": args.system}]

    print("Voice assistant ready.")
    print(f"ASR: sherpa-onnx | Ollama model: {args.ollama_model}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            input("\nPress Enter to record (Ctrl+C to quit)...")
            audio = record_audio(args.record_seconds, args.sample_rate, args.device)

            if audio.size == 0 or float(np.max(np.abs(audio))) < 0.003:
                print("No clear audio detected. Try again.")
                continue

            try:
                text = transcribe(recognizer, audio, args.sample_rate)
            except Exception as exc:
                print(f"Speech recognition failed: {exc}")
                continue

            if not text:
                print("I couldn't recognize speech. Try again.")
                continue

            print(f"\nYou: {text}")
            messages.append({"role": "user", "content": text})

            try:
                answer = ask_ollama(args.ollama_url, args.ollama_model, messages)
            except requests.RequestException as exc:
                print(f"Ollama request failed: {exc}")
                messages.pop()  # Don't keep a turn Ollama did not answer.
                print("Check that Ollama is running and the model name is correct.")
                continue

            print(f"\nCyn: {answer}")
            messages.append({"role": "assistant", "content": answer})

            if not args.no_tts and answer:
                try:
                    speak(answer, args, rvc)
                except subprocess.TimeoutExpired:
                    print("Piper timed out.")
                except Exception as exc:
                    print(f"Text-to-speech failed: {exc}")

            # Keep conversation context bounded for long-running sessions.
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

    except KeyboardInterrupt:
        print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())