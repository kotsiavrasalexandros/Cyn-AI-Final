from __future__ import annotations

import argparse
import sys
import time
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

            # Keep conversation context bounded for long-running sessions.
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

    except KeyboardInterrupt:
        print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())