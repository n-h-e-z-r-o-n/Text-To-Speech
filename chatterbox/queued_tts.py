import argparse
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from main import (
    DEFAULT_OUTPUT_DIR,
    EXIT_WORDS,
    detect_device,
    load_model,
    maybe_play_audio,
    save_wav,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "chatterbox-turbo-model"
VOICE_PROMPT = "voice samples/indian female.mp3"


def split_text_into_chunks(text: str, max_chars: int, first_chunk_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    split_parts = re.split(r"(?<=[.!?,;:])\s+", normalized)
    chunks: list[str] = []
    current = ""
    current_limit = first_chunk_chars

    for part in split_parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > current_limit:
            words = part.split()
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= current_limit:
                    current = candidate
                    continue

                if current:
                    chunks.append(current)
                    current_limit = max_chars
                current = word
            continue

        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= current_limit:
            current = candidate
            continue

        chunks.append(current)
        current_limit = max_chars
        current = part

    if current:
        chunks.append(current)

    return chunks


def synthesize_chunk(model, text: str, output_path: Path, voice_prompt: Path) -> float:
    wav = model.generate(text, audio_prompt_path=str(voice_prompt))
    audio = wav.squeeze(0).detach().cpu().numpy()
    save_wav(output_path, model.sr, audio)
    return float(wav.shape[-1] / model.sr)


def generate_chunks(
    model,
    chunks: list[str],
    output_dir: Path,
    voice_prompt: Path,
    audio_queue: queue.Queue,
) -> None:
    for index, chunk in enumerate(chunks, start=1):
        output_path = output_dir / f"chunk_{index:03d}.wav"
        print(f"Generating chunk {index}/{len(chunks)}: {chunk}")
        duration = synthesize_chunk(model, chunk, output_path, voice_prompt)
        audio_queue.put((index, chunk, output_path, duration))

    audio_queue.put(None)


def play_chunks(audio_queue: queue.Queue, should_play: bool) -> None:
    while True:
        item = audio_queue.get()
        if item is None:
            return

        index, chunk, audio_path, duration = item
        print(f"Playing chunk {index}: {audio_path.name} ({duration:.2f}s)")
        if should_play:
            maybe_play_audio(audio_path, True)
        else:
            time.sleep(duration)

        print(f"Finished chunk {index}: {chunk}")


def run_queued_generation(
    model,
    text: str,
    output_root: Path,
    voice_prompt: Path,
    should_play: bool,
    max_chars: int,
    first_chunk_chars: int,
) -> int:
    chunks = split_text_into_chunks(
        text,
        max_chars=max_chars,
        first_chunk_chars=first_chunk_chars,
    )
    if not chunks:
        print("No text provided.", file=sys.stderr)
        return 1

    batch_dir = output_root / f"queued_reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using reference voice sample: {voice_prompt}")
    print(f"Created {len(chunks)} chunk(s). Output folder: {batch_dir}")

    audio_queue: queue.Queue = queue.Queue(maxsize=2)
    generator = threading.Thread(
        target=generate_chunks,
        args=(model, chunks, batch_dir, voice_prompt, audio_queue),
        daemon=True,
    )
    player = threading.Thread(
        target=play_chunks,
        args=(audio_queue, should_play),
        daemon=True,
    )

    started = time.time()
    generator.start()
    player.start()
    generator.join()
    player.join()
    elapsed = time.time() - started

    print(f"Completed {len(chunks)} chunk(s) in {elapsed:.2f}s")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and play chunked speech with overlapped queue-based synthesis."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to the local chatterbox-turbo-model folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated chunk wav files will be saved.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("cpu", "cuda", "xpu", "mps"),
        help="Torch device to use. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to convert to speech. If omitted, the script runs in a loop.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=140,
        help="Approximate maximum characters per chunk.",
    )
    parser.add_argument(
        "--first-chunk-chars",
        type=int,
        default=60,
        help="Use a smaller first chunk so audio starts sooner.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Generate chunks without local audio playback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    voice_prompt = (APP_DIR / VOICE_PROMPT).resolve()

    if not voice_prompt.exists():
        print(f"Voice prompt not found: {voice_prompt}", file=sys.stderr)
        return 1

    if args.chunk_chars < 20:
        print("--chunk-chars must be at least 20.", file=sys.stderr)
        return 1

    if args.first_chunk_chars < 20:
        print("--first-chunk-chars must be at least 20.", file=sys.stderr)
        return 1

    if args.first_chunk_chars > args.chunk_chars:
        print("--first-chunk-chars must be less than or equal to --chunk-chars.", file=sys.stderr)
        return 1

    try:
        device = args.device or detect_device()
        model = load_model(model_dir, device)
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    should_play = not args.no_play
    if args.text and args.text.strip():
        return run_queued_generation(
            model=model,
            text=args.text.strip(),
            output_root=output_dir,
            voice_prompt=voice_prompt,
            should_play=should_play,
            max_chars=args.chunk_chars,
            first_chunk_chars=args.first_chunk_chars,
        )

    print("Queued TTS is ready. Type text to test chunked generation.")
    print("Type 'exit', 'quit', or 'bye' to stop.")

    while True:
        try:
            text = input("Enter text to convert to speech: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0

        if not text:
            continue

        if text.lower() in EXIT_WORDS:
            print("Stopping.")
            return 0

        run_queued_generation(
            model=model,
            text=text,
            output_root=output_dir,
            voice_prompt=voice_prompt,
            should_play=should_play,
            max_chars=args.chunk_chars,
            first_chunk_chars=args.first_chunk_chars,
        )


if __name__ == "__main__":
    raise SystemExit(main())
