import argparse
import importlib
import importlib.util
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "chatterbox-turbo-model"
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"
VOICE_PROMPT = "voice samples/indian female.mp3"
EXIT_WORDS = {"exit", "quit", "bye"}
SUPPORTED_PYTHON = {(3, 11), (3, 12)}


def ensure_runtime_modules():
    version = sys.version_info[:2]
    if version not in SUPPORTED_PYTHON:
        supported = ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_PYTHON))
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            f"Unsupported Python runtime: {current}. "
            f"Use Python {supported} for local Chatterbox Turbo runs."
        )

    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Missing required package: torch.")

    if importlib.util.find_spec("chatterbox") is None:
        raise RuntimeError("Missing required package: chatterbox-tts.")

    torch = importlib.import_module("torch")
    perth = importlib.import_module("perth")
    chatterbox_turbo = importlib.import_module("chatterbox.tts_turbo")

    # Some Perth installs only expose the dummy implementation.
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

    return torch, chatterbox_turbo.ChatterboxTurboTTS


def detect_device() -> str:
    torch, _ = ensure_runtime_modules()

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def validate_model_dir(model_dir: Path) -> None:
    required_files = (
        "ve.safetensors",
        "t3_turbo_v1.safetensors",
        "s3gen_meanflow.safetensors",
        "tokenizer_config.json",
    )
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model files in '{model_dir}': {', '.join(missing)}"
        )


def load_model(model_dir: Path, device: str):
    _torch, chatterbox_tts = ensure_runtime_modules()
    validate_model_dir(model_dir)

    print(f"Loading Chatterbox Turbo from: {model_dir}")
    print(f"Using device: {device}")
    return chatterbox_tts.from_local(model_dir, device=device)


def synthesize_to_file(model, text: str, output_dir: Path, voice_prompt: Path | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"chatterbox_reply_{timestamp}.wav"

    wav = model.generate(
        text,
        audio_prompt_path=str(voice_prompt) if voice_prompt else None,
    )
    audio = wav.squeeze(0).detach().cpu().numpy()
    save_wav(output_path, model.sr, audio)
    return output_path


def save_wav(output_path: Path, sample_rate: int, audio) -> None:
    numpy = importlib.import_module("numpy")
    clipped = numpy.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(numpy.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def maybe_play_audio(audio_path: Path, should_play: bool) -> None:
    if not should_play:
        return

    if winsound is None:
        print("Playback skipped because winsound is not available on this platform.")
        return

    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speech from text with a local Chatterbox Turbo model."
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
        help="Directory where generated wav files will be saved.",
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
        help="Text to convert to speech. If omitted, the script will prompt once.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip local audio playback after synthesis.",
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

    try:
        device = args.device or detect_device()
        model = load_model(model_dir, device)
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    play_audio = not args.no_play
    if args.text and args.text.strip():
        started = time.time()
        audio_path = synthesize_to_file(model, args.text.strip(), output_dir, voice_prompt)
        print(f"Saved audio to: {audio_path}")
        print(f"Synthesis finished in {time.time() - started:.2f}s")
        maybe_play_audio(audio_path, play_audio)
        return 0

    print(f"Using reference voice sample: {voice_prompt}")
    print("Enter text to generate speech. Type 'exit', 'quit', or 'bye' to stop.")

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

        started = time.time()
        audio_path = synthesize_to_file(model, text, output_dir, voice_prompt)
        print(f"Saved audio to: {audio_path}")
        print(f"Synthesis finished in {time.time() - started:.2f}s")
        maybe_play_audio(audio_path, play_audio)


if __name__ == "__main__":
    raise SystemExit(main())
