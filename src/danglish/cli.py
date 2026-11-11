"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .backends import create_backend
from .bench import run_benchmark
from .config import COMPUTE_TYPES, AppConfig, load_config
from .doctor import build_checks, overall_status, render_checks
from .errors import DanglishError
from .memory import DEFAULT_BUDGET_MB, current_rss_mb
from .models import (
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUANTIZATION,
    ensure_model,
    model_is_ready,
)
from .pipeline import StreamingPipeline, translate_document

MOCK_NOTE = "mock backend: translations are fake (word order reversed)"


def _add_model_options(parser: argparse.ArgumentParser, *, mock: bool = True) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help="TOML configuration file (default: ./danglish.toml if present)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        metavar="DIR",
        help="converted CTranslate2 model directory",
    )
    parser.add_argument(
        "--compute-type",
        choices=COMPUTE_TYPES,
        help="CTranslate2 compute type (default: int8)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        metavar="N",
        help="inference threads (default: auto, capped at 4)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        metavar="N",
        help="beam size (default: 2; use 1 for lower latency)",
    )
    if mock:
        parser.add_argument(
            "--mock",
            action="store_true",
            help="use a deterministic fake backend (no model needed)",
        )


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    config = load_config(getattr(args, "config", None))
    if getattr(args, "model_path", None) is not None:
        config.model.path = args.model_path
    if getattr(args, "compute_type", None) is not None:
        config.model.compute_type = args.compute_type
    if getattr(args, "threads", None) is not None:
        config.model.intra_threads = args.threads
    if getattr(args, "beam_size", None) is not None:
        config.model.beam_size = args.beam_size
    config.validate()
    return config


def _require_model(config: AppConfig, *, mock: bool) -> None:
    if not mock and not model_is_ready(config.model.path):
        raise DanglishError(
            f"model not found at {config.model.path}. Run `danglish fetch` first, "
            "pass --model-path, or use --mock for a no-model demo."
        )


def _read_translate_input(args: argparse.Namespace) -> str:
    if args.file is not None and args.text:
        raise DanglishError("pass either text or --file, not both")

    if args.file is not None:
        if str(args.file) == "-":
            return sys.stdin.read()
        try:
            return args.file.read_text(encoding="utf-8")
        except OSError as exc:
            raise DanglishError(f"could not read {args.file}: {exc}") from exc

    if args.text:
        if args.text == ["-"]:
            return sys.stdin.read()
        return " ".join(args.text)

    if not sys.stdin.isatty():
        return sys.stdin.read()

    raise DanglishError("no input: pass text, use --file, or pipe text on stdin")


def _emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def cmd_translate(args: argparse.Namespace) -> int:
    text = _read_translate_input(args)
    if not text.strip():
        raise DanglishError("input is empty")

    config = _config_from_args(args)
    _require_model(config, mock=args.mock)
    backend = create_backend(config, mock=args.mock)
    try:
        translation = translate_document(backend, text)
    finally:
        backend.close()

    if args.json:
        _emit_json(
            {
                "source": text,
                "translation": translation,
                "model": "mock" if args.mock else str(config.model.path),
            }
        )
    else:
        print(translation)
    return 0


def _print_event(event: object, *, as_json: bool) -> None:
    if as_json:
        _emit_json(event.to_dict())  # type: ignore[attr-defined]
        return
    print(f"da  {event.source}")  # type: ignore[attr-defined]
    print(f"en  {event.target}")  # type: ignore[attr-defined]
    print()


def _transcript_line(pipeline: StreamingPipeline, line: str, *, as_json: bool) -> None:
    events = pipeline.push(line + " ")
    events.extend(pipeline.flush())
    for event in events:
        _print_event(event, as_json=as_json)


def cmd_repl(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _require_model(config, mock=args.mock)
    if args.mock:
        print(MOCK_NOTE, file=sys.stderr)

    backend = create_backend(config, mock=args.mock)
    pipeline = StreamingPipeline(
        backend,
        min_partial_chars=config.stream.min_partial_chars,
        max_pending_chars=config.stream.max_pending_chars,
    )
    try:
        if sys.stdin.isatty():
            print("Danglish REPL — type Danish, Ctrl-D to quit.", file=sys.stderr)
            while True:
                try:
                    line = input("da> ")
                except EOFError:
                    break
                if line.strip() in {"/quit", "/exit"}:
                    break
                if line.strip():
                    _transcript_line(pipeline, line, as_json=args.json)
        else:
            for line in sys.stdin:
                line = line.rstrip("\n")
                if line.strip():
                    _transcript_line(pipeline, line, as_json=args.json)
    finally:
        backend.close()
    return 0


def _clear_line() -> None:
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def cmd_stream(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _require_model(config, mock=args.mock)
    if args.mock:
        print(MOCK_NOTE, file=sys.stderr)

    debounce = (
        args.debounce_ms if args.debounce_ms is not None else config.stream.partial_debounce_ms
    ) / 1000.0
    show_partials = not args.no_partial and not args.json and sys.stdout.isatty()
    if sys.stdin.isatty() and not args.json:
        print(
            "Reading stdin. Terminal input is line-buffered; pipe a stream "
            "(e.g. `cat text.txt | danglish stream`) for live partials.",
            file=sys.stderr,
        )

    backend = create_backend(config, mock=args.mock)
    pipeline = StreamingPipeline(
        backend,
        min_partial_chars=config.stream.min_partial_chars,
        max_pending_chars=config.stream.max_pending_chars,
    )
    partial_visible = False
    last_preview = 0.0

    def print_final(event: object) -> None:
        nonlocal partial_visible
        if partial_visible:
            _clear_line()
            partial_visible = False
        _print_event(event, as_json=args.json)

    try:
        while True:
            char = sys.stdin.read(args.chunk_chars)
            if not char:
                break
            for event in pipeline.push(char):
                print_final(event)

            if not show_partials:
                continue
            if len(pipeline.pending) < config.stream.min_partial_chars:
                continue
            now = time.monotonic()
            if now - last_preview < debounce:
                continue
            event = pipeline.preview()
            if event is not None:
                _clear_line()
                sys.stdout.write(f"… {event.target}")
                sys.stdout.flush()
                partial_visible = True
                last_preview = now

        for event in pipeline.flush():
            print_final(event)
        if partial_visible:
            _clear_line()
    finally:
        backend.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _require_model(config, mock=args.mock)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    config.validate()

    try:
        from .server.app import run_server
    except ImportError as exc:
        raise DanglishError(
            "the web server needs the optional server dependencies.\n"
            "  uv:  uv sync --extra server\n"
            '  pip: pip install "danglish[server]"'
        ) from exc

    run_server(config, mock=args.mock)
    return 0


def cmd_talk(args: argparse.Namespace) -> int:
    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise DanglishError(
                "audio support is optional; install it with `uv sync --extra speech`"
            ) from exc
        print(sd.query_devices())
        return 0

    config = _config_from_args(args)
    _require_model(config, mock=args.mock)

    try:
        from .audio.mic import run_talk
    except ImportError as exc:
        raise DanglishError(
            "microphone-to-speaker mode needs the optional speech dependencies.\n"
            "  uv:  uv sync --extra speech\n"
            '  pip: pip install "danglish[speech]"'
        ) from exc

    backend = create_backend(config, mock=args.mock)
    try:
        run_talk(
            backend,
            config,
            asr_model=args.asr_model,
            asr_compute_type=args.asr_compute_type,
            tts_voice=args.tts_voice,
            enable_tts=not args.no_tts,
            full_duplex=args.full_duplex,
            input_device=args.input_device,
            output_device=args.output_device,
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            input_rate=args.input_rate,
            output_rate=args.output_rate,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            min_energy=args.min_energy,
            silence_ms=args.silence_ms,
            language=args.language,
            as_json=args.json,
            stats=args.stats,
        )
    finally:
        backend.close()
    return 0


def cmd_audio_check(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise DanglishError(
            "audio diagnostics need the optional speech dependencies.\n"
            "  uv:  uv sync --extra speech\n"
            '  pip: pip install "danglish[speech]"'
        ) from exc

    try:
        from .audio.check import check_input, check_output, list_devices
        from .audio.device import resolve_device

        input_index = None
        output_index = None
        if not args.list:
            input_index = resolve_device(sd, args.input_device, "input")
            if not args.no_tone:
                output_index = resolve_device(sd, args.output_device, "output")

        list_devices(sd, input_index=input_index, output_index=output_index)
        if args.list:
            return 0

        print()
        check_input(
            sd,
            args.input_device,
            seconds=args.seconds,
            channels=args.input_channels,
            channel=args.input_channel,
            sample_rate=args.input_rate,
        )
        if not args.no_tone:
            print()
            check_output(
                sd,
                args.output_device,
                channels=args.output_channels,
                channel=args.output_channel,
                sample_rate=args.output_rate,
            )
    except (ValueError, sd.PortAudioError) as exc:
        raise DanglishError(f"audio check failed: {exc}") from exc
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    ensure_model(
        args.output,
        model_id=args.model_id,
        quantization=args.quantization,
        revision=args.revision,
        force=args.force,
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    checks = build_checks(config, mock=args.mock)
    status = overall_status(checks)

    if args.json:
        _emit_json(
            {
                "status": status,
                "checks": [
                    {"label": check.label, "status": check.status, "detail": check.detail}
                    for check in checks
                ],
            }
        )
    else:
        print(f"Danglish doctor {__version__}")
        print("─" * 44)
        print(render_checks(checks))
        print()

    if args.load:
        backend = create_backend(config, mock=args.mock)
        try:
            before = current_rss_mb()
            backend.warmup()
            sample = backend.translate(["God morgen, hvordan har du det?"])[0]
            after = current_rss_mb()
            print(f"loaded backend: {after - before:+,.0f} MB, RSS now {after:,.0f} MB")
            print(f"sample: {sample}")
        finally:
            backend.close()

    if not args.json:
        verdict = {"ok": "PASS", "warn": "PASS (with warnings)", "fail": "FAIL"}[status]
        print(f"verdict: {verdict}")

    return 1 if args.strict and status == "fail" else 0


def cmd_bench(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _require_model(config, mock=args.mock)
    backend = create_backend(config, mock=args.mock)
    try:
        result = run_benchmark(
            backend,
            model_dir=None if args.mock else config.model.path,
            repeats=args.repeats,
            budget_mb=args.budget_mb,
        )
    finally:
        backend.close()

    if args.json:
        print(
            json.dumps(
                result.to_dict(include_samples=args.show_samples),
                ensure_ascii=False,
            )
        )
    else:
        print(f"Danglish benchmark {__version__}")
        print("─" * 44)
        print(f"model          {'mock' if args.mock else config.model.path}")
        print(f"samples        {result.sentences} translations ({args.repeats}x)")
        print(f"latency        p50 {result.p50_ms:,.1f} ms   p95 {result.p95_ms:,.1f} ms")
        print(f"throughput     {result.target_words_per_second:,.1f} target words/s")
        print(f"RSS after load {result.rss_after_load_mb:,.1f} MB")
        print(f"peak RSS       {result.peak_rss_mb:,.1f} MB")
        headroom = result.budget_mb - result.peak_rss_mb
        verdict = "PASS" if result.within_budget else "FAIL"
        print(f"budget         {result.budget_mb:,} MB — {verdict} ({headroom:+,.0f} MB headroom)")
        if args.show_samples:
            print()
            for sample in result.samples:
                print(f"da  {sample.source}")
                print(f"en  {sample.target}")
                print(f"    {sample.latency_ms:,.1f} ms")
                print()

    return 0 if result.within_budget else 1


def cmd_version(args: argparse.Namespace) -> int:
    print(f"danglish {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="danglish",
        description="Local, streaming Danish-to-English translation under 2 GB of RAM.",
    )
    parser.add_argument("--version", action="version", version=f"danglish {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    translate = subparsers.add_parser(
        "translate", help="translate text once and exit", description="Translate text once."
    )
    translate.add_argument("text", nargs="*", help="Danish text (or '-' for stdin)")
    translate.add_argument("-f", "--file", type=Path, metavar="FILE", help="read input from FILE")
    translate.add_argument("--json", action="store_true", help="emit JSON")
    _add_model_options(translate)
    translate.set_defaults(func=cmd_translate)

    repl = subparsers.add_parser(
        "repl", help="interactive line-by-line translation", description="Interactive translation."
    )
    repl.add_argument("--json", action="store_true", help="emit JSON lines")
    _add_model_options(repl)
    repl.set_defaults(func=cmd_repl)

    stream = subparsers.add_parser(
        "stream",
        help="translate stdin incrementally (partials on a TTY)",
        description="Translate stdin incrementally. Best used with a pipe.",
    )
    stream.add_argument("--no-partial", action="store_true", help="disable partial translations")
    stream.add_argument(
        "--debounce-ms",
        type=int,
        metavar="MS",
        help="partial-preview debounce (default: from config, 200 ms)",
    )
    stream.add_argument(
        "--chunk-chars",
        type=int,
        default=1,
        metavar="N",
        help="read N characters at a time (default: 1)",
    )
    stream.add_argument("--json", action="store_true", help="emit final events as JSON lines")
    _add_model_options(stream)
    stream.set_defaults(func=cmd_stream)

    serve = subparsers.add_parser(
        "serve", help="run the local web demo", description="Run the local WebSocket + web UI."
    )
    serve.add_argument("--host", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default: 8765)")
    _add_model_options(serve)
    serve.set_defaults(func=cmd_serve)

    talk = subparsers.add_parser(
        "talk",
        aliases=["mic"],
        help="speech-to-speech: Danish microphone → English voice",
        description=(
            "Live Danish speech → English speech. Uses local Whisper ASR, OPUS-MT "
            "translation, and a local Piper voice. Capture, translation, and playback "
            "run in parallel; use --full-duplex with isolated audio I/O."
        ),
        epilog=(
            "Recommended Piper voices: en_GB-alba-medium (63 MB, CC BY 4.0), "
            "en_GB-cori-high (114 MB, public domain)."
        ),
    )
    talk.add_argument("--asr-model", default="small", help="faster-whisper model (default: small)")
    talk.add_argument(
        "--asr-compute-type", default="int8", choices=COMPUTE_TYPES, help="ASR compute type"
    )
    talk.add_argument(
        "--tts-voice",
        help="Piper voice name or local .onnx path (default: from config, en_GB-alba-medium)",
    )
    talk.add_argument(
        "--no-tts", action="store_true", help="print the English translation instead of speaking it"
    )
    talk.add_argument(
        "--full-duplex",
        action="store_true",
        help=(
            "keep listening while English audio plays; use isolated input/output "
            "or an external echo canceller"
        ),
    )
    talk.add_argument("--language", default="da", help="source language code (default: da)")
    talk.add_argument("--input-device", help="input device index or name substring (e.g. 'Volt 4')")
    talk.add_argument(
        "--output-device", help="output device index or name substring (e.g. 'Volt 4')"
    )
    talk.add_argument(
        "--input-channel",
        type=int,
        default=0,
        help="0-based input channel (UAD input 1 is usually channel 0)",
    )
    talk.add_argument(
        "--output-channel",
        type=int,
        help="0-based output channel; default duplicates mono to both opened channels",
    )
    talk.add_argument("--input-rate", type=int, help="capture rate (default: device rate)")
    talk.add_argument("--output-rate", type=int, help="playback rate (default: device rate)")
    talk.add_argument(
        "--input-channels",
        type=int,
        help="number of input channels to open (default: enough for --input-channel)",
    )
    talk.add_argument(
        "--output-channels",
        type=int,
        help="number of output channels to open (default: 2 when available)",
    )
    talk.add_argument(
        "--min-energy",
        type=float,
        default=0.006,
        help="VAD energy threshold (default: 0.006)",
    )
    talk.add_argument(
        "--silence-ms", type=int, default=600, help="silence before cutting (default: 600 ms)"
    )
    talk.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    talk.add_argument("--stats", action="store_true", help="print ASR/MT/TTS timings per turn")
    talk.add_argument("--json", action="store_true", help="emit JSON lines")
    _add_model_options(talk)
    talk.set_defaults(func=cmd_talk)

    audio_check = subparsers.add_parser(
        "audio-check",
        aliases=["check-audio"],
        help="test audio devices, input levels, and output",
        description=(
            "List audio devices, record a few seconds from the input and report "
            "per-channel levels, and play a test tone on the output."
        ),
    )
    audio_check.add_argument("--input-device", help="index or name substring")
    audio_check.add_argument("--output-device", help="index or name substring")
    audio_check.add_argument("--input-channel", type=int, default=0, help="0-based channel")
    audio_check.add_argument("--output-channel", type=int, help="0-based channel")
    audio_check.add_argument(
        "--input-channels", type=int, help="channels to open for the level test"
    )
    audio_check.add_argument("--output-channels", type=int, help="channels to open for the tone")
    audio_check.add_argument("--input-rate", type=int)
    audio_check.add_argument("--output-rate", type=int)
    audio_check.add_argument("--seconds", type=float, default=3.0, help="input test duration")
    audio_check.add_argument("--no-tone", action="store_true", help="skip the output test")
    audio_check.add_argument("--list", action="store_true", help="only list devices")
    audio_check.set_defaults(func=cmd_audio_check)

    fetch = subparsers.add_parser(
        "fetch",
        help="download and convert the model",
        description="Download and convert the model.",
    )
    fetch.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    fetch.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    fetch.add_argument(
        "--quantization",
        default=DEFAULT_QUANTIZATION,
        help=f"CT2 quantization (default: {DEFAULT_QUANTIZATION})",
    )
    fetch.add_argument("--revision", help="Hugging Face revision")
    fetch.add_argument("--force", action="store_true", help="re-convert even if the model exists")
    fetch.set_defaults(func=cmd_fetch)

    doctor = subparsers.add_parser(
        "doctor", help="check the environment and memory budget", description="Environment checks."
    )
    doctor.add_argument("--json", action="store_true", help="emit JSON")
    doctor.add_argument("--load", action="store_true", help="load the model and measure real RSS")
    doctor.add_argument("--strict", action="store_true", help="exit 1 on failed checks")
    _add_model_options(doctor)
    doctor.set_defaults(func=cmd_doctor)

    bench = subparsers.add_parser(
        "bench", help="measure latency and peak RSS", description="Latency and memory benchmark."
    )
    bench.add_argument(
        "--repeats", type=int, default=1, metavar="N", help="repeat the corpus N times"
    )
    bench.add_argument(
        "--budget-mb",
        type=int,
        default=DEFAULT_BUDGET_MB,
        help=f"memory budget in MB (default: {DEFAULT_BUDGET_MB})",
    )
    bench.add_argument("--json", action="store_true", help="emit JSON")
    bench.add_argument("--show-samples", action="store_true", help="print every translation")
    _add_model_options(bench)
    bench.set_defaults(func=cmd_bench)

    version = subparsers.add_parser("version", help="print the version")
    version.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except DanglishError as exc:
        print(f"danglish: error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
