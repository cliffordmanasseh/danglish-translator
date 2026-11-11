#!/usr/bin/env python3
"""Assemble a self-contained model bundle for the desktop app.

The bundle is platform-independent: CTranslate2/ONNX weights and SentencePiece
models are copied into one directory that Electron ships as ``resources/models``.

Usage:
    uv run --extra convert python scripts/bundle_models.py \
        --output desktop/resources/models
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from danglish import __version__
from danglish.memory import directory_size_mb
from danglish.models import (
    DEFAULT_MODEL_ID,
    DEFAULT_QUANTIZATION,
    ensure_model,
    model_is_ready,
)
from danglish.tts import DEFAULT_VOICE, ensure_voice

WHISPER_REPO = "Systran/faster-whisper-small"
WHISPER_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt", "README.md"]


def _copy_dir(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def bundle_mt(
    output: Path,
    *,
    source_dir: Path | None,
    model_id: str,
    quantization: str,
    force: bool,
) -> Path:
    destination = output / "opus-mt-da-en-int8"
    if model_is_ready(destination) and not force:
        print(f"MT: reusing {destination} ({directory_size_mb(destination):,.0f} MB)")
        return destination

    if source_dir is not None and model_is_ready(source_dir):
        print(f"MT: copying {source_dir} ...")
        _copy_dir(source_dir, destination)
    else:
        print(f"MT: fetching and converting {model_id} ...")
        ensure_model(
            destination,
            model_id=model_id,
            quantization=quantization,
            force=force,
        )

    if not model_is_ready(destination):
        raise SystemExit(f"MT bundle is incomplete: {destination}")
    print(f"MT: {destination} ({directory_size_mb(destination):,.0f} MB)")
    return destination


def bundle_asr(output: Path, *, repo_id: str, force: bool) -> Path:
    destination = output / "whisper-small"
    if (destination / "model.bin").is_file() and not force:
        print(f"ASR: reusing {destination} ({directory_size_mb(destination):,.0f} MB)")
        return destination

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface-hub is required to bundle Whisper") from exc

    print(f"ASR: downloading {repo_id} ...")
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        allow_patterns=WHISPER_FILES,
    )
    # Hugging Face may leave a local metadata cache; it is not needed at runtime.
    shutil.rmtree(destination / ".cache", ignore_errors=True)

    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        if not (destination / name).is_file():
            raise SystemExit(f"ASR bundle is incomplete: missing {destination / name}")
    print(f"ASR: {destination} ({directory_size_mb(destination):,.0f} MB)")
    return destination


def bundle_voice(output: Path, *, voice: str, force: bool) -> Path:
    destination = output / "voices"
    model_name = Path(voice).name
    if not model_name.endswith(".onnx"):
        model_name = f"{voice}.onnx"
    target = destination / model_name
    config = destination / f"{model_name}.json"
    if target.is_file() and config.is_file() and not force:
        print(f"TTS: reusing {target} ({target.stat().st_size / (1024 * 1024):,.0f} MB)")
        return target

    print(f"TTS: preparing voice {voice} ...")
    source_model, source_config = ensure_voice(voice)
    destination.mkdir(parents=True, exist_ok=True)
    _copy_file(source_model, target)
    _copy_file(source_config, config)

    attribution = destination / "README.md"
    attribution.write_text(
        "# Bundled Piper voices\n\n"
        f"- `{model_name}` — voice `{voice}` from "
        "https://huggingface.co/rhasspy/piper-voices.\n"
        "- Check the voice's model card for its dataset license and attribution.\n"
        "- The default `en_GB-alba-medium` voice is CC BY 4.0 "
        "(dataset: https://datashare.ed.ac.uk/handle/10283/3270).\n"
        "- `piper-tts` itself is GPL-3.0-or-later. See NOTICE.\n",
        encoding="utf-8",
    )
    print(f"TTS: {target} ({target.stat().st_size / (1024 * 1024):,.0f} MB)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("desktop/resources/models"),
        help="destination directory",
    )
    parser.add_argument(
        "--mt-source",
        type=Path,
        help="existing converted MT model to copy (default: try models/opus-mt-da-en-int8)",
    )
    parser.add_argument("--mt-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--whisper-repo", default=WHISPER_REPO)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--force", action="store_true", help="re-copy/re-download everything")
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    mt_source = args.mt_source
    if mt_source is None:
        candidate = Path("models/opus-mt-da-en-int8")
        if candidate.is_dir():
            mt_source = candidate

    mt_path = bundle_mt(
        output,
        source_dir=mt_source,
        model_id=args.mt_model_id,
        quantization=args.quantization,
        force=args.force,
    )
    asr_path = bundle_asr(output, repo_id=args.whisper_repo, force=args.force)
    voice_path = bundle_voice(output, voice=args.voice, force=args.force)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "danglish_version": __version__,
        "total_mb": round(directory_size_mb(output), 1),
        "translation": {
            "path": str(mt_path.relative_to(output)),
            "source_model": args.mt_model_id,
            "license": "Apache-2.0",
        },
        "asr": {
            "path": str(asr_path.relative_to(output)),
            "source_model": args.whisper_repo,
            "license": "MIT",
        },
        "tts": {
            "runtime": "piper-tts",
            "runtime_license": "GPL-3.0-or-later",
            "voice": args.voice,
            "path": str(voice_path.relative_to(output)),
            "voice_license": "CC BY 4.0 (en_GB-alba-medium; check other voices)",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nmodel bundle ready: {output} ({directory_size_mb(output):,.0f} MB total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
