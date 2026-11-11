"""Download and convert the default OPUS-MT model to CTranslate2."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .errors import DanglishError, DependencyMissingError
from .memory import directory_size_mb

DEFAULT_MODEL_ID = "Helsinki-NLP/opus-mt-da-en"
DEFAULT_OUTPUT_DIR = Path("models/opus-mt-da-en-int8")
DEFAULT_QUANTIZATION = "int8"

_DOWNLOAD_PATTERNS = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
    "README.md",
    "LICENSE",
]

_COPY_AFTER_CONVERSION = (
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
)

_REQUIRED = ("model.bin", "source.spm", "target.spm")


def model_is_ready(path: Path | str) -> bool:
    root = Path(path)
    return all((root / name).is_file() for name in _REQUIRED)


def ensure_model(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    quantization: str = DEFAULT_QUANTIZATION,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download *model_id* and convert it to a CTranslate2 model.

    Conversion requires the optional ``convert`` extra (torch + transformers).
    The resulting directory only needs the runtime dependencies.
    """

    output = Path(output_dir)
    if model_is_ready(output) and not force:
        print(f"model already present: {output} ({directory_size_mb(output):,.0f} MB)")
        return output

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - core dependency
        raise DependencyMissingError(
            "huggingface-hub is required to download the model; install the danglish package"
        ) from exc

    print(f"downloading {model_id} ...", flush=True)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=_DOWNLOAD_PATTERNS,
            )
        )
    except Exception as exc:  # huggingface_hub has several exception types
        raise DanglishError(f"download failed: {exc}") from exc

    print("converting to CTranslate2 (one-time; this needs torch) ...", flush=True)
    try:
        import ctranslate2
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise DependencyMissingError(
            "model conversion needs the optional convert dependencies.\n"
            "  uv:  uv run --extra convert danglish fetch\n"
            '  pip: pip install "danglish[convert]" '
            "--extra-index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    copied = [name for name in _COPY_AFTER_CONVERSION if (snapshot / name).is_file()]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter = TransformersConverter(str(snapshot), copy_files=copied)
        converter.convert(str(output), quantization=quantization, force=True)
    except Exception as exc:
        raise DanglishError(f"conversion failed: {exc}") from exc

    # Some converter versions do not copy the tokenizer assets. Copy them
    # explicitly; having them twice is harmless.
    for name in _COPY_AFTER_CONVERSION:
        source = snapshot / name
        destination = output / name
        if source.is_file() and not destination.is_file():
            shutil.copy2(source, destination)

    if not model_is_ready(output):
        missing = ", ".join(name for name in _REQUIRED if not (output / name).is_file())
        raise DanglishError(f"conversion finished but the model is incomplete (missing: {missing})")

    metadata = {
        "format": "ctranslate2",
        "source_model": model_id,
        "revision": revision,
        "quantization": quantization,
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "danglish_version": __version__,
        "ctranslate2_version": ctranslate2.__version__,
    }
    (output / "danglish-model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"model ready: {output} ({directory_size_mb(output):,.0f} MB, {quantization})",
        flush=True,
    )
    return output
