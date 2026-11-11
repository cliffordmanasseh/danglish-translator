from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from danglish.cli import build_parser, main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "danglish 0.1.0" in capsys.readouterr().out


def test_translate_mock(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["translate", "--mock", "God morgen"]) == 0
    output = capsys.readouterr().out
    assert "[mock]" in output
    assert "morgen" in output


def test_translate_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["translate", "--mock", "--json", "Hej med dig."]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["translation"].startswith("[mock]")
    assert payload["model"] == "mock"


def test_translate_from_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "input.txt"
    source.write_text("Hej. Farvel.", encoding="utf-8")

    assert main(["translate", "--mock", "--file", str(source)]) == 0
    output = capsys.readouterr().out
    assert "Hej." in output
    assert "Farvel." in output


def test_translate_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Hej fra stdin."))

    assert main(["translate", "--mock"]) == 0
    assert "[mock]" in capsys.readouterr().out


def test_translate_empty_input_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["translate", "--mock", "   "]) == 1
    assert "empty" in capsys.readouterr().err


def test_translate_missing_model_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["translate", "--model-path", str(tmp_path), "Hej"]) == 1
    error = capsys.readouterr().err
    assert "model not found" in error
    assert "danglish fetch" in error


def test_repl_mock(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Hej.\nFarvel.\n"))

    assert main(["repl", "--mock"]) == 0
    output = capsys.readouterr().out
    assert "da  Hej." in output
    assert "en  [mock] Hej." in output
    assert "da  Farvel." in output


def test_stream_mock(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Hej. Farvel. "))

    assert main(["stream", "--mock", "--no-partial"]) == 0
    output = capsys.readouterr().out
    assert "da  Hej." in output
    assert "da  Farvel." in output


def test_bench_mock(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--mock", "--repeats", "1"]) == 0
    output = capsys.readouterr().out
    assert "peak RSS" in output
    assert "PASS" in output


def test_bench_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--mock", "--json", "--show-samples"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["within_budget"] is True
    assert payload["sentences"] == 10
    assert len(payload["samples"]) == 10


def test_doctor_mock_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--mock", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    labels = {check["label"] for check in payload["checks"]}
    assert "Python" in labels
    assert "Model" in labels
    assert "Memory estimate" in labels


def test_doctor_strict_does_not_fail_on_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--mock", "--strict"]) == 0
    assert "verdict:" in capsys.readouterr().out


def test_missing_config_file_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor", "--mock", "--config", str(tmp_path / "nope.toml")]) == 1
    assert "not found" in capsys.readouterr().err


def test_talk_requires_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["talk", "--model-path", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert "model not found" in error


def test_mic_alias_requires_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["mic", "--model-path", str(tmp_path)]) == 1
    assert "model not found" in capsys.readouterr().err


def test_talk_full_duplex_flag() -> None:
    args = build_parser().parse_args(["talk", "--full-duplex"])
    assert args.full_duplex is True

    default_args = build_parser().parse_args(["talk"])
    assert default_args.full_duplex is False


def test_bad_arguments_exit(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["unknown-command"])
    assert excinfo.value.code == 2
