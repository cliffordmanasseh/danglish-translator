def test_collapse_repetitions() -> None:
    from danglish.audio.asr import collapse_repetitions

    looped = "Så " + "det eneste jeg kunne bruge " * 6 + "kronik i"
    assert collapse_repetitions(looped) == "Så det eneste jeg kunne bruge kronik i"
    assert collapse_repetitions("ja ja") == "ja ja"
    assert collapse_repetitions("Gud er god") == "Gud er god"
