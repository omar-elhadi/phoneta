import pytest

from phoneta.__main__ import build_parser, main


def test_parser_has_phoneta_prog() -> None:
    assert build_parser().prog == "phoneta"


def test_main_returns_zero_and_mentions_phoneta(capsys) -> None:
    assert main([]) == 0
    assert "Phoneta" in capsys.readouterr().out


def test_version_flag_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
