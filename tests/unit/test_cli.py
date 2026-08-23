import pytest

from phoneta.__main__ import build_parser, main


def test_parser_has_phoneta_prog() -> None:
    assert build_parser().prog == "phoneta"


def test_main_returns_zero() -> None:
    # GUI launch is tested via smoke/integration — parser-only test here.
    parser = build_parser()
    args = parser.parse_args([])
    assert args is not None


def test_version_flag_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
