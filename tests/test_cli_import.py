from camarca.cli import main


def test_cli_importable() -> None:
    assert callable(main)
