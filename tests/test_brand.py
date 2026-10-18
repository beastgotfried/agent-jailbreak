"""Veyra name and aliases do not change the selected model identity."""
import hashlib
from pathlib import Path
import tomllib

from src.release import cli

ROOT = Path(__file__).resolve().parents[1]


def test_veyra_cli_aliases_keep_legacy_entrypoints():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "veyra"
    assert project["scripts"]["veyra"] == project["scripts"]["safety-watch"]
    assert project["scripts"]["veyra-replay"] == project["scripts"]["safety-watch-replay"]


def test_public_product_name_is_veyra(capsys):
    assert cli.PRODUCT_NAME == "Veyra"
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "usage: veyra" in output
    assert "Veyra" in output


def test_original_selected_identity_is_byte_preserved():
    path = ROOT / "src/release/selected.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "543dc30856ea8646ce2fe9363fced0e5b2eb4424dd1e6276eec88b89e92e0389"
    spec = cli.selected_release()
    assert spec["release_id"] == "safety-watch-tfidf-20261017"
    assert spec["model_pair_id"] == "adb5b660369f0a8b9024a3d725548aeca034d86eac72cbaa3e975a4a77fa2790"
