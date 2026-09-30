"""Tests du gestionnaire de runs (aucun appel IA, dossiers temporaires uniquement)."""

import json
import re
from datetime import datetime

import pytest

from core import config
from core.run_manager import (
    TraceError,
    create_run_dirs,
    generate_run_id,
    init_run,
    load_metadata,
    save_input_file,
    save_metadata,
    validate_extension,
)


@pytest.fixture
def roots(tmp_path):
    """Dossiers inputs/outputs isolés pour chaque test."""
    return tmp_path / "inputs", tmp_path / "outputs"


# --- Identifiant de run ---------------------------------------------------

def test_generate_run_id_format():
    run_id = generate_run_id(datetime(2026, 9, 30, 14, 5, 7))
    assert re.fullmatch(r"run_20260930_140507_[0-9a-f]{8}", run_id)


def test_generate_run_id_unique_at_same_instant():
    now = datetime(2026, 9, 30, 14, 5, 7)
    ids = {generate_run_id(now) for _ in range(1000)}
    assert len(ids) == 1000


# --- Dossiers --------------------------------------------------------------

def test_create_run_dirs(roots):
    inputs_root, outputs_root = roots
    input_dir, output_dir = create_run_dirs("run_test", inputs_root, outputs_root)
    assert input_dir == inputs_root / "run_test" and input_dir.is_dir()
    assert output_dir == outputs_root / "run_test" and output_dir.is_dir()


def test_create_run_dirs_refuses_existing_run(roots):
    create_run_dirs("run_test", *roots)
    with pytest.raises(TraceError, match="existe déjà"):
        create_run_dirs("run_test", *roots)


# --- Métadonnées -----------------------------------------------------------

def test_save_and_load_metadata(tmp_path):
    metadata = {"run_id": "run_x", "problematique": "Usages des IAG — été", "files": []}
    path = save_metadata(metadata, tmp_path)
    assert path == tmp_path / config.METADATA_FILENAME
    assert json.loads(path.read_text(encoding="utf-8")) == metadata
    assert load_metadata(tmp_path) == metadata


# --- Extensions ------------------------------------------------------------

@pytest.mark.parametrize("name", ["a.pdf", "b.DOCX", "c.txt", "d.e.Txt"])
def test_validate_extension_accepts_allowed(name):
    assert validate_extension(name) in config.ALLOWED_EXTENSIONS


@pytest.mark.parametrize("name", ["a.exe", "b.doc", "c.odt", "d.py", "sans_extension", "pdf"])
def test_validate_extension_rejects_others(name):
    with pytest.raises(TraceError, match="format accepté"):
        validate_extension(name)


def test_init_run_rejects_bad_extension_without_creating_run(roots):
    inputs_root, outputs_root = roots
    with pytest.raises(TraceError):
        init_run([("ok.txt", b"x"), ("virus.exe", b"x")], "", inputs_root, outputs_root)
    assert not inputs_root.exists() and not outputs_root.exists()


# --- Fichiers et run complet -----------------------------------------------

def test_save_input_file_keeps_content_and_strips_path(tmp_path):
    info = save_input_file("../../evil/entretien.txt", b"bonjour", tmp_path)
    assert info["stored_name"] == "entretien.txt"
    assert (tmp_path / "entretien.txt").read_bytes() == b"bonjour"
    assert info["size_bytes"] == 7 and info["format"] == "TXT"


def test_save_input_file_does_not_overwrite_duplicates(tmp_path):
    save_input_file("e.txt", b"1", tmp_path)
    info = save_input_file("e.txt", b"2", tmp_path)
    assert info["stored_name"] == "e_1.txt"
    assert (tmp_path / "e.txt").read_bytes() == b"1"


def test_init_run_requires_files(roots):
    with pytest.raises(TraceError, match="Aucun fichier"):
        init_run([], "", *roots)


def test_init_run_full(roots):
    inputs_root, outputs_root = roots
    meta = init_run([("e1.txt", b"abc"), ("e2.pdf", b"%PDF")], "Ma problématique", *roots)
    run_id = meta["run_id"]
    assert meta["file_count"] == 2
    assert sorted(p.name for p in (inputs_root / run_id).iterdir()) == ["e1.txt", "e2.pdf"]
    output_dir = outputs_root / run_id
    assert (output_dir / config.RUN_PROBLEMATIQUE_FILENAME).read_text(encoding="utf-8") == "Ma problématique"
    assert load_metadata(output_dir) == meta
    assert set(meta["pipeline"]) == set(config.PIPELINE_STEPS)


def test_two_runs_do_not_collide(roots):
    inputs_root, outputs_root = roots
    first = init_run([("e.txt", b"premier")], "", *roots)
    second = init_run([("e.txt", b"second")], "", *roots)
    assert first["run_id"] != second["run_id"]
    assert (inputs_root / first["run_id"] / "e.txt").read_bytes() == b"premier"
    assert (inputs_root / second["run_id"] / "e.txt").read_bytes() == b"second"
    assert len(list(outputs_root.iterdir())) == 2
