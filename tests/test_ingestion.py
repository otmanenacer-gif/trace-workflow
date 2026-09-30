"""Tests de validation, de robustesse et d'intégration de l'ingestion (sans IA)."""

import hashlib
import json

import pytest

from core import config
from core.ingestion import ingest_file, ingest_run, unique_interview_id
from core.ingestion_validator import check_coverage, check_turn_sequence, compute_status
from core.run_manager import init_run, load_metadata
from core.schemas import STATUS_FAIL, STATUS_PASS, STATUS_PASS_WITH_WARNINGS
from tests.synthetic_docs import SAMPLE_LINES, make_docx, make_pdf

CLEAN_DIALOGUE = "Enquêteur : Bonjour, ça va ?\nEnquêté : Euh… ouais, ouais.\nEnquêteur : Tu utilises l’IA ?\nEnquêté : Bah oui : tout le temps."


def run_ingest(tmp_path, name, data):
    source = tmp_path / name
    source.write_bytes(data)
    out = tmp_path / "out"
    summary = ingest_file(source, "TEST", out)
    transcript = json.loads((out / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
    report = json.loads((out / config.INGESTION_REPORT_FILENAME).read_text(encoding="utf-8"))
    raw = (out / config.RAW_TEXT_FILENAME).read_text(encoding="utf-8")
    return summary, transcript, report, raw


def codes(report):
    return [w["code"] for w in report["warnings"]]


# --- Contrôle de couverture -----------------------------------------------------------

def turn(text, marker=None, index=1, line=1):
    return {"turn_id": f"T{index}", "index": index, "marker": marker, "text": text, "source": {"line_start": line}}


def test_coverage_exact():
    metrics, warnings = check_coverage("Enquêteur : a b\nEnquêté : c", [turn("a b", "Enquêteur :"), turn("c", "Enquêté :", 2, 2)])
    assert metrics["exact_match"] and metrics["token_coverage_ratio"] == 1.0 and warnings == []


def test_coverage_detects_loss():
    raw = " ".join(f"mot{i}" for i in range(100))
    metrics, warnings = check_coverage(raw, [turn(" ".join(f"mot{i}" for i in range(50)))])
    assert not metrics["exact_match"] and metrics["missing_token_count"] == 50
    assert [w["code"] for w in warnings] == ["CONTENT_LOSS"]
    assert compute_status(warnings) == STATUS_FAIL


def test_coverage_detects_added_content():
    raw = " ".join(f"mot{i}" for i in range(100))
    _, warnings = check_coverage(raw, [turn(raw + " inventé1 inventé2 inventé3")])
    assert "CONTENT_ADDED" in [w["code"] for w in warnings]


def test_coverage_minor_mismatch():
    raw = " ".join(f"mot{i}" for i in range(500))
    _, warnings = check_coverage(raw, [turn(raw.replace("mot7 ", ""))])
    assert [w["code"] for w in warnings] == ["CONTENT_MISMATCH_MINOR"]
    assert compute_status(warnings) == STATUS_PASS_WITH_WARNINGS


def test_turn_sequence_check():
    assert check_turn_sequence([turn("a", index=1), turn("b", index=2, line=2)]) == []
    duplicated = [turn("a", index=1), turn("b", index=1)]
    assert [w["code"] for w in check_turn_sequence(duplicated)] == ["INVALID_TURN_SEQUENCE"]


# --- Ingestion d'un fichier -------------------------------------------------------------

def test_ingest_clean_txt_pass(tmp_path):
    summary, transcript, report, raw = run_ingest(tmp_path, "e.txt", CLEAN_DIALOGUE.encode("utf-8"))
    assert summary["status"] == STATUS_PASS and report["status"] == STATUS_PASS
    assert raw == CLEAN_DIALOGUE
    assert transcript["schema_version"] == "1.0" and transcript["turn_count"] == 4
    assert transcript["source"]["sha256"] == hashlib.sha256(CLEAN_DIALOGUE.encode()).hexdigest()
    assert report["interviewer_turn_count"] == 2 and report["interviewee_turn_count"] == 2
    assert report["unknown_segment_count"] == 0 and report["attributed_text_ratio"] == 1.0
    assert report["coverage"]["exact_match"] is True
    assert report["source"]["sha256_verified_after_processing"] is True
    assert report["char_count"] == len(CLEAN_DIALOGUE)


@pytest.mark.parametrize("name, data", [
    ("e.txt", ("\n".join(SAMPLE_LINES)).encode("utf-8")),
    ("e.docx", make_docx(SAMPLE_LINES)),
    ("e.pdf", make_pdf([SAMPLE_LINES[:4], SAMPLE_LINES[4:]])),
])
def test_no_text_lost_for_each_format(tmp_path, name, data):
    _, transcript, report, raw = run_ingest(tmp_path, name, data)
    assert report["coverage"]["exact_match"] is True
    assert report["status"] == STATUS_PASS_WITH_WARNINGS  # titre avant le premier marqueur
    assert codes(report) == ["TEXT_BEFORE_FIRST_LABEL"]
    speakers = [t["speaker"] for t in transcript["turns"]]
    assert speakers == ["unknown", "enqueteur", "enquete", "enqueteur", "enquete"]
    assert "J’utilise ChatGPT, genre : pour tout, enfin presque." in transcript["turns"][2]["text"]
    for line in SAMPLE_LINES:
        assert line in raw


def test_pdf_pages_in_turns_and_report(tmp_path):
    _, transcript, report, _ = run_ingest(tmp_path, "e.pdf", make_pdf([SAMPLE_LINES[:4], SAMPLE_LINES[4:]]))
    turns = transcript["turns"]
    assert turns[2]["source"]["page"] == 1 and turns[2]["source"]["page_end"] == 2
    assert turns[3]["source"]["page"] == 2
    assert report["extraction"]["page_count"] == 2
    assert [p["page"] for p in report["extraction"]["pages"]] == [1, 2]


def test_sha256_is_stable(tmp_path):
    first, *_ = run_ingest(tmp_path, "a.txt", CLEAN_DIALOGUE.encode("utf-8"))
    report_a = json.loads((tmp_path / "out" / config.INGESTION_REPORT_FILENAME).read_text(encoding="utf-8"))
    (tmp_path / "b").mkdir()
    _, _, report_b, _ = run_ingest(tmp_path / "b", "a.txt", CLEAN_DIALOGUE.encode("utf-8"))
    assert report_a["source"]["sha256"] == report_b["source"]["sha256"]


# --- Robustesse ----------------------------------------------------------------------------

@pytest.mark.parametrize("name, data", [
    ("vide.txt", b""),
    ("vide.docx", make_docx([])),
    ("vide.pdf", make_pdf([[], []])),
    ("blancs.txt", b"  \n\n\t \n"),
])
def test_empty_documents_fail_without_inventing(tmp_path, name, data):
    summary, transcript, report, _ = run_ingest(tmp_path, name, data)
    assert summary["status"] == STATUS_FAIL and report["needs_manual_review"] is True
    assert "NO_TEXT_EXTRACTED" in codes(report)
    assert transcript["turns"] == []


@pytest.mark.parametrize("name, data", [
    ("casse.pdf", b"%PDF-1.4 garbage"),
    ("casse.docx", b"PK\x03\x04 pas vraiment un zip"),
])
def test_corrupted_files_fail_cleanly(tmp_path, name, data):
    summary, transcript, report, raw = run_ingest(tmp_path, name, data)
    assert summary["status"] == STATUS_FAIL
    assert "EXTRACTION_FAILED" in codes(report)
    assert transcript["turns"] == [] and raw == ""


def test_forbidden_extension_fails_cleanly(tmp_path):
    summary, _, report, _ = run_ingest(tmp_path, "e.odt", b"x")
    assert summary["status"] == STATUS_FAIL and codes(report) == ["UNSUPPORTED_FORMAT"]


def test_no_speaker_detected(tmp_path):
    summary, transcript, report, _ = run_ingest(tmp_path, "e.txt", "Un texte.\n\nUn autre.".encode("utf-8"))
    assert summary["status"] == STATUS_PASS_WITH_WARNINGS
    assert "NO_SPEAKER_LABEL_DETECTED" in codes(report)
    assert report["unknown_segment_count"] == 2 and report["attributed_text_ratio"] == 0.0
    assert report["coverage"]["exact_match"] is True


# --- Intégration : runs ------------------------------------------------------------------------

@pytest.fixture
def roots(tmp_path):
    return tmp_path / "inputs", tmp_path / "outputs"


def test_run_with_several_interviews_and_one_bad_file(roots):
    files = [
        ("Alice.txt", CLEAN_DIALOGUE.encode("utf-8")),
        ("Bruno.docx", make_docx(CLEAN_DIALOGUE.split("\n"))),
        ("Chloe.pdf", make_pdf([CLEAN_DIALOGUE.split("\n")])),
        ("Casse.pdf", b"%PDF-1.4 garbage"),
    ]
    metadata = ingest_run(init_run(files, "problématique", *roots))
    statuses = {f["stored_name"]: f["ingestion"]["status"] for f in metadata["files"]}
    assert statuses == {"Alice.txt": STATUS_PASS, "Bruno.docx": STATUS_PASS,
                        "Chloe.pdf": STATUS_PASS, "Casse.pdf": STATUS_FAIL}
    assert metadata["pipeline"][config.INGESTION_STEP] == "terminé (1 échec(s))"
    assert all(v == "inactive" for k, v in metadata["pipeline"].items() if k != config.INGESTION_STEP)

    output_dir = roots[1] / metadata["run_id"]
    assert load_metadata(output_dir) == metadata
    for interview_id in ("ALICE", "BRUNO", "CHLOE", "CASSE"):
        folder = output_dir / config.INTERVIEWS_SUBDIR / interview_id
        for filename in (config.RAW_TEXT_FILENAME, config.STRUCTURED_TRANSCRIPT_FILENAME, config.INGESTION_REPORT_FILENAME):
            assert (folder / filename).is_file()
    alice = json.loads((output_dir / "interviews/ALICE" / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
    assert alice["turns"][0]["turn_id"] == "ALICE_T0001"
    assert alice["turns"][0]["source"]["file"] == "Alice.txt"
    # les fichiers source copiés sont intacts
    assert (roots[0] / metadata["run_id"] / "Alice.txt").read_bytes() == CLEAN_DIALOGUE.encode("utf-8")


def test_two_runs_stay_separate(roots):
    first = ingest_run(init_run([("E.txt", b"Enqu\xc3\xaateur : un")], "", *roots))
    second = ingest_run(init_run([("E.txt", b"Enqu\xc3\xaateur : deux")], "", *roots))
    assert first["run_id"] != second["run_id"]
    texts = []
    for run in (first, second):
        path = roots[1] / run["run_id"] / "interviews/E" / config.STRUCTURED_TRANSCRIPT_FILENAME
        texts.append(json.loads(path.read_text(encoding="utf-8"))["turns"][0]["text"])
    assert texts == ["un", "deux"]


def test_interview_ids_unique_within_run():
    taken = set()
    assert [unique_interview_id(n, taken) for n in ("Eloise.pdf", "eloise.txt", "ÉLOISE.docx")] == \
        ["ELOISE", "ELOISE_2", "ELOISE_3"]
