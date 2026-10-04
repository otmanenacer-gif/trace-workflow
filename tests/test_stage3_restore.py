"""Restauration de sorties de l'étape 3 déjà calculées, puis étape 4 sans relancer l'étape 3. agents simulés."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import accountability, config, stage3_restore
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from core.stage3_restore import RestoreError, check_restore, restore_stage3
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S
from tests.fake_llm import ACCOUNTABILITY, FakeAgents, fake_settings

NAMES = ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json",
         "speaker_attribution_audit.json")


def downloaded_stage3(tmp_path, files=S.FILES, prefix=S.INTERVIEW_ID + "_") -> list[tuple[str, bytes]]:
    """Étape 3 calculée (agents simulés) dans un premier run, puis « téléchargée » (noms préfixés comme dans l'UI)."""
    run = si.make_ingested_run(tmp_path / "before", files)
    analyze_run(run, settings=fake_settings(), client=FakeAgents(S.stage3_responders()),
                cache=AnalysisCache(tmp_path / "before_cache"))
    out = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    return [(prefix + name, (out / name).read_bytes()) for name in NAMES]


def fresh_run(tmp_path, files=S.FILES):
    """Après un redéploiement : le même entretien réimporté, sans aucune sortie de l'étape 3."""
    return si.make_ingested_run(tmp_path / "after", files)


def edited(uploads, name, change):
    result = []
    for upload_name, data in uploads:
        if upload_name.endswith(name):
            document = json.loads(data)
            change(document)
            data = json.dumps(document, ensure_ascii=False).encode("utf-8")
        result.append((upload_name, data))
    return result


def rejected(run, uploads, interview_id=S.INTERVIEW_ID) -> str:
    with pytest.raises(RestoreError) as info:
        restore_stage3(run, interview_id, uploads)
    analysis_dir = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    assert not (analysis_dir / "practice_extractor.json").exists()  # rien n'est installé
    return " ".join(info.value.problems)


def stage4_only_transport():
    """Seul l'Accountability Episode Builder répond : un appel de l'étape 3 lèverait une erreur."""
    return FakeAgents({ACCOUNTABILITY: S.REFERENCE_BUILDER})


# --- Restauration réussie --------------------------------------------------------------------------

def test_restored_stage3_feeds_stage4_without_any_stage3_call(tmp_path):
    uploads = downloaded_stage3(tmp_path)
    run = fresh_run(tmp_path)
    analysis_dir = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    cache = AnalysisCache(tmp_path / "after_cache")  # cache TRACE vide, comme après un redéploiement
    assert accountability.plan_stage4(run, [S.INTERVIEW_ID], fake_settings(), cache)["blocked"] == 1

    run = restore_stage3(run, S.INTERVIEW_ID, uploads)
    # fichiers installés octet pour octet, sans modification
    for name, data in uploads:
        assert (analysis_dir / name.removeprefix(S.INTERVIEW_ID + "_")).read_bytes() == data
    assert accountability.stage3_state(analysis_dir)["status"] == "COMPLETE"
    summary = run["files"][0]["analysis"]
    assert summary["restored"] is True and summary["agents"]["practice_extractor"]["api_calls"] == 0
    assert run["pipeline"][config.PRACTICE_STEP] == "terminé (1/1 entretien(s))"
    manifest = json.loads((analysis_dir / stage3_restore.RESTORE_MANIFEST_FILENAME).read_text())
    assert manifest["api_calls"] == 0 and manifest["notice"] == "Stage 3 restauré depuis fichiers — 0 appel API"
    assert set(manifest["files"]) == set(NAMES)

    plan = accountability.plan_stage4(run, [S.INTERVIEW_ID], fake_settings(), cache)
    assert plan["blocked"] == 0 and plan["calls"] == 2 and plan["candidates"] == 6  # blocs de 4 et 2
    transport = stage4_only_transport()
    run = accountability.analyze_run_stage4(run, settings=fake_settings(), client=transport, cache=cache)
    assert [c["agent"] for c in transport.calls] == [ACCOUNTABILITY] * 2  # deux blocs ; ni étape 3, ni audit
    result = run["files"][0]["accountability"]
    assert result["status"] == "SUCCESS" and result["stage3_status"] == "COMPLETE"
    assert (result["accountability_episode_count"], result["ordinary_practice_count"], result["uncertain_count"],
            result["unmarked_practice_count"]) == (4, 1, 1, 4)
    # l'avertissement de locuteur restauré est bien propagé à l'étape 4
    document = json.loads((analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME).read_text())
    assert any(e["speaker_warnings"] for e in document["episodes"])


def test_restore_replaces_a_failed_stage3_and_removes_stale_stage4_outputs(tmp_path):
    from tests.fake_llm import INTERACTION, agent_error
    uploads = downloaded_stage3(tmp_path)
    run = fresh_run(tmp_path)
    responders = {**S.stage3_responders(), INTERACTION: agent_error()}
    run = analyze_run(run, settings=fake_settings(), client=FakeAgents(responders),
                      cache=AnalysisCache(tmp_path / "c"))
    analysis_dir = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    run = accountability.analyze_run_stage4(run, settings=fake_settings(), client=stage4_only_transport(),
                                            cache=AnalysisCache(tmp_path / "c"))
    assert run["files"][0]["accountability"]["status"] == "BLOCKED"
    run = restore_stage3(run, S.INTERVIEW_ID, uploads)
    assert "accountability" not in run["files"][0]
    assert not (analysis_dir / config.ACCOUNTABILITY_MANIFEST_FILENAME).exists()
    assert accountability.stage3_state(analysis_dir)["status"] == "COMPLETE"


def test_files_are_recognized_by_content_whatever_their_names(tmp_path):
    uploads = [(f"fichier_{i}.json", data) for i, (_, data) in enumerate(reversed(downloaded_stage3(tmp_path)))]
    checked = check_restore(fresh_run(tmp_path), S.INTERVIEW_ID, uploads)
    assert set(checked["files"]) == set(NAMES) and checked["invalid_evidence_count"] == 0


# --- Refus --------------------------------------------------------------------------------------------

def test_missing_file_is_rejected(tmp_path):
    uploads = downloaded_stage3(tmp_path)
    assert "Fichier manquant : evidence_validation.json" in rejected(fresh_run(tmp_path), uploads[:2] + uploads[3:])


def test_invalid_json_is_rejected(tmp_path):
    uploads = downloaded_stage3(tmp_path)
    uploads[1] = (uploads[1][0], b'{"signals": [')
    assert "JSON invalide" in rejected(fresh_run(tmp_path), uploads)


def test_duplicate_or_unknown_file_is_rejected(tmp_path):
    uploads = downloaded_stage3(tmp_path)
    assert "fourni deux fois" in rejected(fresh_run(tmp_path), uploads + [uploads[0]])
    assert "non reconnu" in rejected(fresh_run(tmp_path / "x"), uploads + [("autre.json", b'{"a": 1}')])


def test_mixed_interviews_are_rejected(tmp_path):
    uploads = edited(downloaded_stage3(tmp_path), "speaker_attribution_audit.json",
                     lambda d: d.update(interview_id="AUTRE_ENTRETIEN"))
    assert "plusieurs entretiens" in rejected(fresh_run(tmp_path), uploads)


def test_outputs_of_another_interview_are_rejected(tmp_path):
    uploads = downloaded_stage3(tmp_path)
    run = si.make_ingested_run(tmp_path / "after", [*S.FILES, *S.PLAIN_FILES])
    assert "pas l'entretien choisi" in rejected(run, uploads, interview_id=S.PLAIN_ID)


def test_outputs_computed_on_another_transcript_are_rejected(tmp_path):
    """Ex. sorties calculées avant le correctif de segmentation : même nom de fichier, autres tours."""
    uploads = downloaded_stage3(tmp_path)
    changed = S.TEXT.replace("Je sais pas trop.", "Je ne sais pas trop.")
    run = fresh_run(tmp_path, [(S.FILENAME, changed.encode("utf-8"))])
    assert "autre transcription structurée" in rejected(run, uploads)


def test_quote_no_longer_matching_the_transcript_is_rejected(tmp_path):
    def tamper(document):
        document["practices"][0]["evidence"][0]["quote"] = "Citation qui n'est pas dans l'entretien."
    uploads = edited(downloaded_stage3(tmp_path), "practice_extractor.json", tamper)
    assert "ne correspondent plus au texte" in rejected(fresh_run(tmp_path), uploads)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(analysis_complete=False), "analysis_complete = false"),
    (lambda d: d.update(status="PARTIAL", analysis_complete=False), "statut PARTIAL non exploitable"),
    (lambda d: d.update(status="FAILED"), "statut FAILED non exploitable"),
    (lambda d: d.update(schema_version="0.9"), "schéma 0.9 incompatible"),
])
def test_incomplete_or_incompatible_agent_output_is_rejected(tmp_path, change, message):
    uploads = edited(downloaded_stage3(tmp_path), "interaction_signals.json", change)
    assert message in rejected(fresh_run(tmp_path), uploads)


def test_evidence_validation_with_critical_invalid_quotes_is_rejected(tmp_path):
    def critical(document):
        document["agents"]["practice_extractor"]["issues"].append(
            {"object_id": "X", "code": "UNKNOWN_TURN_ID", "severity": "error", "message": "turn_id inexistant."})
        document["total_invalid_evidence"] = 9
    uploads = edited(downloaded_stage3(tmp_path), "evidence_validation.json", critical)
    problems = rejected(fresh_run(tmp_path), uploads)
    assert "anomalie(s) critique(s)" in problems and "ne correspond pas aux sorties importées" in problems
    assert "Trop de citations invalides" in problems


def test_older_agent_version_is_accepted_with_a_warning(tmp_path):
    uploads = edited(downloaded_stage3(tmp_path), "practice_extractor.json", lambda d: d.update(agent_version="1.1"))
    checked = check_restore(fresh_run(tmp_path), S.INTERVIEW_ID, uploads)
    assert checked["warnings"] == ["Practice Extractor : version 1.1 (version actuelle 1.2)."]


# --- Interface ----------------------------------------------------------------------------------------

def test_app_offers_restore_then_shows_restored_stage3_and_stage4(tmp_path, monkeypatch):
    from tests.fake_llm import use_fake_runtime
    transport = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S.REFERENCE_BUILDER})  # seul l'agent de l'étape 4
    uploads = downloaded_stage3(tmp_path)
    run = fresh_run(tmp_path)
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=30)
    at.session_state["last_run"] = run
    at.run()
    assert not at.exception
    assert "Restaurer des résultats Stage 3 existants" in [e.label for e in at.expander]
    assert at.button(key="restore_launch").disabled  # aucun fichier choisi
    assert not [b for b in at.button if b.key == "run_stage4"]  # l'étape 4 attend l'étape 3

    at.session_state["last_run"] = restore_stage3(run, S.INTERVIEW_ID, uploads)
    at.run()
    assert "Stage 3 restauré depuis fichiers — 0 appel API" in " ".join(str(s.value) for s in at.success)
    assert "Restaurer des résultats Stage 3 existants" not in [e.label for e in at.expander]
    at.button(key="run_stage4").click().run()
    assert not at.exception and [c["agent"] for c in transport.calls] == [ACCOUNTABILITY] * 2  # deux blocs, aucune étape 3
    assert list(at.table[-1].value["Épisodes accountability"]) == [4]
