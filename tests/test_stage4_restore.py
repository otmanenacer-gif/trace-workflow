"""Restauration de sorties de l'étape 4 déjà calculées, puis étape 5 sans relancer les étapes 3 et 4. LLM simulé."""

import json
from pathlib import Path

import pytest

from core import config, stage4_restore, trajectory
from core.analysis_cache import AnalysisCache
from core.stage3_restore import RestoreError, restore_stage3
from core.stage4_restore import check_restore_stage4, restore_stage4
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests.fake_llm import TRAJECTORY, fake_settings

STAGE3_NAMES = ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json",
                "speaker_attribution_audit.json")
STAGE4_NAMES = (config.ACCOUNTABILITY_EPISODES_FILENAME, config.ACCOUNTABILITY_VALIDATION_FILENAME)
PREFIX = S4.INTERVIEW_ID + "_"


def downloaded(tmp_path, files=S4.FILES, responders=None, builder=S4.REFERENCE_BUILDER):
    """Étapes 3 et 4 calculées (LLM simulé) dans un premier run, puis « téléchargées » (noms préfixés comme l'UI)."""
    run, _ = S5.run_to_stage4(tmp_path / "before", files, responders or S4.stage3_responders(), builder)
    out = S5.analysis_dir(run)
    return ([(PREFIX + n, (out / n).read_bytes()) for n in STAGE3_NAMES],
            [(PREFIX + n, (out / n).read_bytes()) for n in STAGE4_NAMES])


def after_redeploy(tmp_path, stage3_uploads, files=S4.FILES):
    """Nouveau run (stockage effacé) : même entretien réimporté, étape 3 restaurée, aucune étape 4."""
    run = si.make_ingested_run(tmp_path / "after", files)
    return restore_stage3(run, S4.INTERVIEW_ID, stage3_uploads)


def edited(uploads, name, change):
    result = []
    for upload_name, data in uploads:
        if upload_name.endswith(name):
            document = json.loads(data)
            change(document)
            data = json.dumps(document, ensure_ascii=False).encode("utf-8")
        result.append((upload_name, data))
    return result


def rejected(run, uploads, interview_id=S4.INTERVIEW_ID) -> str:
    with pytest.raises(RestoreError) as info:
        restore_stage4(run, interview_id, uploads)
    assert not (S5.analysis_dir(run) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()  # rien n'est installé
    return " ".join(info.value.problems)


# --- Restauration réussie ---------------------------------------------------------------------------------

def test_restored_stage4_feeds_stage5_with_zero_stage3_or_stage4_call(tmp_path):
    stage3, stage4 = downloaded(tmp_path)
    run = after_redeploy(tmp_path, stage3)
    cache = AnalysisCache(tmp_path / "after_cache")  # cache TRACE vide, comme après un redéploiement
    assert trajectory.plan_stage5(run, [S4.INTERVIEW_ID], fake_settings(), cache)["blocked"] == 1

    run = restore_stage4(run, S4.INTERVIEW_ID, stage4)
    analysis_dir = S5.analysis_dir(run)
    for name, data in stage4:  # octet pour octet
        assert (analysis_dir / name.removeprefix(PREFIX)).read_bytes() == data
    state = trajectory.stage4_state(analysis_dir.parent)
    assert state["status"] == "COMPLETE" and state["restored"] is True
    summary = run["files"][0]["accountability"]
    assert summary["restored"] is True and summary["api_calls"] == 0 and summary["status"] == "SUCCESS"
    assert (summary["accountability_episode_count"], summary["unmarked_practice_count"]) == (4, 4)
    assert run["pipeline"][config.ACCOUNTABILITY_STEP] == "terminé (1/1 entretien(s))"
    manifest = json.loads((analysis_dir / stage4_restore.RESTORE_MANIFEST_FILENAME).read_text())
    assert manifest["api_calls"] == 0 and manifest["notice"] == "Stage 4 restauré depuis fichiers — 0 appel API"
    assert set(manifest["files"]) == set(STAGE4_NAMES)

    plan = trajectory.plan_stage5(run, [S4.INTERVIEW_ID], fake_settings(), cache)
    assert plan["blocked"] == 0 and plan["calls"] == 1 and plan["per_interview"][0]["stage4_restored"] is True
    run, transport = S5.run_stage5(run, cache, S5.scripted_mapper(S5.REFERENCE_PLAN))
    assert [c["agent"] for c in transport.calls] == [TRAJECTORY]  # ni étape 3, ni audit, ni étape 4
    result = run["files"][0]["trajectory"]
    assert result["status"] == "SUCCESS" and result["stage4_restored"] is True


def test_restored_stage4_gives_the_same_stage5_as_the_original(tmp_path):
    """Même matériau que le calcul initial : même requête, donc même clé de cache (0 appel la seconde fois)."""
    stage3, stage4 = downloaded(tmp_path)
    before, cache = S5.run_to_stage4(tmp_path / "orig", S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    before, first = S5.run_stage5(before, cache, S5.scripted_mapper(S5.REFERENCE_PLAN))
    run = restore_stage4(after_redeploy(tmp_path, stage3), S4.INTERVIEW_ID, stage4)
    run, second = S5.run_stage5(run, cache, S5.scripted_mapper(S5.REFERENCE_PLAN))
    assert len(first.calls) == 1 and second.calls == [] and run["files"][0]["trajectory"]["status"] == "CACHED"


def test_files_are_recognized_by_content_and_stale_stage5_outputs_are_removed(tmp_path):
    stage3, stage4 = downloaded(tmp_path)
    run = after_redeploy(tmp_path, stage3)
    stale = S5.analysis_dir(run) / config.STUDENT_TRAJECTORY_FILENAME
    stale.write_text("{}", encoding="utf-8")
    run["files"][0]["trajectory"] = {"status": "SUCCESS"}
    renamed = [(f"fichier_{i}.json", data) for i, (_, data) in enumerate(reversed(stage4))]
    checked = check_restore_stage4(run, S4.INTERVIEW_ID, renamed)
    assert set(checked["files"]) == set(STAGE4_NAMES) and checked["warnings"] == []
    run = restore_stage4(run, S4.INTERVIEW_ID, renamed)
    assert not stale.exists() and "trajectory" not in run["files"][0]


# --- Refus ---------------------------------------------------------------------------------------------------

def test_missing_invalid_duplicate_or_unknown_files_are_rejected(tmp_path):
    stage3, stage4 = downloaded(tmp_path)
    run = after_redeploy(tmp_path, stage3)
    assert "Fichier manquant : accountability_episode_validation.json" in rejected(run, stage4[:1])
    assert "JSON invalide" in rejected(run, [stage4[0], (stage4[1][0], b'{"issues": [')])
    assert "fourni deux fois" in rejected(run, stage4 + [stage4[0]])
    assert "non reconnu" in rejected(run, stage4 + [("autre.json", b'{"a": 1}')])
    assert "non reconnu" in rejected(run, stage4[:1] + [(PREFIX + "practice_extractor.json", stage3[0][1])])


def test_stage3_must_be_available_first(tmp_path):
    _, stage4 = downloaded(tmp_path)
    run = si.make_ingested_run(tmp_path / "after", S4.FILES)
    assert "restaurez (ou relancez) d'abord l'étape 3" in rejected(run, stage4)


def test_files_of_another_interview_or_mixed_interviews_are_rejected(tmp_path):
    stage3, stage4 = downloaded(tmp_path)
    run = si.make_ingested_run(tmp_path / "after", [*S4.FILES, *S4.PLAIN_FILES])
    run = restore_stage3(run, S4.INTERVIEW_ID, stage3)
    assert "pas l'entretien choisi" in rejected(run, stage4, interview_id=S4.PLAIN_ID)
    mixed = edited(stage4, config.ACCOUNTABILITY_VALIDATION_FILENAME, lambda d: d.update(interview_id="AUTRE"))
    assert "plusieurs entretiens" in rejected(run, mixed)


@pytest.mark.parametrize("name, change, message", [
    (config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(schema_version="0.9"), "schema_version = 0.9"),
    (config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(validator_version="1.0"), "validator_version = 1.0"),
    (config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(status="PARTIAL", analysis_complete=False),
     "Statut PARTIAL non exploitable"),
    (config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(analysis_complete=False), "analysis_complete = false"),
    (config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(stage3_status="PARTIAL"), "étape 3 PARTIAL"),
    (config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(validation_error_count=1), "validation_error_count = 1"),
    (config.ACCOUNTABILITY_VALIDATION_FILENAME, lambda d: d.update(error_count=2), "validation_error_count = 0 / 2"),
    (config.ACCOUNTABILITY_VALIDATION_FILENAME, lambda d: d.update(status="FAILED"), "Fichier de validation incohérent"),
])
def test_incompatible_incomplete_or_erroneous_stage4_is_rejected(tmp_path, name, change, message):
    stage3, stage4 = downloaded(tmp_path)
    assert message in rejected(after_redeploy(tmp_path, stage3), edited(stage4, name, change))


def test_stage4_computed_on_other_stage3_outputs_is_rejected(tmp_path):
    """Étape 4 d'un premier calcul, étape 3 restaurée d'un autre calcul (autres signaux) : empreintes différentes."""
    _, stage4 = downloaded(tmp_path / "one")
    other_signals = json.loads(json.dumps(S4.STAGE3_SIGNALS))
    other_signals["signals"] = other_signals["signals"][:-1]
    from tests.fake_llm import INTERACTION, text_response
    responders = {**S4.stage3_responders(), INTERACTION: lambda p: text_response(other_signals)}
    stage3_other, _ = downloaded(tmp_path / "two", responders=responders)
    problems = rejected(after_redeploy(tmp_path, stage3_other), stage4)
    assert "Empreintes différentes" in problems and "interaction_signals_sha256" in problems


@pytest.mark.parametrize("change, message", [
    (lambda d: d["episodes"][0]["evidence"][0].update(quote="Citation qui n'est pas dans l'entretien."),
     "ne correspondent plus au texte"),
    (lambda d: d["episodes"][0].update(episode_status="ordinary_practice"), "re-validation par TRACE"),
    (lambda d: next(e for e in d["episodes"] if e["review_reasons"]).update(review_reasons=[], needs_review=False),
     "re-validation par TRACE"),
    (lambda d: d["episodes"][0].update(usable_for_next_stages=False), "re-validation par TRACE"),
    (lambda d: d["episodes"].pop(), "Comptes du fichier d'épisodes incohérents"),
    (lambda d: d.update(accountability_episode_count=5), "Comptes du fichier d'épisodes incohérents"),
    (lambda d: d["unmarked_practices"].pop(), "Pratiques sans marqueur différentes"),
    (lambda d: d["candidates"]["candidates"][0]["practice_ids"].append("X"), "Candidats différents"),
])
def test_altered_episodes_file_is_rejected(tmp_path, change, message):
    stage3, stage4 = downloaded(tmp_path)
    problems = rejected(after_redeploy(tmp_path, stage3), edited(stage4, config.ACCOUNTABILITY_EPISODES_FILENAME, change))
    assert message in problems


def test_altered_validation_file_is_rejected(tmp_path):
    stage3, stage4 = downloaded(tmp_path)
    run = after_redeploy(tmp_path, stage3)
    no_issue = edited(stage4, config.ACCOUNTABILITY_VALIDATION_FILENAME, lambda d: d.update(issues=[]))
    assert "Anomalies du fichier de validation" in rejected(run, no_issue)
    episodes = edited(stage4, config.ACCOUNTABILITY_VALIDATION_FILENAME, lambda d: d.update(episode_count=9))
    assert "(episode_count)" in rejected(run, episodes)


def test_other_prompt_version_is_accepted_with_a_warning(tmp_path):
    stage3, stage4 = downloaded(tmp_path)
    changed = edited(stage4, config.ACCOUNTABILITY_EPISODES_FILENAME, lambda d: d.update(agent_version="1.0"))
    checked = check_restore_stage4(after_redeploy(tmp_path, stage3), S4.INTERVIEW_ID, changed)
    assert checked["warnings"] == ["Accountability Episode Builder : version 1.0 (version actuelle 1.1)."]


def test_restoration_never_calls_any_llm(tmp_path, monkeypatch):
    import core.llm_client as llm_client

    def forbidden(*args, **kwargs):
        raise AssertionError("Appel LLM pendant une restauration")
    stage3, stage4 = downloaded(tmp_path)
    monkeypatch.setattr(llm_client.LLMClient, "complete_json", forbidden)
    monkeypatch.setattr(llm_client.LLMClient, "__init__", forbidden)
    run = restore_stage4(after_redeploy(tmp_path, stage3), S4.INTERVIEW_ID, stage4)
    assert Path(run["files"][0]["accountability"]["analysis_dir"]).is_dir()
