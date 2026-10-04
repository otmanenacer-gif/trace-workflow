"""Régression OTMANE_NACER : la resélection (`refilter`) est la version canonique de l'étape 3.

Bug observé : étape 3 à 33 pratiques / 23 signaux, `refilter` → 30 / 22 (routine campus / bibliothèque NO_AI_LINK,
fausse contradiction écartée). Streamlit rouvert : l'entretien réimporté (nouveau run) et l'étape 3 restaurée depuis
des JSON téléchargés AVANT le `refilter` ; la restauration les installait octet pour octet → l'étape 4 recevait de
nouveau 33 / 23 et reconstruisait un candidat sur la routine campus.

Ici : 33 / 23 → `refilter` → 30 / 22, puis réouverture du run (métadonnées relues sur le disque) ET restauration des
fichiers d'avant le `refilter` dans un run neuf : le constructeur de candidats de l'étape 4 reçoit exactement 30 / 22,
sans la routine campus ni la fausse contradiction. Faux Ollama, aucun réseau, aucun nouvel appel de l'étape 3.
"""

import copy
import json
from pathlib import Path

import pytest

from core import accountability, config, practice_selectivity, signal_selectivity, stage3_restore, trajectory
from core import local_pipeline as lp
from core.run_manager import load_metadata
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests.fake_llm import ACCOUNTABILITY, AUDITOR, INTERACTION, PRACTICE, text_response, use_fake_runtime

ID = S4.INTERVIEW_ID
CAMPUS_TURN = 27
TURNS = S4.TURNS + [
    ("Enquêteur", "Maintenant, raconte-moi une semaine type : où est-ce que tu travailles ?"),
    ("Enquêté", "Je me rends sur le campus, je travaille à la bibliothèque jusqu'à ce qu'elle ferme, puis je rentre "
                "chez moi."),
]
FILES = [(S4.FILENAME, ("\n".join(f"{s} : {t}" for s, t in TURNS) + "\n").encode("utf-8"))]
NAMES = ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json",
         "speaker_attribution_audit.json")


def tid(number: int) -> str:
    return f"{ID}_T{number:04d}"


CAMPUS = [si.practice(summary=f"L'étudiant travaille sur le campus, à la bibliothèque ({n}).", use_status=status,
                      non_use_reason="not_stated" if status == "non_use" else None, turn_start=tid(CAMPUS_TURN),
                      turn_end=tid(CAMPUS_TURN), context="semaine type sur le campus",
                      evidence=[{"turn_id": tid(CAMPUS_TURN), "quote": quote}])
          for n, (status, quote) in enumerate([
              ("non_use", "Je me rends sur le campus"),
              ("use", "je travaille à la bibliothèque jusqu'à ce qu'elle ferme"),
              ("non_use", "puis je rentre chez moi.")], start=1)]
FALSE_CONTRADICTION = si.signal(
    turn_ids=[tid(8), tid(14)], signal_type="cross_turn_contradiction", surface_form="contradiction",
    description="Deux passages présentés comme contradictoires.", topic="usage",
    cross_turn_reference="Rapprochement de deux tours.",
    evidence=[{"turn_id": tid(8), "quote": "Je lui demande des idées de lectures"},
              {"turn_id": tid(14), "quote": "J'aurais peur que le professeur pense que je n'ai rien fait."}])


def stage3_outputs() -> tuple[dict, dict]:
    """33 pratiques (30 liées à une IAG + 3 routines campus) et 23 signaux (22 + une fausse contradiction)."""
    base = copy.deepcopy(S4.STAGE3_PRACTICES["practices"])
    variants = [{**copy.deepcopy(base[i % len(base)]), "summary": f"{base[i % len(base)]['summary']} (variante {i})"}
                for i in range(18)]
    signals = [s for s in copy.deepcopy(S4.STAGE3_SIGNALS["signals"])
               if s["signal_type"] != "normative_formulation"]  # citation de l'enquêteur : écartée avant comme après
    substantive = [s for s in signals if s["signal_type"] in ("restriction", "explicit_emotion", "preference_statement")]
    signal_variants = [{**copy.deepcopy(substantive[i % len(substantive)]),
                        "description": f"{substantive[i % len(substantive)]['description']} (variante {i})"}
                       for i in range(13)]
    return ({**copy.deepcopy(S4.STAGE3_PRACTICES), "practices": [*base, *variants, *CAMPUS]},
            {**copy.deepcopy(S4.STAGE3_SIGNALS), "signals": [*signals, *signal_variants, FALSE_CONTRADICTION]})


def responders(practices: dict, signals: dict) -> dict:
    return {PRACTICE: lambda p: text_response(practices), INTERACTION: lambda p: text_response(signals),
            AUDITOR: lambda p: text_response(S4.STAGE3_AUDIT), ACCOUNTABILITY: S4.REFERENCE_BUILDER}


def interview_dir(run: dict) -> Path:
    return Path(run["files"][0]["ingestion"]["output_dir"])


def analysis_file(run: dict, name: str) -> dict:
    return json.loads((interview_dir(run) / config.ANALYSIS_SUBDIR / name).read_text(encoding="utf-8"))


def counts(run: dict) -> tuple[int, int, int, int]:
    practices, signals = analysis_file(run, NAMES[0]), analysis_file(run, NAMES[1])
    return (len(practices["practices"]), len(practices.get("set_aside_practices", [])),
            len(signals["signals"]), len(signals.get("set_aside_signals", [])))


def spy_candidate_builder(monkeypatch) -> list[tuple[list, list]]:
    """Ce que le constructeur de candidats de l'étape 4 reçoit réellement (pratiques, signaux)."""
    received = []
    real = accountability.candidates_mod.build_candidates

    def build(transcript, practices, signals, warnings):
        received.append((practices, signals))
        return real(transcript, practices, signals, warnings)

    monkeypatch.setattr(accountability.candidates_mod, "build_candidates", build)
    return received


def assert_stage4_receives_30_22(received: list) -> None:
    assert received  # préparation de l'étape 4 et contrôles de méthode : chaque appel reçoit la même étape 3
    for practices, signals in received:
        _check_30_22(practices, signals)


def _check_30_22(practices: list, signals: list) -> None:
    assert (len(practices), len(signals)) == (30, 22)
    cited = {e["turn_id"] for p in practices for e in p["evidence"]}
    assert tid(CAMPUS_TURN) not in cited
    assert not any("campus" in p["summary"] for p in practices)
    quotes = [e["quote"] for e in FALSE_CONTRADICTION["evidence"]]
    assert all([e["quote"] for e in s["evidence"]] != quotes for s in signals)
    assert not any(s["signal_type"] == "cross_turn_contradiction" and s["turn_ids"] == FALSE_CONTRADICTION["turn_ids"]
                   for s in signals)


@pytest.fixture
def refiltered(tmp_path, monkeypatch):
    """Étapes 3 et 4 calculées avec l'ANCIENNE sélectivité (33 / 23), fichiers de l'étape 3 « téléchargés », puis
    `refilter` sans modèle (30 / 22). Renvoie (run, téléchargements d'avant le refilter, empreintes d'avant)."""
    practices, signals = stage3_outputs()
    with monkeypatch.context() as old:
        old.setattr(practice_selectivity, "relevance_problem", lambda *a: None)
        old.setattr(practice_selectivity, "normalize_non_use_reason", lambda p: (p, None))
        old.setattr(signal_selectivity, "contradiction_problem", lambda s: None)
        use_fake_runtime(monkeypatch, responders(practices, signals))
        run = lp.run_until(si.make_ingested_run(tmp_path / "run", FILES), "4")["metadata"]
    assert counts(run) == (33, 0, 23, 0)
    assert lp.stage_complete("4", run["files"][0])
    analysis_dir = interview_dir(run) / config.ANALYSIS_SUBDIR
    downloads = []
    for name in NAMES:  # comme les « OTMANE_NACER_*.json (2) » : produits avant la sélectivité, sans ses champs
        document = json.loads((analysis_dir / name).read_text(encoding="utf-8"))
        for key in ("practice_selectivity", "set_aside_practices", "selectivity", "set_aside_signals"):
            document.pop(key, None)
        downloads.append((f"{ID}_{name}", json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")))
    hashes_before = trajectory.stage3_hashes(interview_dir(run))

    ollama = use_fake_runtime(monkeypatch, {})  # aucun répondeur : tout appel au modèle échouerait
    run = lp.refilter_stage3(run)["metadata"]
    assert ollama.calls == []
    assert counts(run) == (30, 3, 22, 1)
    return run, downloads, hashes_before


def test_refilter_changes_the_stage4_source_hashes_and_makes_stage4_stale(refiltered):
    run, _, before = refiltered
    after = trajectory.stage3_hashes(interview_dir(run))
    changed = {k for k in after if after[k] != before[k]}
    assert {"practice_extractor_sha256", "interaction_signals_sha256"} <= changed
    assert after["structured_transcript_sha256"] == before["structured_transcript_sha256"]
    state = trajectory.stage4_state(interview_dir(run))
    assert state["status"] == trajectory.STAGE4_STALE
    assert "practice_extractor_sha256" in state["reasons"][0]
    assert not lp.stage_complete("4", run["files"][0])  # la garde rejouera l'étape 4


def test_after_restart_stage4_receives_exactly_the_refiltered_30_practices_and_22_signals(refiltered, monkeypatch):
    run, _, _ = refiltered
    reopened = load_metadata(Path(run["output_dir"]))  # Streamlit rouvert / travail en arrière-plan : relu sur disque
    assert counts(reopened) == (30, 3, 22, 1)
    received = spy_candidate_builder(monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    result = lp.run_stage("4", reopened)
    assert result["status"]["skipped"] == []  # périmée : rejouée
    assert all(call["agent"] == ACCOUNTABILITY for call in ollama.calls)  # aucun appel de l'étape 3
    assert_stage4_receives_30_22(received)
    document = analysis_file(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert document["source_hashes"] == trajectory.stage3_hashes(interview_dir(run))
    assert lp.stage_complete("4", result["metadata"]["files"][0])


def test_restoring_files_downloaded_before_refilter_never_resurrects_set_aside_objects(refiltered, tmp_path,
                                                                                         monkeypatch):
    """Le chemin réel du bug : entretien réimporté (run neuf), étape 3 restaurée depuis les JSON d'avant le refilter."""
    run, downloads, _ = refiltered
    fresh = si.make_ingested_run(tmp_path / "reimported", FILES)
    fresh = stage3_restore.restore_stage3(fresh, ID, downloads)

    assert counts(fresh) == (30, 3, 22, 1)
    for name, key in ((NAMES[0], "practices"), (NAMES[1], "signals")):  # mêmes objets, mêmes identifiants
        assert analysis_file(fresh, name)[key] == analysis_file(run, name)[key]
    reasons = [p["set_aside_reason"] for p in analysis_file(fresh, NAMES[0])["set_aside_practices"]]
    assert reasons == [practice_selectivity.REASON_NO_AI_LINK] * 3
    [aside] = analysis_file(fresh, NAMES[1])["set_aside_signals"]
    assert aside["set_aside_reason"] == signal_selectivity.REASON_CONTRADICTION_OBJECT
    validation = analysis_file(fresh, config.EVIDENCE_VALIDATION_FILENAME)
    assert validation["agents"]["practice_extractor"]["object_count"] == 30
    assert validation["total_invalid_evidence"] == (analysis_file(fresh, NAMES[0])["invalid_evidence_count"]
                                                   + analysis_file(fresh, NAMES[1])["invalid_evidence_count"])
    manifest = analysis_file(fresh, stage3_restore.RESTORE_MANIFEST_FILENAME)
    assert manifest["files"]["practice_extractor.json"]["reselected"] is True
    assert manifest["reselection"]["practice_extractor.json"]["practices_kept"] == 30
    assert sorted(fresh["files"][0]["analysis"]["reselected"]) == sorted(NAMES[:3])

    # même après relecture du disque (redémarrage), l'étape 4 reçoit 30 / 22, sans aucun appel de l'étape 3
    received = spy_candidate_builder(monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    result = lp.run_stage("4", load_metadata(Path(fresh["output_dir"])))
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    assert all(call["agent"] == ACCOUNTABILITY for call in ollama.calls)
    assert_stage4_receives_30_22(received)
    # mêmes épisodes que l'étape 4 rejouée sur le run refiltré lui-même
    strip = lambda e: {k: v for k, v in e.items() if k not in ("validated_at", "generated_at")}  # noqa: E731
    episodes = [analysis_file(r, config.ACCOUNTABILITY_EPISODES_FILENAME)["episodes"]
                for r in (result["metadata"], lp.run_stage("4", run)["metadata"])]
    assert [strip(e) for e in episodes[0]] == [strip(e) for e in episodes[1]]


def test_restoring_files_downloaded_after_refilter_installs_them_byte_for_byte(refiltered, tmp_path):
    run, _, _ = refiltered
    analysis_dir = interview_dir(run) / config.ANALYSIS_SUBDIR
    downloads = [(f"{ID}_{name}", (analysis_dir / name).read_bytes()) for name in NAMES]
    fresh = stage3_restore.restore_stage3(si.make_ingested_run(tmp_path / "reimported", FILES), ID, downloads)
    for name, data in downloads:
        assert (interview_dir(fresh) / config.ANALYSIS_SUBDIR / name.removeprefix(f"{ID}_")).read_bytes() == data
    assert fresh["files"][0]["analysis"]["reselected"] == []
    assert counts(fresh) == (30, 3, 22, 1)
