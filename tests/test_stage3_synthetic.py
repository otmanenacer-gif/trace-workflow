"""Test qualitatif SYNTHÉTIQUE de l'étape 3 (section 18 du cahier des charges).

Les agents sont simulés : ce test vérifie la chaîne complète (ingestion → deux agents
→ validation) et les garde-fous déterministes sur un entretien fictif. Il ne
mesure PAS la qualité des réponses d'un vrai agent : seule la lecture des sorties produites
par le modèle local (docs/local_runtime.md) sur des entretiens réels le permet.
"""

import json
from pathlib import Path

from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from core.interpretation_guard import fold
from tests import synthetic_interviews as si
from tests.fake_llm import INTERACTION, PRACTICE, FakeAgents, fake_settings, text_response

# Ce que l'étape 3 ne doit PAS conclure (réservé à l'étape interprétative).
FORBIDDEN_CONCLUSIONS = ("honte", "identite menacee", "reparation", "breach", "strategie defensive", "accountab",
                         "culpabil", "triche", "dependan")


def run_synthetic(tmp_path, practices=si.GOOD_PRACTICES, signals=si.GOOD_SIGNALS):
    run = si.make_ingested_run(tmp_path)
    assert run["files"][0]["ingestion"]["status"] == "PASS"
    transport = FakeAgents({PRACTICE: text_response(practices), INTERACTION: text_response(signals)})
    run = analyze_run(run, settings=fake_settings(), client=transport, cache=AnalysisCache(tmp_path / "cache"))
    out = Path(run["files"][0]["analysis"]["analysis_dir"])
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    return load("practice_extractor.json"), load("interaction_signals.json"), load("evidence_validation.json")


def authored_text(items, fields):
    return fold(" ".join(json.dumps([item.get(f) for f in fields], ensure_ascii=False) for item in items))


def test_synthetic_interview_end_to_end(tmp_path):
    practices, signals, validation = run_synthetic(tmp_path)
    assert validation["total_invalid_evidence"] == 0 and validation["total_objects_needing_review"] == 0

    # Practice Extractor : pratiques descriptives
    statuses = {p["use_status"] for p in practices["practices"]}
    assert {"use", "non_use", "hypothetical"} <= statuses
    tasks = " ".join(p["academic_task"] or "" for p in practices["practices"])
    assert "dissertation" in tasks and "comprendre un mot" in tasks
    assert all(p["summary"].startswith("L'étudiante") for p in practices["practices"])  # discours rapporté
    text = authored_text(practices["practices"], ("summary", "context", "stated_reason", "ai_action"))
    for concept in FORBIDDEN_CONCLUSIONS + ("scrupule",):  # l'affect appartient à l'autre agent
        assert concept not in text, concept

    # Interaction Signal Reader : signaux observables
    kinds = {s["signal_type"] for s in signals["signals"]}
    assert {"explicit_emotion", "self_correction", "minimization", "hesitation", "reference_to_teacher_judgment",
            "cross_turn_contradiction", "transcribed_laughter"} <= kinds
    affects = [s["explicit_affect"] for s in signals["signals"] if s["explicit_affect"]]
    assert affects == ["scrupules"]
    text = authored_text(signals["signals"], ("description", "topic", "explicit_affect", "cross_turn_reference"))
    for concept in FORBIDDEN_CONCLUSIONS:
        assert concept not in text, concept


def test_prompt_injection_in_transcript_is_only_data(tmp_path):
    practices, signals, _ = run_synthetic(tmp_path)
    # L'injection (T0010) n'a produit ni pratique inventée ni changement de format
    assert all(p["turn_start"] != si.tid(10) for p in practices["practices"])
    assert {s["signal_type"] for s in signals["signals"] if si.tid(10) in s["turn_ids"]} == {
        "transcribed_laughter", "generalization"}


def test_forbidden_conclusions_are_flagged_by_the_guard(tmp_path):
    _, _, validation = run_synthetic(tmp_path, practices=si.INTERPRETIVE_PRACTICES, signals=si.INTERPRETIVE_SIGNALS)
    for agent in ("practice_extractor", "interaction_signal_reader"):
        report = validation["agents"][agent]
        assert report["objects_needing_review"] and any(i["code"] == "INTERPRETIVE_VOCABULARY" for i in report["issues"])


def test_patch_scope_qualifier_and_preference_statement(tmp_path):
    """Correctif de l'étape 3 sur un mini-entretien synthétique (agents simulés) :
    « je l'utilise surtout pour reformuler » → scope_qualifier, pas stated_frequency ;
    « je préfère faire mes plans moi-même » → preference_statement, pas other."""
    run = si.make_ingested_run(tmp_path, si.PATCH_FILES)
    assert run["files"][0]["ingestion"]["status"] == "PASS"
    transport = FakeAgents({PRACTICE: text_response(si.PATCH_PRACTICES),
                               INTERACTION: text_response(si.PATCH_SIGNALS)})
    run = analyze_run(run, settings=fake_settings(), client=transport, cache=AnalysisCache(tmp_path / "cache"))
    out = Path(run["files"][0]["analysis"]["analysis_dir"])
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    practices, signals = load("practice_extractor.json"), load("interaction_signals.json")
    assert load("evidence_validation.json")["total_invalid_evidence"] == 0

    # Les consignes et schémas réellement envoyés portent les deux corrections
    practice_call, = transport.calls_for(PRACTICE)
    interaction_call, = transport.calls_for(INTERACTION)
    assert "`scope_qualifier`" in practice_call["params"]["system"][0]["text"]
    assert "scope_qualifier" in practice_call["params"]["output_schema"]["$defs"]["Practice"]["required"]
    assert "`preference_statement`" in interaction_call["params"]["system"][0]["text"]
    assert "preference_statement" in json.dumps(interaction_call["params"]["output_schema"])

    surtout = next(p for p in practices["practices"] if "surtout" in p["evidence"][0]["quote"])
    assert surtout["stated_frequency"] is None and surtout["scope_qualifier"] == "surtout"
    assert all(p["stated_frequency"] is None for p in practices["practices"])

    prefere = [s for s in signals["signals"] if "je préfère" in s["evidence"][0]["quote"]]
    assert [s["signal_type"] for s in prefere] == ["preference_statement"]
    assert not [s for s in signals["signals"] if s["signal_type"] == "other"]
    assert all(not s["needs_review"] for s in signals["signals"])
