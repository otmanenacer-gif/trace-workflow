"""Test synthétique GLOBAL de l'étape 3.7 (entretien long de 330 tours), agents simulés.

Chaîne complète : ingestion → audit des locuteurs → Practice Extractor par blocs → Interaction
Reader par blocs + lecture à longue distance → fusion → sélectivité → validation. Le lecteur
Interaction simulé SURCODE (un signal par remplisseur) : le pire cas observé sur le vrai entretien.
Ce test vérifie la chaîne et les garde-fous déterministes ; il ne mesure PAS la qualité d'un vrai modèle.
"""

import hashlib
import json
from pathlib import Path

from core import config
from core.analysis import analyze_run, plan_analysis
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests import synthetic_stage37 as S
from tests.fake_llm import AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, FakeAgents, fake_settings, text_response
from tests.synthetic_long_interview import sent_turns


def transport():
    return FakeAgents({PRACTICE: S.practice_reader, INTERACTION: S.naive_reader,
                          LONG_DISTANCE: S.long_distance_reader,
                          AUDITOR: lambda p: text_response({"assessments": [], "audit_notes": None})})


def test_stage_3_7_long_synthetic_interview_end_to_end(tmp_path):
    run = si.make_ingested_run(tmp_path, S.files())
    ingestion = run["files"][0]["ingestion"]
    assert ingestion["status"] == "PASS" and ingestion["turn_count"] == S.TURN_COUNT == 330
    transcript_path = Path(ingestion["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME
    transcript_sha = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    cache = AnalysisCache(tmp_path / "cache")

    fake = transport()
    run = analyze_run(run, settings=fake_settings(), client=fake, cache=cache)
    out = Path(run["files"][0]["analysis"]["analysis_dir"])
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    practices, signals = load("practice_extractor.json"), load("interaction_signals.json")
    audit, validation = load("speaker_attribution_audit.json"), load("evidence_validation.json")

    # Coût : 1 audit + 4 blocs Practice + 3 blocs Interaction + 1 lecture à longue distance ; jamais un appel par tour
    assert {a: len(fake.calls_for(a)) for a in (AUDITOR, PRACTICE, INTERACTION, LONG_DISTANCE)} == {
        AUDITOR: 1, PRACTICE: 4, INTERACTION: 3, LONG_DISTANCE: 1}
    assert all(len(sent_turns(c["params"])) < S.TURN_COUNT for c in fake.calls)  # jamais l'entretien entier

    # Practice Extractor : plusieurs blocs, SUCCESS, usages / non-usages / refus conservés, pas de troncature
    assert practices["status"] == "SUCCESS" and practices["analysis_complete"] is True
    assert practices["chunking"]["chunk_count"] == 4 and practices["chunking"]["truncated_chunks"] == []
    assert practices["item_count"] == S.EXPECTED_PRACTICE_COUNT
    statuses = {p["use_status"] for p in practices["practices"]}
    assert {"use", "non_use", "refusal", "past_use"} <= statuses
    assert {p["practice_domain"] for p in practices["practices"]} == {"academic", "personal"}

    # Interaction Reader : plusieurs blocs, SUCCESS, signaux nettement moins nombreux que les micro-marqueurs
    info = signals["chunking"]
    assert signals["status"] == "SUCCESS" and info["chunk_count"] == 3 and info["long_distance"]["signals_kept"] == 2
    markers = S.micro_marker_count()
    assert info["signals_before_dedup"] > markers > 600 and signals["item_count"] * 20 < markers
    found = {(s["signal_type"], s["turn_ids"][0]) for s in signals["signals"]}
    for expected in (("self_correction", S.tid(10)), ("exception", S.tid(10)), ("preference_statement", S.tid(40)),
                     ("metadiscursive_self_evaluation", S.tid(60)), ("explicit_emotion", S.tid(80)),
                     ("reference_to_teacher_judgment", S.tid(80)), ("normative_formulation", S.tid(120)),
                     ("cross_turn_contradiction", S.tid(24)), ("cross_turn_contradiction", S.tid(70))):
        assert expected in found, expected

    # Preuves : toutes valides, aucune inventée ; aucune anomalie de validation
    assert validation["total_invalid_evidence"] == 0
    assert all(e["validation"]["valid"] for p in practices["practices"] for e in p["evidence"])
    assert all(e["validation"]["valid"] for s in signals["signals"] for e in s["evidence"])
    assert all(a["warning_count"] == a["error_count"] == 0 for a in validation["agents"].values())

    # Speaker Attribution Auditor : comportement inchangé, non destructif, avertissements seulement
    assert audit["transcript_modified"] is False and audit["llm_called"] is True
    assert {i["turn_id"] for i in audit["items"]} == {S.tid(n) for n in S.WARNED_TURNS}
    assert hashlib.sha256(transcript_path.read_bytes()).hexdigest() == transcript_sha
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert all(transcript["turns"][n - 1]["speaker"] == "enqueteur" for n in S.WARNED_TURNS)
    for call in fake.calls_for(PRACTICE) + fake.calls_for(INTERACTION):  # avertissements : seulement les blocs concernés
        sent = sent_turns(call["params"])
        flagged = {t["turn_id"] for t in sent if "speaker_warning" in t}
        assert flagged == {S.tid(n) for n in S.WARNED_TURNS} & {t["turn_id"] for t in sent}

    # Relance identique : 0 appel, pour tous les agents
    assert plan_analysis(run, [S.INTERVIEW_ID], fake_settings(), cache)["calls"] == 0
    again = transport()
    run = analyze_run(run, settings=fake_settings(), client=again, cache=cache)
    assert again.calls == [] and run["last_analysis"]["usage"]["api_calls"] == 0
    assert json.loads((out / "interaction_signals.json").read_text())["signals"] == signals["signals"]
