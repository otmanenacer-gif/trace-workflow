"""Étape 4 — entretien long synthétique (320 tours) : beaucoup de pratiques ordinaires, peu d'épisodes.

Chaîne complète, LLM simulé : ingestion → étape 3 (audit, Practice Extractor et Interaction Reader par
blocs, lecture à longue distance) → candidats déterministes → UN appel Accountability Episode Builder
→ validation. Vérifie que l'étape 4 ne produit PAS un épisode par pratique. Les nombres ne sont pas des
quotas : ils décrivent ce que contient l'entretien synthétique.
"""

import json
from pathlib import Path

from core import accountability, config
from core.accountability_candidates import SINGLE_CALL_MAX_INPUT_TOKENS
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests import synthetic_stage4_long as L
from tests.fake_llm import ACCOUNTABILITY, FakeTransport, fake_settings
from tests.synthetic_stage4 import sent_payload, turn_number


def test_long_interview_does_not_turn_every_practice_into_an_episode(tmp_path):
    run = si.make_ingested_run(tmp_path, L.files())
    assert run["files"][0]["ingestion"]["turn_count"] == L.TURN_COUNT == 320
    cache = AnalysisCache(tmp_path / "cache")
    t3 = FakeTransport(L.stage3_responders())
    run = analyze_run(run, settings=fake_settings(), transport=t3, cache=cache)
    agents = run["files"][0]["analysis"]["agents"]
    assert agents["practice_extractor"]["status"] == agents["interaction_signal_reader"]["status"] == "SUCCESS"
    assert agents["practice_extractor"]["item_count"] == L.EXPECTED_PRACTICE_COUNT == 31
    signal_count = agents["interaction_signal_reader"]["item_count"]
    assert signal_count >= 70  # signaux locaux + contradiction à longue distance

    t4 = FakeTransport({ACCOUNTABILITY: L.BUILDER})
    run = accountability.analyze_run_stage4(run, settings=fake_settings(), transport=t4, cache=cache)
    out = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    doc = json.loads((out / config.ACCOUNTABILITY_EPISODES_FILENAME).read_text(encoding="utf-8"))
    manifest = json.loads((out / config.ACCOUNTABILITY_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    # Coût : UN appel pour tout l'entretien, sous le seuil d'un appel unique
    assert len(t4.calls) == 1 and manifest["api_calls"] == 1
    assert manifest["over_single_call_threshold"] is False
    assert manifest["estimated_input_tokens"] < SINGLE_CALL_MAX_INPUT_TOKENS / 3
    payload = sent_payload(t4.calls[0]["params"])
    assert len(payload["turns"]) < 25  # jamais l'entretien entier

    # Sélectivité : 31 pratiques, ≥ 70 signaux → 8 candidats → 5 épisodes + 3 pratiques ordinaires examinées
    summary = doc["candidates"]["summary"]
    assert summary["practice_count"] == 31 and summary["signal_count"] == signal_count
    assert doc["candidate_count"] == 8 < summary["practice_count"] // 3
    assert (doc["accountability_episode_count"], doc["ordinary_practice_count"], doc["uncertain_count"]) == (5, 3, 0)
    assert doc["unmarked_practice_count"] == 20
    assert summary["unattached_signal_count"] == 1  # signal fort sans pratique proche : aucun candidat
    assert summary["ignored_signal_count"] >= 50    # rires, hésitations, intensifications, généralisations isolées
    anchors = {min(turn_number(e["turn_id"]) for e in ep["evidence"]): ep["episode_status"] for ep in doc["episodes"]}
    assert {a for a, s in anchors.items() if s == "accountability_episode"} == set(L.ACCOUNTABILITY_ANCHORS)
    assert {a for a, s in anchors.items() if s == "ordinary_practice"} == set(L.ORDINARY_EXAMINED_ANCHORS)
    assert doc["validation_error_count"] == doc["validation_warning_count"] == 0

    # Contradiction à distance : les deux passages éloignés, rien entre eux
    [far] = [e for e in doc["episodes"] if len({turn_number(x["turn_id"]) for x in e["evidence"]} & {60, 280}) == 2]
    assert len(far["practice_ids"]) == 2 and turn_number(far["turn_start"]) < 60 and turn_number(far["turn_end"]) == 280
    candidate = next(c for c in doc["candidates"]["candidates"] if c["candidate_id"] in far["candidate_ids"])
    assert [turn_number(t) for t in candidate["turn_ids"]] == [59, 60, 279, 280]

    # Relance : 0 appel
    again = FakeTransport({ACCOUNTABILITY: L.BUILDER})
    accountability.analyze_run_stage4(run, settings=fake_settings(), transport=again, cache=cache)
    assert again.calls == []
