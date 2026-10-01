"""Test synthétique GLOBAL de l'étape 3.5 (section 12), LLM simulé.

Un entretien fictif contient : un usage de rédaction, la revendication ultérieure
d'écrire soi-même, un refus explicite de déléguer, une préférence, « c'est assez
ridicule ce que je dis », la réparation d'un lave-vaisselle et un tour
volontairement mal attribué (réponse de l'enquêté marquée « Enquêteur »).

Ce test vérifie la chaîne (ingestion → audit des locuteurs → deux agents →
validation) et les garde-fous déterministes ; il ne mesure PAS la qualité d'un
vrai modèle.
"""

import hashlib
import json
from pathlib import Path

from core import config
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests.fake_llm import AUDITOR, INTERACTION, PRACTICE, FakeTransport, fake_settings, text_response


def test_stage_3_5_synthetic_interview_end_to_end(tmp_path):
    run = si.make_ingested_run(tmp_path, si.STAGE35_FILES)
    assert run["files"][0]["ingestion"]["status"] == "PASS"
    interview_dir = Path(run["files"][0]["ingestion"]["output_dir"])
    transcript_path = interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME
    transcript_sha = hashlib.sha256(transcript_path.read_bytes()).hexdigest()

    transport = FakeTransport({PRACTICE: text_response(si.STAGE35_PRACTICES),
                               INTERACTION: text_response(si.STAGE35_SIGNALS),
                               AUDITOR: text_response(si.STAGE35_AUDIT)})
    run = analyze_run(run, settings=fake_settings(), transport=transport, cache=AnalysisCache(tmp_path / "cache"))
    out = interview_dir / config.ANALYSIS_SUBDIR
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    practices, signals = load("practice_extractor.json")["practices"], load("interaction_signals.json")["signals"]
    audit, validation = load("speaker_attribution_audit.json"), load("evidence_validation.json")

    # Coût : un seul appel d'audit (un seul tour suspect), puis les deux agents
    assert [c["agent"] for c in transport.calls].count(AUDITOR) == 1 and len(transport.calls) == 3
    assert validation["total_invalid_evidence"] == 0

    # Practice Extractor : usage ET non-usage / refus, distincts
    statuses = [(p["use_status"], p["non_use_reason"]) for p in practices]
    assert ("use", None) in statuses and ("non_use", "preference") in statuses
    assert ("refusal", "personal_rule") in statuses and len(practices) == 4
    redaction_use, written_myself = practices[0], practices[1]
    assert redaction_use["practice_id"] != written_myself["practice_id"]
    dishwasher = practices[3]
    assert dishwasher["practice_domain"] == "personal"  # usage personnel conservé, pas filtré

    # Interaction Reader : préférence et évaluation métadiscursive, pas « other »
    kinds = [s["signal_type"] for s in signals]
    assert "preference_statement" in kinds and "metadiscursive_self_evaluation" in kinds and "other" not in kinds
    meta = next(s for s in signals if s["signal_type"] == "metadiscursive_self_evaluation")
    assert meta["surface_form"] == "c'est assez ridicule ce que je dis" and meta["review_reasons"] == []

    # Garde-fou : aucune alerte sur « réparation du lave-vaisselle »
    assert "réparation du lave-vaisselle" in dishwasher["context"].casefold()
    issues = [i for agent in validation["agents"].values() for i in agent["issues"]]
    assert not [i for i in issues if i["code"] == "INTERPRETIVE_VOCABULARY"]
    assert not [i for i in audit["issues"] if i["code"] == "INTERPRETIVE_VOCABULARY"]

    # Speaker Auditor : détecte le tour mal attribué, ne modifie pas le transcript
    item, = audit["items"]
    assert item["turn_id"] == si.STAGE35_MISATTRIBUTED_TURN
    assert (item["current_speaker"], item["suggested_speaker"], item["confidence"]) == ("enqueteur", "enquete", "high")
    assert item["needs_review"] and audit["candidate_count"] == 1 and audit["review_count"] == 1
    assert hashlib.sha256(transcript_path.read_bytes()).hexdigest() == transcript_sha
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    turn = next(t for t in transcript["turns"] if t["turn_id"] == si.STAGE35_MISATTRIBUTED_TURN)
    assert turn["speaker"] == "enqueteur" and "réparation du lave-vaisselle" in turn["text"]

    # L'avertissement est transmis aux deux agents (identiquement), le speaker officiel reste « enqueteur »
    contents = {c["agent"]: c["params"]["messages"][0]["content"] for c in transport.calls}
    assert contents[PRACTICE] == contents[INTERACTION]
    sent = json.loads(contents[PRACTICE][contents[PRACTICE].index("<transcript>\n") + 13:
                                         contents[PRACTICE].rindex("\n</transcript>")])
    flagged = [t for t in sent["turns"] if "speaker_warning" in t]
    assert [(t["turn_id"], t["speaker"], t["speaker_warning"]) for t in flagged] == [
        (si.STAGE35_MISATTRIBUTED_TURN, "enqueteur", {"suggested_speaker": "enquete", "confidence": "high"})]
    # La pratique qui s'appuie sur ce tour est marquée à revoir (sans rien corriger)
    assert "EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN" in dishwasher["review_reasons"]
