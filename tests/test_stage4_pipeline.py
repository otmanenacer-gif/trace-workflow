"""Étape 4 — orchestration : entretien de référence, cache, étape 3 FAILED / PARTIAL, coût. LLM simulé."""

import dataclasses
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core import accountability, config
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from core.llm_client import LLMError, LLMSettings
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S
from tests.fake_llm import (ACCOUNTABILITY, AUDITOR, INTERACTION, PRACTICE, FakeTransport, fake_settings,
                            server_error, text_response)

ROOT = Path(__file__).resolve().parent.parent


def stage3(tmp_path, files=S.FILES, responders=None):
    run = si.make_ingested_run(tmp_path, files)
    cache = AnalysisCache(tmp_path / "cache")
    transport = FakeTransport(responders or S.stage3_responders())
    run = analyze_run(run, settings=fake_settings(), transport=transport, cache=cache)
    return run, cache, transport


def stage4(run, cache, builder=S.REFERENCE_BUILDER, **kwargs):
    transport = FakeTransport({ACCOUNTABILITY: builder})
    run = accountability.analyze_run_stage4(run, settings=kwargs.pop("settings", fake_settings()), transport=transport,
                                            cache=cache, **kwargs)
    return run, transport


def out_dir(run) -> Path:
    return Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR


def load(run, name):
    return json.loads((out_dir(run) / name).read_text(encoding="utf-8"))


def episodes_by_anchor(document) -> dict[int, dict]:
    practices = {}
    for c in document["candidates"]["candidates"]:
        for pid in c["practice_ids"]:
            practices[pid] = c
    result = {}
    for e in document["episodes"]:
        turns = [S.turn_number(ev["turn_id"]) for ev in e["evidence"]]
        result[min(turns)] = e
    return result


# --- Entretien synthétique de référence -----------------------------------------------------------

def test_reference_interview_end_to_end(tmp_path):
    run, cache, t3 = stage3(tmp_path)
    assert [c["agent"] for c in t3.calls].count(AUDITOR) == 1  # tour mal attribué (T0022) audité
    files_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out_dir(run).glob("*.json")}
    transcript = Path(run["files"][0]["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME
    transcript_sha = hashlib.sha256(transcript.read_bytes()).hexdigest()

    run, t4 = stage4(run, cache)
    assert [c["agent"] for c in t4.calls] == [ACCOUNTABILITY]  # 1 appel, l'étape 3 n'est jamais relancée
    doc = load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert doc["status"] == "SUCCESS" and doc["analysis_complete"] is True and doc["stage3_status"] == "COMPLETE"
    assert (doc["candidate_count"], doc["episode_count"], doc["accountability_episode_count"],
            doc["ordinary_practice_count"], doc["uncertain_count"], doc["unmarked_practice_count"]) == (6, 6, 4, 1, 1, 4)
    assert doc["validation_error_count"] == doc["validation_warning_count"] == 0

    by_anchor = episodes_by_anchor(doc)
    assert {n: e["episode_status"] for n, e in by_anchor.items()} == S.EXPECTED_STATUS_BY_ANCHOR
    # A et F : racontées comme allant de soi, pratiques ordinaires sans marqueur, jamais envoyées au modèle
    unmarked = {S.turn_number(u["turn_end"]): u for u in doc["unmarked_practices"]}
    assert sorted(unmarked) == list(S.EXPECTED_UNMARKED_TURNS)
    assert {u["episode_status"] for u in unmarked.values()} == {"ordinary_practice"}
    # B : restriction + refus regroupés, frontière explicite
    b = by_anchor[4]
    assert {m["type"] for m in b["accounting_moves"]} >= {"restriction", "refusal"} and len(b["practice_ids"]) == 2
    assert "écrire / reformuler" in b["boundary_objects"]
    # C : règle générale / exception, deux citations éloignées conservées, raison reprise telle quelle
    c = by_anchor[6]
    assert [m["type"] for m in c["accounting_moves"]] == ["general_rule", "exception"]
    assert {ev["turn_id"] for ev in c["evidence"]} == {S.tid(6), S.tid(12)}
    assert "parce que j'étais en retard" in c["episode_summary"]
    assert any(doc_sig.endswith("S002") for doc_sig in c["signal_ids"])  # la contradiction à distance
    # D : jugement d'autrui ; E : auto-évaluation ; G : candidat examiné → pratique ordinaire
    assert "appeal_to_external_judgment" in {m["type"] for m in by_anchor[14]["accounting_moves"]}
    assert by_anchor[14]["external_reference"] == "le professeur"
    assert [m["type"] for m in by_anchor[16]["accounting_moves"]] == ["self_evaluation"]
    assert by_anchor[20]["accounting_moves"] == [] and by_anchor[20]["accountability_problem"] is None
    # H : locuteur douteux → incertain, à revoir, avertissement propagé
    h = by_anchor[22]
    assert h["confidence"] == "low" and h["needs_review"] is True
    assert h["speaker_warnings"] == [{"turn_id": S.tid(22), "suggested_speaker": "enquete", "confidence": "high"}]
    # I : la question de l'enquêteur (« triche ») n'est dans aucun candidat ni aucun épisode
    assert all(S.tid(23) not in c["turn_ids"] for c in doc["candidates"]["candidates"])
    assert "triche" not in json.dumps(doc["episodes"], ensure_ascii=False)

    # L'étape 3 et le transcript ne sont jamais modifiés
    assert hashlib.sha256(transcript.read_bytes()).hexdigest() == transcript_sha
    for name, sha in files_before.items():
        assert hashlib.sha256((out_dir(run) / name).read_bytes()).hexdigest() == sha, name

    manifest = load(run, config.ACCOUNTABILITY_MANIFEST_FILENAME)
    validation = load(run, config.ACCOUNTABILITY_VALIDATION_FILENAME)
    assert manifest["agent_version"] == "1.0" and manifest["model"] == "fake-model" and manifest["cache_hit"] is False
    assert set(manifest["source_hashes"]) == {"source_sha256", "structured_transcript_sha256",
                                              "practice_extractor_sha256", "interaction_signals_sha256",
                                              "speaker_audit_sha256"}
    assert manifest["api_calls"] == 1 and manifest["over_single_call_threshold"] is False
    assert validation["status"] == "SUCCESS" and validation["candidate_summary"]["candidate_count"] == 6
    assert run["pipeline"][config.ACCOUNTABILITY_STEP] == "terminé (1/1 entretien(s))"
    assert run["last_accountability"]["usage"]["api_calls"] == 1


def test_llm_receives_only_candidates_never_the_whole_interview(tmp_path):
    run, cache, _ = stage3(tmp_path)
    run, t4 = stage4(run, cache)
    params = t4.calls[0]["params"]
    payload = S.sent_payload(params)
    sent_turns = {t["turn_id"] for t in payload["turns"]}
    assert len(sent_turns) < len(S.TURNS) and S.tid(2) not in sent_turns and S.tid(18) not in sent_turns
    assert params["system"][0]["text"].startswith("# Accountability Episode Builder")
    assert len(params["messages"]) == 1
    warned = [t for t in payload["turns"] if "speaker_warning" in t]
    assert [t["turn_id"] for t in warned] == [S.tid(22)] and warned[0]["speaker"] == "enqueteur"


# --- Cache ------------------------------------------------------------------------------------------

def test_identical_interview_is_served_from_cache_with_zero_call(tmp_path):
    run, cache, _ = stage3(tmp_path)
    run, first = stage4(run, cache)
    assert len(first.calls) == 1
    assert accountability.plan_stage4(run, [S.INTERVIEW_ID], fake_settings(), cache)["calls"] == 0
    run, again = stage4(run, cache)
    assert again.calls == []
    doc = load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert doc["status"] == "CACHED" and doc["cache_hit"] is True and doc["accountability_episode_count"] == 4
    assert run["last_accountability"]["usage"]["api_calls"] == 0
    # Stage 3 relancée depuis son propre cache : aucun appel, et la clé de l'étape 4 ne change pas
    t3 = FakeTransport(S.stage3_responders())
    run = analyze_run(run, settings=fake_settings(), transport=t3, cache=cache)
    assert t3.calls == []
    run, third = stage4(run, cache)
    assert third.calls == []


def test_stage4_version_or_prompt_change_invalidates_only_stage4(tmp_path, monkeypatch):
    run, cache, _ = stage3(tmp_path)
    run, _ = stage4(run, cache)
    monkeypatch.setattr(accountability, "SPEC", dataclasses.replace(accountability.SPEC, version="1.1"))
    assert accountability.plan_stage4(run, [S.INTERVIEW_ID], fake_settings(), cache)["calls"] == 1
    run, after_version = stage4(run, cache)
    assert len(after_version.calls) == 1
    assert load(run, config.ACCOUNTABILITY_MANIFEST_FILENAME)["agent_version"] == "1.1"
    changed_prompt = dataclasses.replace(accountability.SPEC, user_template=accountability.SPEC.user_template + " ")
    monkeypatch.setattr(accountability, "SPEC", changed_prompt)
    run, after_prompt = stage4(run, cache)
    assert len(after_prompt.calls) == 1
    # l'étape 3 reste entièrement en cache
    t3 = FakeTransport(S.stage3_responders())
    analyze_run(run, settings=fake_settings(), transport=t3, cache=cache)
    assert t3.calls == []
    assert len(list((tmp_path / "cache" / "accountability_episode_builder").glob("*.json"))) == 3


def test_force_ignores_the_cache(tmp_path):
    run, cache, _ = stage3(tmp_path)
    run, _ = stage4(run, cache)
    run, forced = stage4(run, cache, force=True)
    assert len(forced.calls) == 1


def test_changed_stage3_output_changes_the_stage4_request(tmp_path):
    run, cache, _ = stage3(tmp_path)
    run, _ = stage4(run, cache)
    path = out_dir(run) / "interaction_signals.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["signals"] = [s for s in document["signals"] if s["signal_type"] != "metadiscursive_self_evaluation"]
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    run, again = stage4(run, cache)
    assert len(again.calls) == 1
    assert load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)["candidate_count"] == 5


# --- Entrées incomplètes ------------------------------------------------------------------------------

def test_stage3_failed_blocks_stage4_without_any_call(tmp_path):
    responders = S.stage3_responders()
    responders[INTERACTION] = server_error(500)
    run, cache, _ = stage3(tmp_path, responders=responders)
    assert run["files"][0]["analysis"]["agents"]["interaction_signal_reader"]["status"] == "FAILED"
    plan = accountability.plan_stage4(run, [S.INTERVIEW_ID], fake_settings(), cache)
    assert plan["calls"] == 0 and plan["blocked"] == 1
    run, t4 = stage4(run, cache)
    assert t4.calls == []
    summary = run["files"][0]["accountability"]
    assert summary["status"] == "BLOCKED" and summary["analysis_complete"] is False
    assert not (out_dir(run) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()
    manifest = load(run, config.ACCOUNTABILITY_MANIFEST_FILENAME)
    assert manifest["error"]["code"] == "STAGE3_UNAVAILABLE" and "Interaction Signal Reader" in manifest["error"]["message"]
    assert load(run, config.ACCOUNTABILITY_VALIDATION_FILENAME)["available"] is False
    assert "1 bloqué(s)" in run["pipeline"][config.ACCOUNTABILITY_STEP]


def test_stage3_not_run_blocks_stage4(tmp_path):
    run = si.make_ingested_run(tmp_path, S.FILES)
    run, t4 = stage4(run, AnalysisCache(tmp_path / "cache"))
    assert t4.calls == [] and run["files"][0]["accountability"]["status"] == "BLOCKED"
    assert run["files"][0]["accountability"]["stage3_status"] == "NOT_RUN"


def test_stale_episodes_are_removed_when_stage3_later_fails(tmp_path):
    run, cache, _ = stage3(tmp_path)
    run, _ = stage4(run, cache)
    assert (out_dir(run) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()
    (out_dir(run) / "practice_extractor.json").unlink()
    manifest = json.loads((out_dir(run) / "practice_manifest.json").read_text())
    manifest["status"] = "FAILED"
    (out_dir(run) / "practice_manifest.json").write_text(json.dumps(manifest))
    run, t4 = stage4(run, cache)
    assert t4.calls == [] and not (out_dir(run) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()


def simulate_partial(run, agent_files=("practice_manifest.json", "practice_extractor.json")):
    for name in agent_files:
        path = out_dir(run) / name
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(status="PARTIAL", analysis_complete=False)
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_stage3_partial_gives_a_partial_stage4_never_complete(tmp_path):
    run, cache, _ = stage3(tmp_path)
    simulate_partial(run)
    run, t4 = stage4(run, cache)
    assert len(t4.calls) == 1
    doc = load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert doc["status"] == "PARTIAL" and doc["analysis_complete"] is False and doc["stage3_status"] == "PARTIAL"
    validation = load(run, config.ACCOUNTABILITY_VALIDATION_FILENAME)
    assert any(i["code"] == "STAGE3_INCOMPLETE" for i in validation["issues"])
    assert "1 incomplet(s)" in run["pipeline"][config.ACCOUNTABILITY_STEP]
    # même depuis le cache, le résultat reste PARTIAL
    run, again = stage4(run, cache)
    assert again.calls == [] and load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)["status"] == "PARTIAL"


def test_llm_failure_is_isolated_and_leaves_no_episodes(tmp_path):
    run, cache, _ = stage3(tmp_path)
    run, t4 = stage4(run, cache, builder=server_error(503))
    summary = run["files"][0]["accountability"]
    assert summary["status"] == "FAILED" and summary["error"]["code"] == "SERVER_ERROR"
    assert not (out_dir(run) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()
    assert load(run, config.ACCOUNTABILITY_VALIDATION_FILENAME)["available"] is False
    # réponse non conforme au schéma → échec également, jamais d'épisode inventé
    run, _ = stage4(run, cache, builder=lambda p: text_response({"episodes": [{"episode_status": "x"}]}))
    assert run["files"][0]["accountability"]["error"]["code"] == "SCHEMA_VALIDATION"


def test_invalid_model_output_is_kept_but_rejected(tmp_path):
    def inventing(params):
        output = S.scripted_output(params, S.REFERENCE_RULES)
        output["episodes"][0]["evidence"] = [{"turn_id": S.tid(4), "quote": "Je ne veux jamais qu'il écrive."}]
        output["episodes"][1]["practice_ids"].append(f"{S.INTERVIEW_ID}_P099")
        output["episodes"][2]["signal_ids"].append(f"{S.INTERVIEW_ID}_S099")
        return text_response(output)

    run, cache, _ = stage3(tmp_path)
    run, _ = stage4(run, cache, builder=inventing)
    doc = load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert doc["status"] == "SUCCESS_WITH_WARNINGS" and doc["rejected_episode_count"] == 3
    assert doc["accountability_episode_count"] == 1  # seuls les épisodes non rejetés sont comptés
    reasons = {code for e in doc["episodes"] for code in e["review_reasons"]}
    assert {"QUOTE_NOT_FOUND", "UNKNOWN_PRACTICE_ID", "UNKNOWN_SIGNAL_ID"} <= reasons


def test_interview_without_accountability_produces_no_forced_episode(tmp_path):
    run, cache, _ = stage3(tmp_path, files=S.PLAIN_FILES, responders=S.plain_responders())
    run, t4 = stage4(run, cache, builder=S.scripted_builder(S.PLAIN_RULES))
    doc = load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert len(t4.calls) == 1 and doc["candidate_count"] == 1
    assert doc["accountability_episode_count"] == 0
    assert doc["ordinary_practice_count"] == 1 and doc["unmarked_practice_count"] == 4
    assert all(u["episode_status"] == "ordinary_practice" for u in doc["unmarked_practices"])


def test_no_candidate_means_no_call(tmp_path):
    responders = S.plain_responders()
    responders[INTERACTION] = lambda p: text_response({"signals": [], "reading_notes": None})
    run, cache, _ = stage3(tmp_path, files=S.PLAIN_FILES, responders=responders)
    assert accountability.plan_stage4(run, [S.PLAIN_ID], fake_settings(), cache)["calls"] == 0
    run, t4 = stage4(run, cache)
    doc = load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert t4.calls == [] and doc["llm_called"] is False and doc["status"] == "SUCCESS"
    assert doc["episode_count"] == 0 and doc["unmarked_practice_count"] == 5


def test_several_interviews_one_blocked_others_continue(tmp_path):
    files = [*S.FILES, *S.PLAIN_FILES]

    def by_interview(agent_reference, agent_plain):
        return lambda p: (agent_reference(p) if S.INTERVIEW_ID + "_T" in p["messages"][0]["content"]
                          else agent_plain(p))

    ref, plain = S.stage3_responders(), S.plain_responders()
    responders = {PRACTICE: by_interview(ref[PRACTICE], plain[PRACTICE]),
                  INTERACTION: by_interview(ref[INTERACTION], lambda p: server_error(500)),
                  AUDITOR: ref[AUDITOR]}
    run, cache, _ = stage3(tmp_path, files=files, responders=responders)
    run, t4 = stage4(run, cache)
    statuses = {f["accountability"]["interview_id"]: f["accountability"]["status"] for f in run["files"]}
    assert statuses == {S.INTERVIEW_ID: "SUCCESS", S.PLAIN_ID: "BLOCKED"} and len(t4.calls) == 1


def test_stage4_is_refused_without_configuration(tmp_path):
    run, cache, _ = stage3(tmp_path)
    with pytest.raises(LLMError) as info:
        accountability.analyze_run_stage4(run, settings=LLMSettings.from_env({}), transport=FakeTransport({}))
    assert info.value.code == "NOT_CONFIGURED"
    assert not (out_dir(run) / config.ACCOUNTABILITY_MANIFEST_FILENAME).exists()


# --- Hygiène ---------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="git indisponible")
def test_no_stage4_output_or_real_interview_is_versioned():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    stage4_outputs = (config.ACCOUNTABILITY_EPISODES_FILENAME, config.ACCOUNTABILITY_MANIFEST_FILENAME,
                      config.ACCOUNTABILITY_VALIDATION_FILENAME)
    assert not [f for f in tracked if f.endswith(stage4_outputs)]
    assert all(f.endswith(".gitkeep") for f in tracked if f.startswith(("data/", "logs/")))
    assert not [f for f in tracked if f.lower().endswith((".pdf", ".docx")) or f.startswith("data/cache")]
