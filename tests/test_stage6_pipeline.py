"""Étape 6 — orchestration : cache, invalidation ciblée, échecs, isolement des étapes 3 à 5, seuil, déterminisme,
mesures sur 2, 8 et 17 entretiens. LLM simulé, aucun appel réel."""

import hashlib
import json
from pathlib import Path

import pytest

from core import config
from core import cross_interview as X
from core import cross_interview_corpus as corpus
from core.analysis_cache import AnalysisCache
from core.llm_client import LLMError
from tests import synthetic_stage5 as S5
from tests import synthetic_stage6 as S6
from tests.fake_llm import COMPARATOR, FakeTransport, fake_settings, server_error

VOLATILE = ("generated_at", "cache_hit", "llm_called", "status", "validation_warning_count")


def run(tmp_path, uploads, transport=None, **kwargs):
    transport = transport or FakeTransport({COMPARATOR: S6.scripted_comparator(S6.GOOD_PLAN)})
    manifest = X.run_stage6(uploads, settings=kwargs.pop("settings", fake_settings()), transport=transport,
                            cache=kwargs.pop("cache", AnalysisCache(tmp_path / "cache")),
                            base_dir=kwargs.pop("base_dir", tmp_path / "out"), **kwargs)
    return manifest, transport


def comparison(manifest) -> dict:
    return json.loads(X.read_outputs(manifest["corpus_dir"])["comparison"])


# --- Cache et invalidation ciblée --------------------------------------------------------------------------------

def test_same_corpus_and_version_gives_zero_call(tmp_path):
    first, transport = run(tmp_path, S6.uploads(S6.IDS_8))
    assert first["status"] == "SUCCESS" and first["api_calls"] == 1 and len(transport.calls) == 1
    again, transport2 = run(tmp_path, list(reversed(S6.uploads(S6.IDS_8))))  # même corpus, autre ordre d'import
    assert again["status"] == "CACHED" and again["api_calls"] == 0 and transport2.calls == []
    assert again["corpus_id"] == first["corpus_id"] and again["cache_key"] == first["cache_key"]
    strip = lambda d: {k: v for k, v in d.items() if k not in VOLATILE}  # noqa: E731
    assert strip(comparison(again)) == strip(comparison(first))
    plan = X.plan_stage6(X.prepare_stage6(S6.uploads(S6.IDS_8)), fake_settings(), AnalysisCache(tmp_path / "cache"))
    assert (plan["calls"], plan["cached"]) == (0, True)


def test_adding_removing_or_modifying_an_interview_invalidates_only_stage6(tmp_path):
    base, _ = run(tmp_path, S6.uploads(S6.IDS_4))
    added, t_added = run(tmp_path, S6.uploads(S6.IDS_4 + ["ENT_E"]))
    removed, t_removed = run(tmp_path, S6.uploads(S6.IDS_4[:3]))
    assert len(t_added.calls) == len(t_removed.calls) == 1
    assert len({base["cache_key"], added["cache_key"], removed["cache_key"]}) == 3
    # ENT_A modifié (même structure, autre description) : nouvelle requête, nouvel appel
    triplet = dict(S6.stage5_files("ENT_A"))
    name = f"ENT_A_{config.STUDENT_TRAJECTORY_FILENAME}"
    document = json.loads(triplet[name])
    document["trajectory_claims"][0]["description"] += " (formulation révisée)"
    triplet[name] = json.dumps(document, ensure_ascii=False).encode()
    modified, t_modified = run(tmp_path, list(triplet.items()) + S6.uploads(S6.IDS_4[1:]))
    assert len(t_modified.calls) == 1 and modified["cache_key"] != base["cache_key"]
    # le corpus d'origine reste en cache
    again, t_again = run(tmp_path, S6.uploads(S6.IDS_4))
    assert again["status"] == "CACHED" and t_again.calls == []


def test_force_ignores_the_cache(tmp_path):
    run(tmp_path, S6.uploads(S6.IDS_2))
    forced, transport = run(tmp_path, S6.uploads(S6.IDS_2), force=True)
    assert forced["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS") and forced["cache_hit"] is False
    assert len(transport.calls) == 1


def test_stage3_to_5_outputs_are_never_called_nor_modified(tmp_path):
    run5, cache5 = S5.case_to_stage4(tmp_path / "run", S5.EXCEPTION)
    run5, _ = S5.run_stage5(run5, cache5, S5.EXCEPTION.mapper())
    root = Path(run5["output_dir"])
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    uploads = corpus.run_stage5_uploads(run5) + S6.uploads(S6.IDS_4)
    manifest, transport = run(tmp_path, uploads)
    assert [c["agent"] for c in transport.calls] == [COMPARATOR]  # aucun agent des étapes 3, 4 ou 5
    assert manifest["upstream_stage_calls"] == 0 and manifest["corpus_n_usable"] == 5
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert after == before
    assert not str(manifest["corpus_dir"]).startswith(str(root))


def test_corpus_directory_keeps_byte_identical_copies_and_the_import_report(tmp_path):
    manifest, _ = run(tmp_path, S6.uploads(S6.IDS_4) + S6.uploads(["ENT_X_INVALIDE"]))
    corpus_dir = Path(manifest["corpus_dir"])
    for iid in S6.IDS_4:
        for name, data in S6.stage5_files(iid):
            assert (corpus_dir / "stage5" / iid / name.removeprefix(f"{iid}_")).read_bytes() == data
    assert not (corpus_dir / "stage5" / "ENT_X_INVALIDE").exists()
    report = X.read_outputs(corpus_dir)["corpus"]
    assert (report["n_total"], report["n_usable"]) == (5, 4)
    assert [r["status"] for r in report["interviews"]] == ["exploitable"] * 4 + ["exclu"]
    assert manifest["input_hashes"]["ENT_A"][config.STUDENT_TRAJECTORY_FILENAME] == hashlib.sha256(
        dict(S6.stage5_files("ENT_A"))[f"ENT_A_{config.STUDENT_TRAJECTORY_FILENAME}"]).hexdigest()


# --- Échecs ---------------------------------------------------------------------------------------------------------

def test_failed_call_writes_no_comparison_and_is_not_cached(tmp_path):
    failing = FakeTransport({COMPARATOR: server_error(500)})
    manifest, _ = run(tmp_path, S6.uploads(S6.IDS_4), transport=failing, settings=fake_settings(max_retries=0))
    assert manifest["status"] == "FAILED" and manifest["analysis_complete"] is False
    outputs = X.read_outputs(manifest["corpus_dir"])
    assert outputs["comparison"] is None and json.loads(outputs["validation"])["available"] is False
    retry, transport = run(tmp_path, S6.uploads(S6.IDS_4))
    assert retry["status"] == "SUCCESS" and len(transport.calls) == 1


def test_a_previous_comparison_is_removed_after_a_failure(tmp_path):
    ok, _ = run(tmp_path, S6.uploads(S6.IDS_4))
    assert X.read_outputs(ok["corpus_dir"])["comparison"]
    failed, _ = run(tmp_path, S6.uploads(S6.IDS_4), transport=FakeTransport({COMPARATOR: server_error(500)}),
                    settings=fake_settings(max_retries=0), force=True)
    assert failed["status"] == "FAILED" and X.read_outputs(failed["corpus_dir"])["comparison"] is None


def test_missing_key_is_refused_before_any_call_unless_blocked(tmp_path):
    with pytest.raises(LLMError) as error:
        run(tmp_path, S6.uploads(S6.IDS_2), settings=fake_settings(api_key=None))
    assert error.value.code == "NOT_CONFIGURED"
    blocked, transport = run(tmp_path, S6.uploads(["ENT_A"]), settings=fake_settings(api_key=None))
    assert blocked["status"] == "BLOCKED" and transport.calls == []


def test_over_threshold_is_flagged_and_still_one_call(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "SINGLE_CALL_MAX_INPUT_TOKENS", 1000)
    manifest, transport = run(tmp_path, S6.uploads(S6.IDS_4))
    assert len(transport.calls) == 1 and manifest["over_single_call_threshold"] is True
    assert manifest["map_reduce_used"] is False and manifest["planned_calls"] == 1
    validation = json.loads(X.read_outputs(manifest["corpus_dir"])["validation"])
    assert "PAYLOAD_OVER_THRESHOLD" in {i["code"] for i in validation["issues"]}
    assert comparison(manifest)["needs_review"] is True


# --- Mesures 2 / 8 / 17 entretiens, déterminisme -------------------------------------------------------------------

@pytest.mark.parametrize("ids, max_tokens", [(S6.IDS_2, 2_500), (S6.IDS_8, 6_500), (S6.IDS_17, 11_000)])
def test_one_call_per_corpus_and_payload_size(tmp_path, ids, max_tokens):
    manifest, transport = run(tmp_path, S6.uploads(ids))
    assert len(transport.calls) == 1 and manifest["api_calls"] == 1
    assert manifest["estimated_input_tokens"] < max_tokens < X.SINGLE_CALL_MAX_INPUT_TOKENS
    assert manifest["payload_chars"] < 0.2 * manifest["raw_stage5_chars"]
    message = transport.calls[0]["params"]["messages"][0]["content"]
    sent = S6.sent_material(transport.calls[0]["params"])
    assert [e["interview_id"] for e in sent["interviews"]] == ids  # tous les interview_id conservés
    assert len(message) < 40_000 or len(ids) == 17


def test_seventeen_interviews_fixture(tmp_path):
    manifest, transport = run(tmp_path, S6.uploads(S6.IDS_17))
    document = comparison(manifest)
    assert len(transport.calls) == 1 and manifest["map_reduce_used"] is False
    assert document["interview_ids"] == S6.IDS_17 and (document["corpus_n_total"], document["corpus_n_usable"]) == (
        17, 17)
    assert sorted(n["interview_id"] for n in document["negative_cases"]) == ["ENT_D", "ENT_H", "ENT_L"]
    assert document["configuration_distribution"]["no_clear_pattern"]["interview_ids"] == ["ENT_Q"]
    criteria = {p["criterion_label"]: p for p in document["student_role_criterion_patterns"]}
    assert set(criteria["effort"]["positions"]["not_observed"]) >= {"ENT_B", "ENT_D", "ENT_Q"}
    assert any(c["needs_review"] for c in document["cross_case_claims"])
    assert document["validation_error_count"] == 0


def test_two_independent_runs_are_identical(tmp_path):
    first, t1 = run(tmp_path / "one", S6.uploads(S6.IDS_17))
    second, t2 = run(tmp_path / "two", list(reversed(S6.uploads(S6.IDS_17))))
    assert t1.calls[0]["params"]["messages"] == t2.calls[0]["params"]["messages"]
    assert first["payload_sha256"] == second["payload_sha256"] and first["cache_key"] == second["cache_key"]
    strip = lambda d: {k: v for k, v in d.items() if k not in VOLATILE}  # noqa: E731
    assert strip(comparison(first)) == strip(comparison(second))


def test_manifest_records_corpus_counts_and_cost(tmp_path):
    manifest, _ = run(tmp_path, S6.uploads(S6.IDS_4) + S6.uploads(["ENT_X_INVALIDE"]))
    on_disk = json.loads(X.read_outputs(manifest["corpus_dir"])["manifest"])
    for key in ("corpus_n_total", "corpus_n_usable", "interview_ids", "excluded_interviews", "api_calls",
                "estimated_input_tokens", "payload_chars", "raw_stage5_chars", "single_call_threshold_tokens",
                "map_reduce_used", "upstream_stage_calls", "negative_case_count", "cache_key"):
        assert key in on_disk, key
    assert (on_disk["corpus_n_total"], on_disk["corpus_n_usable"]) == (5, 4)
    assert on_disk["agent"] == "cross_interview_comparator" and on_disk["agent_version"] == "1.0"
