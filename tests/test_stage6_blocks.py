"""Étape 6 d'un run (lot) par blocs : sélection par le manifeste du lot, N = 0 / 1 / ≥ 2, découpage, fusion,
reprise par le cache, bloc en échec. Faux Ollama (comparateur simulé), aucun recalcul des étapes 1 à 5."""

import json
from pathlib import Path

from core import config, stage6_blocks
from scripts import trace_local
from tests import synthetic_interviews as si
from tests import synthetic_stage6 as S6
from tests.fake_llm import COMPARATOR, agent_error, use_fake_runtime

TEXT = "Enquêteur : Tu utilises ChatGPT ?\nEnquêté : Oui, pour réviser.\n"


def make_run(tmp_path, ids, valid=None, failed=()) -> dict:
    """Run de `ids` dont l'étape 5 est déjà enregistrée (triplets synthétiques), et le manifeste du lot."""
    run = si.make_ingested_run(tmp_path, [(f"{iid}.txt", TEXT.encode("utf-8")) for iid in ids])
    for info in run["files"]:
        iid = info["ingestion"]["interview_id"]
        analysis_dir = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name, data in S6.stage5_files(iid):
            (analysis_dir / name.removeprefix(f"{iid}_")).write_bytes(data)
    valid = list(ids) if valid is None else valid
    interviews = {iid: {"interview_id": iid, "final_status": "WARNINGS" if iid in valid else "FAILED",
                        "stages": {"3": {"status": "COMPLETE" if iid in valid else "FAILED",
                                         "error": None if iid in valid else "réponse non conforme"}}}
                  for iid in ids}
    for iid in failed:
        interviews[iid]["final_status"] = "FAILED"
    path = Path(run["output_dir"]) / "batch" / "batch_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run["run_id"], "stage5_valid": valid, "interviews": interviews}),
                    encoding="utf-8")
    return run


def output(run) -> dict:
    return json.loads(stage6_blocks.out_path(run).read_text(encoding="utf-8"))


def good():
    return S6.scripted_comparator(S6.GOOD_PLAN)


def test_block_plan_is_balanced_and_covers_every_interview_once():
    ids = [f"I{n:02d}" for n in range(17)]
    blocks = stage6_blocks.plan_blocks(ids)
    assert [len(b) for b in blocks] == [6, 6, 5] and [i for b in blocks for i in b] == ids
    assert [len(b) for b in stage6_blocks.plan_blocks(ids[:7])] == [4, 3]
    assert [len(b) for b in stage6_blocks.plan_blocks(ids[:6])] == [6]
    assert [len(b) for b in stage6_blocks.plan_blocks(ids[:2])] == [2]


def test_zero_valid_interview_is_blocked_without_any_call(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path, ["ENT_A", "ENT_B"], valid=[])
    ollama = use_fake_runtime(monkeypatch, {COMPARATOR: good()}, available=False)
    assert trace_local.main(["stage6", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    document = output(run)
    assert document["status"] == "BLOCKED" and document["reason"] == stage6_blocks.BLOCKED_REASON
    assert {e["interview_id"] for e in document["excluded_interviews"]} == {"ENT_A", "ENT_B"}
    assert ollama.calls == []


def test_one_valid_interview_is_not_applicable_without_any_call(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path, ["ENT_A", "ENT_B"], valid=["ENT_A"])
    ollama = use_fake_runtime(monkeypatch, {COMPARATOR: good()}, available=False)
    assert trace_local.main(["stage6", run["output_dir"]]) == 0
    document = output(run)
    assert document["status"] == "NOT_APPLICABLE_SINGLE_INTERVIEW"
    assert document["reason"] == "La comparaison inter-entretiens nécessite au moins deux entretiens exploitables."
    assert document["interview_ids"] == ["ENT_A"] and ollama.calls == []
    assert "NOT_APPLICABLE_SINGLE_INTERVIEW" in capsys.readouterr().out


def test_small_corpus_one_block_structured_output_with_references(tmp_path, monkeypatch):
    run = make_run(tmp_path, ["ENT_A", "ENT_B", "ENT_C", "ENT_D"], valid=["ENT_A", "ENT_B", "ENT_C"])
    ollama = use_fake_runtime(monkeypatch, {COMPARATOR: good()})
    assert trace_local.main(["stage6", run["output_dir"]]) == 0
    document = output(run)
    assert document["status"] == "COMPLETE" and document["selection_source"] == "batch_manifest"
    assert document["interview_ids"] == ["ENT_A", "ENT_B", "ENT_C"] and len(ollama.calls) == 1
    [excluded] = document["excluded_interviews"]
    assert excluded["interview_id"] == "ENT_D" and "FAILED" in excluded["reason"]
    [block] = document["blocks"]
    assert block["status"] == "COMPLETE" and block["comparison"]["interview_ids"] == ["ENT_A", "ENT_B", "ENT_C"]
    for key in ("regularities", "variations", "tensions", "negative_cases_and_exceptions", "trajectory_differences",
                "recurring_configurations", "recurring_student_role_criteria", "claims_needing_review",
                "limitations", "needs_review"):
        assert key in document
    claims = [c for name in stage6_blocks.CATEGORIES for c in document[name]]
    assert claims
    for claim in claims:  # jamais une affirmation reliée à moins de deux entretiens… sauf configuration minoritaire
        assert claim["block_id"] == "B01" and claim["interview_ids"] and claim["trajectory_claim_ids"] + \
            claim["student_role_criterion_ids"]
        assert set(claim["interview_ids"]) <= {"ENT_A", "ENT_B", "ENT_C"}
    assert all(c["n_interviews"] >= 2 for c in document["recurring_configurations"])


def test_large_corpus_is_split_into_blocks_merged_and_resumed_from_the_cache(tmp_path, monkeypatch):
    ids = S6.IDS_17[:8]
    run = make_run(tmp_path, ids)
    ollama = use_fake_runtime(monkeypatch, {COMPARATOR: good()})
    assert trace_local.main(["stage6", run["output_dir"]]) == 0
    document = output(run)
    assert [b["interview_ids"] for b in document["blocks"]] == [ids[:4], ids[4:]]
    assert [b["status"] for b in document["blocks"]] == ["COMPLETE", "COMPLETE"] and len(ollama.calls) == 2
    for call, block in zip(ollama.calls, document["blocks"]):  # chaque appel ne reçoit que les entretiens de son bloc
        sent = {e["interview_id"] for e in S6.sent_material(call["params"])["interviews"]}
        assert sent == set(block["interview_ids"])
    blocks_of_claims = {c["block_id"] for name in stage6_blocks.CATEGORIES for c in document[name]}
    assert blocks_of_claims == {"B01", "B02"}  # fusion : affirmations des deux blocs
    assert len(document["configuration_by_interview"]) == 8  # configurations comptées sur TOUS les entretiens

    ollama = use_fake_runtime(monkeypatch, {COMPARATOR: good()})  # reprise : blocs déjà analysés, aucun appel
    assert trace_local.main(["stage6", run["output_dir"]]) == 0
    assert ollama.calls == [] and all(b["skipped"] for b in output(run)["blocks"])


def test_a_failing_block_keeps_the_validated_blocks_and_only_it_is_replayed(tmp_path, monkeypatch):
    ids = S6.IDS_17[:8]
    run = make_run(tmp_path, ids)
    comparator = good()
    failing = lambda p: agent_error() if any(e["interview_id"] == ids[5] for e in S6.sent_material(p)["interviews"]) \
        else comparator(p)  # noqa: E731
    use_fake_runtime(monkeypatch, {COMPARATOR: failing})
    assert trace_local.main(["stage6", run["output_dir"]]) == trace_local.EXIT_CODES["FAILED"]
    document = output(run)
    assert [b["status"] for b in document["blocks"]] == ["COMPLETE", "FAILED"]
    assert document["status"] == "FAILED" and "B02" in document["reason"]
    assert {c["block_id"] for name in stage6_blocks.CATEGORIES for c in document[name]} == {"B01"}

    ollama = use_fake_runtime(monkeypatch, {COMPARATOR: good()})
    assert trace_local.main(["stage6", run["output_dir"]]) == 0
    assert len(ollama.calls) == 1  # seul le bloc en échec est rejoué
    sent = {e["interview_id"] for e in S6.sent_material(ollama.calls[0]["params"])["interviews"]}
    assert sent == set(ids[4:])
    assert [b["status"] for b in output(run)["blocks"]] == ["COMPLETE", "COMPLETE"]
