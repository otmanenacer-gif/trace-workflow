"""Étape 7 — théorisation transversale : à partir de l'étape 6 d'un run (blocs + fusion), blocs thématiques puis
synthèse. Faux Ollama (analyste de blocs et synthèse simulés), aucun recalcul des étapes 1 à 6."""

import json
import re
from pathlib import Path

from core import stage6_blocks, stage7_theory
from scripts import trace_local
from tests import synthetic_stage6 as S6
from tests.fake_llm import (COMPARATOR, THEORY_BLOCK, THEORY_SYNTHESIS, agent_error, text_response,
                            use_fake_runtime)
from tests.test_stage6_blocks import make_run

TYPE_OF = {"recurring": "recurring_mechanism", "divergent": "configuration_variation", "negative": "negative_case",
           "exception": "negative_case", "unresolved": "tension", "temporal": "usage_logic", "minority": "usage_logic",
           "ordinary": "analytic_category", "contextual": "configuration_variation"}


def items_of(params) -> list[dict]:
    body = re.search(r"<items>\n(.*)\n</items>", params["messages"][0]["content"], re.S).group(1)
    return [json.loads(line) for line in body.splitlines()]


def propositions_of(params) -> list[dict]:
    body = re.search(r"<propositions>\n(.*)\n</propositions>", params["messages"][0]["content"], re.S).group(1)
    return [json.loads(line) for line in body.splitlines()]


def block_analyst(params):
    """Une proposition par élément, plus une proposition sans aucun élément identifiable (à rejeter)."""
    props = [{"label": f"catégorie {i['id']}", "formulation": f"Dans {i['n_interviews']} entretien(s), {i['description']}",
              "proposition_type": next((v for k, v in TYPE_OF.items() if (i["type"] or "").startswith(k)),
                                       "good_work_criterion" if i["id"].startswith("R") else "analytic_category"),
              "supporting_item_ids": [i["id"]], "counterexample_item_ids": [], "confidence": "medium",
              "needs_review": False, "limits": None} for i in items_of(params)]
    props.append({"label": "inventée", "proposition_type": "theoretical_proposition", "formulation": "Sans preuve.",
                  "supporting_item_ids": ["X99"], "counterexample_item_ids": [], "confidence": "low",
                  "needs_review": False, "limits": None})
    return text_response({"propositions": props, "block_notes": None})


def synthesizer(params):
    """Fusionne P01 et P02, reprend les autres une à une SAUF la dernière (jamais perdue), une relation invalide."""
    ids = [p["id"] for p in propositions_of(params)]
    claims = [{"category": "fusion", "proposition_type": "recurring_mechanism", "formulation": "P01 et P02 fusionnées.",
               "merged_from": ids[:2], "level": "structuring", "confidence": "medium", "needs_review": False,
               "limits": None}]
    claims += [{"category": f"c{pid}", "proposition_type": "analytic_category", "formulation": pid, "merged_from": [pid],
                "level": "secondary", "confidence": "low", "needs_review": False, "limits": None} for pid in ids[2:-1]]
    return text_response({"theory_claims": claims, "synthesis_summary": "Synthèse.", "synthesis_notes": None,
                          "relations": [{"source": 1, "target": 2, "relation_type": "reinforces", "description": "r"},
                                        {"source": 1, "target": 99, "relation_type": "conditions", "description": "x"}]})


def stage6_run(tmp_path, monkeypatch, ids=None, valid=None):
    ids = ids or S6.IDS_17[:8]
    run = make_run(tmp_path, ids, valid=valid)
    use_fake_runtime(monkeypatch, {COMPARATOR: S6.scripted_comparator(S6.GOOD_PLAN)})
    assert trace_local.main(["stage6", run["output_dir"]]) in (0, trace_local.EXIT_CODES["BLOCKED"])
    return run


def theory(run) -> dict:
    return json.loads(stage7_theory.out_path(run).read_text(encoding="utf-8"))


def responders(block=block_analyst):
    return {THEORY_BLOCK: block, THEORY_SYNTHESIS: synthesizer}


def test_absent_or_incomplete_stage6_blocks_stage7_without_any_call(tmp_path, monkeypatch):
    run = make_run(tmp_path / "absent", S6.IDS_17[:3])
    ollama = use_fake_runtime(monkeypatch, responders(), available=False)
    assert trace_local.main(["stage7", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    assert theory(run)["status"] == "BLOCKED" and "Étape 6 absente" in theory(run)["reason"]

    run = stage6_run(tmp_path / "failed", monkeypatch, S6.IDS_17[:3])
    path = stage6_blocks.out_path(run)
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "status": "FAILED",
                                "reason": "bloc(s) en échec : B01"}), encoding="utf-8")
    ollama = use_fake_runtime(monkeypatch, responders(), available=False)
    assert trace_local.main(["stage7", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    assert "Étape 6 non terminée (FAILED" in theory(run)["reason"] and ollama.calls == []

    run = stage6_run(tmp_path / "single", monkeypatch, S6.IDS_17[:2], valid=[S6.IDS_17[0]])
    assert trace_local.main(["stage7", run["output_dir"]]) == 0
    assert theory(run)["status"] == "NOT_APPLICABLE_SINGLE_INTERVIEW"


def test_multi_interview_corpus_complete_with_traceable_evidence(tmp_path, monkeypatch):
    run = stage6_run(tmp_path, monkeypatch)
    ollama = use_fake_runtime(monkeypatch, responders())
    assert trace_local.main(["stage7", run["output_dir"]]) == 0
    doc = theory(run)
    assert doc["status"] == "COMPLETE"
    blocks = doc["thematic_blocks"]
    assert len(ollama.calls_for(THEORY_BLOCK)) == len(blocks) <= 6 and len(ollama.calls_for(THEORY_SYNTHESIS)) == 1
    coverage = doc["corpus_coverage"]
    assert coverage["expected_interviews"] == 8 and coverage["n_included"] == 8
    # sans preuve identifiable : rejetée, jamais une proposition théorique
    assert doc["rejected_propositions"] and all(r["reason"].startswith("NO_EVIDENCE") for r in doc["rejected_propositions"])
    for claim in doc["theory_claims"]:
        assert claim["theory_claim_id"].startswith("TH") and claim["merged_from"]
        assert claim["supporting_interview_ids"]  # jamais une proposition théorique sans preuve identifiable
        assert all(":" in key for key in claim["supporting_cross_claim_ids"])  # <bloc étape 6>:<cross_claim_id>
        assert set(claim["supporting_interview_ids"]) <= set(coverage["included_interview_ids"])
    # la synthèse conserve les preuves : union de celles des propositions fusionnées
    by_id = {p["proposition_id"]: p for p in doc["propositions"]}
    fused = doc["theory_claims"][0]
    assert fused["merged_from"] == ["P01", "P02"]
    assert set(fused["supporting_cross_claim_ids"]) == set(by_id["P01"]["supporting_cross_claim_ids"]) | \
        set(by_id["P02"]["supporting_cross_claim_ids"])
    assert set(fused["supporting_trajectory_claim_ids"]) >= set(by_id["P01"]["supporting_trajectory_claim_ids"])
    last = list(by_id)[-1]  # proposition omise par la synthèse : conservée, à revoir
    kept = [c for c in doc["theory_claims"] if c["merged_from"] == [last]]
    assert kept and kept[0]["origin"] == "not_merged_by_synthesis" and kept[0]["needs_review"]
    assert doc["relations"] == [{"source": "TH001", "target": "TH002", "relation_type": "reinforces", "description": "r"}]
    assert any("99" in issue for issue in doc["needs_review"])
    for key in ("main_categories", "structuring_regularities", "variations", "tensions", "negative_cases",
                "criteria_and_boundaries", "limitations", "evidence_index"):
        assert key in doc
    assert set(doc["evidence_index"]["cross_claims"]) == {k for c in doc["theory_claims"]
                                                         for k in c["supporting_cross_claim_ids"]}


def test_a_proposition_supported_by_one_interview_is_never_a_corpus_regularity():
    items = {"B01:XC001": {"key": "B01:XC001", "kind": "cross_claim", "claim_type": "recurring_boundary",
                           "interview_ids": ["ENT_A"], "trajectory_claim_ids": [], "counterexamples": []},
             "B01:XC002": {"key": "B01:XC002", "kind": "cross_claim", "claim_type": "recurring_boundary",
                           "interview_ids": ["ENT_A", "ENT_B"], "trajectory_claim_ids": [], "counterexamples": []}}
    index = stage7_theory.EvidenceIndex({"files": []}, items)
    output = {"propositions": [
        {"label": "a", "proposition_type": "recurring_mechanism", "formulation": "un seul entretien",
         "supporting_item_ids": ["X01"], "counterexample_item_ids": [], "confidence": "high", "needs_review": False,
         "limits": None},
        {"label": "b", "proposition_type": "recurring_mechanism", "formulation": "deux entretiens",
         "supporting_item_ids": ["X02"], "counterexample_item_ids": [], "confidence": "high", "needs_review": False,
         "limits": None}]}
    block = stage7_theory.validate_block(output, {"X01": "B01:XC001", "X02": "B01:XC002"}, index, "T1")
    single, double = block["propositions"]
    assert single["scope"] == "individual_case_hypothesis" and single["needs_review"]
    assert double["scope"] == "corpus_regularity" and not double["needs_review"]
    merged = stage7_theory.validate_synthesis({"theory_claims": [
        {"category": "a", "proposition_type": "recurring_mechanism", "formulation": "a", "merged_from": ["P01"],
         "level": "structuring", "confidence": "high", "needs_review": False, "limits": None}], "relations": []},
        {"P01": single, "P02": double})
    first = merged["theory_claims"][0]
    assert first["level"] == "hypothesis" and first["scope"] == "individual_case_hypothesis"  # jamais structurante
    assert any("moins de deux entretiens" in i for i in merged["issues"])


def test_resume_from_cache_and_a_failed_block_is_the_only_one_replayed(tmp_path, monkeypatch):
    run = stage6_run(tmp_path, monkeypatch)
    failing = lambda p: agent_error() if "(T1" in p["messages"][0]["content"] else block_analyst(p)  # noqa: E731
    use_fake_runtime(monkeypatch, responders(block=failing))
    assert trace_local.main(["stage7", run["output_dir"]]) == trace_local.EXIT_CODES["FAILED"]
    doc = theory(run)
    statuses = {b["block_id"]: b["status"] for b in doc["thematic_blocks"]}
    assert statuses["T1"] == "FAILED" and all(s != "FAILED" for k, s in statuses.items() if k != "T1")
    assert doc["status"] == "FAILED" and "T1" in doc["reason"]

    ollama = use_fake_runtime(monkeypatch, responders())
    assert trace_local.main(["stage7", run["output_dir"]]) == 0
    assert len(ollama.calls_for(THEORY_BLOCK)) == 1 and "(T1" in ollama.calls_for(THEORY_BLOCK)[0]["params"][
        "messages"][0]["content"]  # seul le bloc en échec est rejoué
    assert len(ollama.calls_for(THEORY_SYNTHESIS)) == 1  # nouvelle synthèse : ses entrées ont changé
    complete = theory(run)
    assert complete["status"] == "COMPLETE"

    ollama = use_fake_runtime(monkeypatch, responders())  # relance : tout vient du cache
    assert trace_local.main(["stage7", run["output_dir"]]) == 0 and ollama.calls == []
    assert theory(run)["theory_claims"] == complete["theory_claims"]
