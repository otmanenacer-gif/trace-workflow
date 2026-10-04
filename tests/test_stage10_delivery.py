"""Étape 10 — livraison finale : chemin critique seulement. Aucun modèle (faux Ollama indisponible), aucun réseau."""

import json

from core import stage10_delivery as s10
from core import stage9_validation as s9
from scripts import trace_local
from tests import synthetic_stage6 as S6
from tests.fake_llm import REPORT_VALIDATOR, use_fake_runtime
from tests.test_stage6_blocks import make_run
from tests.test_stage9_validation import stage8_run, validator


def load(run, key) -> dict:
    return json.loads(s10.paths(run)[key].read_text(encoding="utf-8"))


def stage9_run(tmp_path, monkeypatch):
    run = stage8_run(tmp_path, monkeypatch)
    use_fake_runtime(monkeypatch, {REPORT_VALIDATOR: validator})
    assert trace_local.main(["stage9", run["output_dir"]]) == 0
    return run


def test_absent_or_blocked_stage9_blocks_delivery_without_any_call(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path, S6.IDS_17[:3])
    ollama = use_fake_runtime(monkeypatch, {}, available=False)
    assert trace_local.main(["stage10", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    manifest = load(run, "manifest")
    assert manifest["status"] == "BLOCKED" and "Étape 9 absente" in manifest["reason"]
    assert not s10.paths(run)["md"].exists() and "TRACE BLOCKED" in capsys.readouterr().out

    assert trace_local.main(["stage9", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]  # étape 8 absente
    assert trace_local.main(["stage10", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    manifest = load(run, "manifest")
    assert manifest["status"] == "BLOCKED" and "non livrable (BLOCKED" in manifest["reason"]
    assert not s10.paths(run)["json"].exists() and ollama.calls == []


def test_delivery_with_warnings_keeps_validated_text_provenance_and_manifest_counts(tmp_path, monkeypatch, capsys):
    run = stage9_run(tmp_path, monkeypatch)
    validated = json.loads(s9.paths(run)["json"].read_text(encoding="utf-8"))
    validation = json.loads(s9.paths(run)["validation"].read_text(encoding="utf-8"))
    ollama = use_fake_runtime(monkeypatch, {}, available=False)  # tout appel au modèle échouerait
    capsys.readouterr()
    assert trace_local.main(["stage10", run["output_dir"]]) == 0
    assert ollama.calls == []
    printed = capsys.readouterr().out
    final, manifest = load(run, "json"), load(run, "manifest")
    markdown = s10.paths(run)["md"].read_text(encoding="utf-8")
    stage6 = json.loads((s10.Path(run["output_dir"]) / "stage6" / "stage6_corpus.json").read_text(encoding="utf-8"))

    # Stage 9 SUCCESS_WITH_WARNINGS -> livraison réussie, avec avertissements
    assert final["status"] == manifest["status"] == "SUCCESS_WITH_WARNINGS"
    for line in ("TRACE COMPLETE", f"Run: {run['run_id']}", f"Interviews included: {len(stage6['interview_ids'])} / 8",
                 "Result: SUCCESS_WITH_WARNINGS", "Final report:", "Manifest:"):
        assert line in printed

    # aucun nouveau claim : exactement les paragraphes validés, textes identiques ; exclus absents
    kept = {p["paragraph_id"]: p for s in validated["analytic_sections"] for p in s["paragraphs"]}
    assert {p["paragraph_id"]: p["text"] for p in final["paragraphs"]} == {pid: p["text"] for pid, p in kept.items()}
    assert "P8-001" not in kept and "P8-001" not in {p["paragraph_id"] for p in final["paragraphs"]}
    assert "[P8-001]" not in markdown and all(p["text"] in markdown for p in final["paragraphs"])
    assert [n for n in range(1, 14)] == [s["number"] for s in final["sections"]]

    # statistiques cohérentes avec les manifestes
    stats = final["statistics"]
    assert stats == manifest["interview_counts"]
    assert stats["planned"] == 8 and stats["included_stage6"] == len(stage6["interview_ids"])
    assert stats["theory_claims"] == len(json.loads((s10.Path(run["output_dir"]) / "stage7" / "stage7_theory.json")
                                                   .read_text(encoding="utf-8"))["theory_claims"])
    assert stats["stage9_pass"] + stats["stage9_warn"] + stats["stage9_fail"] == stats["stage8_paragraphs"]
    assert stats["stage9_excluded"] == len(validation["excluded_paragraphs"]) >= 1
    assert stats["unresolved_warnings"] == len(validation["unresolved_warnings"])
    assert stats["quotes_used"] == sum(len(p["quotes"]) for p in final["paragraphs"])
    # api_calls lus dans les manifestes : le manifeste de lot synthétique ne déclare rien -> non vérifiable
    assert manifest["api_calls"]["total"] == 0 and manifest["api_calls"]["verified"] is False
    assert "stages_3_5" in manifest["api_calls"]["unverifiable"] and manifest["model_calls"] == 0

    # provenance conservée
    for p in final["paragraphs"]:
        assert p["theory_claim_refs"] and p["supporting_interview_ids"] and p["stage9_verdict"] in ("PASS", "WARN", "FAIL")
        for key in ("supporting_cross_claim_ids", "episode_refs", "quotes", "warnings", "corrections_applied"):
            assert key in p
        assert f"| {p['paragraph_id']} |" in markdown
    for key in ("run_id", "status", "generated_at", "pipeline_version", "model", "api_calls", "interview_counts",
                "stages", "stage_statuses", "upstream_files", "output_files", "warnings", "unresolved_warnings",
                "exclusions", "validation_summary", "durations"):
        assert key in manifest

    # relance idempotente : même rapport, aucun appel
    before = (markdown, s10.paths(run)["json"].read_text(encoding="utf-8"))
    assert trace_local.main(["stage10", run["output_dir"]]) == 0 and ollama.calls == []
    assert (s10.paths(run)["md"].read_text(encoding="utf-8"), s10.paths(run)["json"].read_text(encoding="utf-8")) == before
