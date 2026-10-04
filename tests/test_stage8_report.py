"""Étape 8 — brouillon du rapport corpus : à partir des étapes 6 et 7 d'un run, sections factuelles déterministes et
sections analytiques rédigées par un agent simulé ; provenance, citations exactes, reprise. Faux Ollama, aucun réseau."""

import json
import re
from pathlib import Path

from core import config, stage8_report
from core.stage7_theory import INDIVIDUAL
from scripts import trace_local
from tests import synthetic_interviews as si
from tests import synthetic_stage6 as S6
from tests.fake_llm import REPORT_WRITER, agent_error, text_response, use_fake_runtime
from tests.test_stage6_blocks import make_run
from tests.test_stage7_theory import responders as stage7_responders
from tests.test_stage7_theory import stage6_run


def claims_of(params) -> list[dict]:
    body = re.search(r"<claims>\n(.*)\n</claims>", params["messages"][0]["content"], re.S).group(1)
    return [json.loads(line) for line in body.splitlines()]


def writer(params):
    """Un paragraphe par sous-section (deux propositions au plus), un paragraphe sans preuve, une citation inventée."""
    content = params["messages"][0]["content"]
    subsections = re.search(r"Sous-sections attendues \(clés\) : ([^.]+)\.", content).group(1).split(", ")
    claims = claims_of(params)
    paragraphs = []
    for n, sub in enumerate(subsections):
        chosen = claims[n * 2:n * 2 + 2] or claims[:1]
        paragraphs.append({"subsection": sub, "text": f"Paragraphe {sub} : " + " ".join(c["formulation"] for c in chosen),
                           "theory_claim_refs": [c["id"] for c in chosen],
                           "evidence_refs": [e for c in chosen for e in c["episodes"]][:1]})
    paragraphs.append({"subsection": subsections[0], "text": "Sans preuve.", "theory_claim_refs": ["TH999"],
                       "evidence_refs": []})
    paragraphs.append({"subsection": subsections[0], "text": "Il dit « je vérifie toujours tout ce que je rends moi-même ».",
                       "theory_claim_refs": [claims[0]["id"]], "evidence_refs": []})
    return text_response({"paragraphs": paragraphs, "section_notes": None})


def stage7_run(tmp_path, monkeypatch):
    run = stage6_run(tmp_path, monkeypatch)
    use_fake_runtime(monkeypatch, stage7_responders())
    assert trace_local.main(["stage7", run["output_dir"]]) == 0
    return run


def draft(run) -> dict:
    return json.loads(stage8_report.paths(run)[0].read_text(encoding="utf-8"))


def test_absent_stage7_blocks_stage8_without_any_call(tmp_path, monkeypatch):
    run = make_run(tmp_path, S6.IDS_17[:3])
    ollama = use_fake_runtime(monkeypatch, {REPORT_WRITER: writer}, available=False)
    assert trace_local.main(["stage8", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    assert draft(run)["status"] == "BLOCKED" and ollama.calls == []
    assert stage8_report.paths(run)[1].is_file()


def test_complete_draft_from_stage7_with_warnings_keeps_every_id(tmp_path, monkeypatch):
    run = stage7_run(tmp_path, monkeypatch)
    theory = json.loads((Path(run["output_dir"]) / "stage7" / "stage7_theory.json").read_text(encoding="utf-8"))
    assert theory["needs_review"]  # étape 7 avec avertissements : l'étape 8 continue et les propage
    ollama = use_fake_runtime(monkeypatch, {REPORT_WRITER: writer})
    assert trace_local.main(["stage8", run["output_dir"]]) == 0
    doc = draft(run)
    assert doc["status"] == "COMPLETE" and doc["result_status"] == "SUCCESS_WITH_WARNINGS" and doc["validated"] is False
    assert set(theory["needs_review"]) <= set(doc["needs_review"])
    sections = [s for s in doc["analytic_sections"] if s["status"] != "EMPTY"]
    assert len(ollama.calls_for(REPORT_WRITER)) == len(sections) <= 5
    for key in ("status", "corpus_coverage", "sections", "paragraphs", "evidence_refs", "interview_refs",
                "theory_claim_refs", "quotes_used", "needs_review", "warnings", "limitations", "generation_manifest"):
        assert key in doc
    accepted = [p for p in doc["paragraphs"] if p["status"] == "accepted"]
    assert [p["paragraph_id"] for p in accepted] == [f"P8-{n:03d}" for n in range(1, len(accepted) + 1)]
    known = {c["theory_claim_id"]: c for c in theory["theory_claims"]}
    for p in accepted:  # provenance conservée jusqu'au JSON final
        assert p["theory_claim_refs"] and set(p["theory_claim_refs"]) <= set(known)
        assert set(p["interview_refs"]) == {i for t in p["theory_claim_refs"] for i in known[t]["supporting_interview_ids"]}
        assert set(p["supporting_cross_claim_ids"]) == {x for t in p["theory_claim_refs"]
                                                       for x in known[t]["supporting_cross_claim_ids"]}
    assert any(p["status"] == "rejected_no_evidence" for p in doc["paragraphs"])  # sans preuve : jamais dans le texte
    markdown = stage8_report.paths(run)[1].read_text(encoding="utf-8")
    for heading in ("BROUILLON", "## 2. Résumé exécutif", "## 3. Corpus et couverture", "## 4. Méthodologie TRACE",
                    "## 5. Résultats principaux", "## 6. Variations et tensions", "## 7. Discussion analytique",
                    "## 8. Limites", "## 9. Conclusion", "## 10. Annexe de traçabilité"):
        assert heading in markdown
    assert "Sans preuve." not in markdown and "je vérifie toujours tout ce que je rends" not in markdown
    assert stage8_report.REMOVED_QUOTE in markdown


def test_failed_section_alone_is_replayed_then_everything_comes_from_the_cache(tmp_path, monkeypatch):
    run = stage7_run(tmp_path, monkeypatch)
    failing = lambda p: agent_error() if "Discussion" in p["messages"][0]["content"] else writer(p)  # noqa: E731
    use_fake_runtime(monkeypatch, {REPORT_WRITER: failing})
    assert trace_local.main(["stage8", run["output_dir"]]) == trace_local.EXIT_CODES["FAILED"]
    doc = draft(run)
    statuses = {s["section_id"]: s["status"] for s in doc["analytic_sections"]}
    assert statuses["discussion"] == "FAILED" and doc["status"] == "FAILED"
    assert all(s != "FAILED" for k, s in statuses.items() if k != "discussion")

    ollama = use_fake_runtime(monkeypatch, {REPORT_WRITER: writer})
    assert trace_local.main(["stage8", run["output_dir"]]) == 0
    calls = ollama.calls_for(REPORT_WRITER)
    assert len(calls) == 1 and "Discussion" in calls[0]["params"]["messages"][0]["content"]
    complete = draft(run)
    assert complete["status"] == "COMPLETE"

    ollama = use_fake_runtime(monkeypatch, {REPORT_WRITER: writer})  # reprise : rien n'est recalculé
    assert trace_local.main(["stage8", run["output_dir"]]) == 0 and ollama.calls == []
    assert draft(run)["paragraphs"] == complete["paragraphs"]


def test_quotes_evidence_and_individual_hypotheses_are_enforced(tmp_path):
    run = si.make_ingested_run(tmp_path, [("ENT_A.txt", "Enquêteur : Tu utilises ChatGPT ?\nEnquêté : Oui, pour réviser "
                                                         "mes cours le soir.\n".encode("utf-8"))])
    analysis_dir = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME).write_text(json.dumps({"episodes": [{
        "episode_id": "ENT_A_E001", "speaker_warnings": [],
        "evidence": [{"turn_id": "ENT_A_T0002", "quote": "Oui, pour réviser mes cours le soir.",
                      "validation": {"valid": True}}]}]}), encoding="utf-8")
    claims = {"TH001": {"theory_claim_id": "TH001", "scope": "corpus_regularity", "needs_review": False,
                        "supporting_interview_ids": ["ENT_A", "ENT_B"], "supporting_cross_claim_ids": ["B01:XC001"],
                        "representative_episode_ids": ["ENT_A_E001", "ENT_A_E002"],
                        "counterexamples": [{"interview_id": "ENT_C", "description": "cas contraire"}]},
              "TH002": {"theory_claim_id": "TH002", "scope": INDIVIDUAL, "needs_review": True,
                        "supporting_interview_ids": ["ENT_A"], "supporting_cross_claim_ids": [],
                        "representative_episode_ids": [], "counterexamples": []}}
    section = {"section_id": "results", "subsections": ("regularities",), "theory_claim_ids": ["TH001", "TH002"]}
    output = {"paragraphs": [
        {"subsection": "regularities", "text": "Il dit « Oui, pour réviser mes cours le soir. » et « je ne triche "
         "jamais sur mes devoirs ».", "theory_claim_refs": ["TH001"], "evidence_refs": ["EP1", "EP2", "EP9"]},
        {"subsection": "regularities", "text": "Les étudiants vérifient souvent leurs réponses.",
         "theory_claim_refs": ["TH002"], "evidence_refs": []},
        {"subsection": "regularities", "text": "Aucune preuve.", "theory_claim_refs": ["TH999"], "evidence_refs": []}]}
    rows = stage8_report.validate_paragraphs(output, section, claims, {"EP1": "ENT_A_E001", "EP2": "ENT_A_E002"},
                                             stage8_report.Quotes(run))
    first, individual, rejected = rows
    assert [q["quote"] for q in first["quotes_used"]] == ["Oui, pour réviser mes cours le soir."]  # exacte, par TRACE
    assert "« Oui, pour réviser mes cours le soir. »" in first["text"]  # passage verbatim conservé
    assert "je ne triche jamais" not in first["text"] and stage8_report.REMOVED_QUOTE in first["text"]
    assert any("introuvable pour l'épisode ENT_A_E002" in w for w in first["warnings"])  # jamais reconstruite
    assert any("EP9" in w for w in first["warnings"])
    assert first["interview_refs"] == ["ENT_A", "ENT_B"] and first["supporting_cross_claim_ids"] == ["B01:XC001"]
    assert first["counterexamples"] == [{"interview_id": "ENT_C", "description": "cas contraire"}]
    assert individual["scope"] == INDIVIDUAL and individual["needs_review"]
    assert any("hypothèse individuelle" in w for w in individual["warnings"])
    assert rejected["status"] == "rejected_no_evidence"
    draft_doc = {"title": "t", "run_id": "r", "generated_at": "g", "status": "COMPLETE",
                 "analytic_sections": [{"title": "5. Résultats", "status": "SUCCESS", "paragraphs": [
                     {**individual, "paragraph_id": "P8-001"}]}]}
    assert "*(Hypothèse — un seul entretien)* Les étudiants" in stage8_report.to_markdown(draft_doc)
