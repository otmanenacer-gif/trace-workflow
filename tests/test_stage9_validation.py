"""Étape 9 — validation du brouillon (étape 8) : contrôles déterministes, vérification sémantique simulée, corrections
sans boucle, rapport validé. Faux Ollama, aucun réseau, aucun recalcul des étapes 1 à 8."""

import json
import re
from pathlib import Path

from core import config, stage9_validation as s9
from core.stage7_theory import INDIVIDUAL
from scripts import trace_local
from tests import synthetic_interviews as si
from tests import synthetic_stage6 as S6
from tests.fake_llm import REPORT_VALIDATOR, REPORT_WRITER, text_response, use_fake_runtime
from tests.test_stage6_blocks import make_run
from tests.test_stage8_report import stage7_run, writer


def paragraphs_of(params) -> list[dict]:
    body = re.search(r"<paragraphs>\n(.*)\n</paragraphs>", params["messages"][0]["content"], re.S).group(1)
    return [json.loads(line) for line in body.splitlines()]


def validator(params):
    """P8-001 : contradiction (FAIL, exclu) ; P8-002 : généralisation excessive (FAIL, corrigé) ; P8-003 : WARN."""
    verdicts = []
    for p in paragraphs_of(params):
        pid = p["paragraph_id"]
        verdict, issue, action = {"P8-001": ("FAIL", "contradiction", "exclude"),
                                  "P8-002": ("FAIL", "overgeneralization", "add_caution"),
                                  "P8-003": ("WARN", "overinterpretation", "add_caution")}.get(pid, ("PASS", "none", "keep"))
        verdicts.append({"paragraph_id": pid, "verdict": verdict, "issue_type": issue,
                         "explanation": f"verdict simulé {verdict}", "suggested_action": action})
    return text_response({"verdicts": verdicts})


def stage8_run(tmp_path, monkeypatch):
    run = stage7_run(tmp_path, monkeypatch)
    use_fake_runtime(monkeypatch, {REPORT_WRITER: writer})
    assert trace_local.main(["stage8", run["output_dir"]]) == 0
    return run


def load(run, key) -> dict:
    return json.loads(s9.paths(run)[key].read_text(encoding="utf-8"))


def test_absent_stage8_blocks_stage9(tmp_path, monkeypatch):
    run = make_run(tmp_path, S6.IDS_17[:3])
    ollama = use_fake_runtime(monkeypatch, {REPORT_VALIDATOR: validator}, available=False)
    assert trace_local.main(["stage9", run["output_dir"]]) == trace_local.EXIT_CODES["BLOCKED"]
    validation = load(run, "validation")
    assert validation["validation_outcome"] == "BLOCKED" and "Étape 8 absente" in validation["reason"]
    assert ollama.calls == [] and not s9.paths(run)["md"].exists()


def test_validation_of_a_synthetic_draft_few_fails_do_not_block(tmp_path, monkeypatch):
    run = stage8_run(tmp_path, monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {REPORT_VALIDATOR: validator})
    assert trace_local.main(["stage9", run["output_dir"]]) == 0
    validation, report = load(run, "validation"), load(run, "json")
    assert validation["status"] == "COMPLETE" and validation["validation_outcome"] == "SUCCESS_WITH_WARNINGS"
    groups = validation["semantic_groups"]
    assert len(ollama.calls_for(REPORT_VALIDATOR)) == len(groups) and all(len(g["paragraph_ids"]) <= 8 for g in groups)
    for key in ("status", "result_status", "paragraph_checks", "deterministic_checks", "semantic_checks",
                "passed_paragraph_ids", "warned_paragraph_ids", "failed_paragraph_ids", "removed_quotes",
                "corrected_paragraphs", "excluded_paragraphs", "upstream_warnings", "unresolved_warnings",
                "limitations", "validation_manifest"):
        assert key in validation
    excluded = [e["paragraph_id"] for e in validation["excluded_paragraphs"]]
    assert "P8-001" in excluded and "P8-002" not in excluded
    assert any(c["paragraph_id"] == "P8-002" for c in validation["corrected_paragraphs"])
    assert {"P8-001", "P8-002"} <= set(validation["failed_paragraph_ids"])
    # avertissements en amont propagés (étapes 7 et 8), jamais FAIL à eux seuls
    assert any(w.startswith("étape 7 :") for w in validation["upstream_warnings"])
    assert any(w.startswith("étape 8 :") for w in validation["upstream_warnings"])
    # rapport validé : aucun paragraphe exclu, versions corrigées, validated: true
    assert report["validated"] is True and report["stage9_manifest"].endswith("stage9_validation.json")
    kept = [p for s in report["analytic_sections"] for p in s["paragraphs"]]
    assert kept and "P8-001" not in {p["paragraph_id"] for p in kept}
    corrected = next(p for p in kept if p["paragraph_id"] == "P8-002")
    assert corrected["text"].startswith((s9.CAUTION_PREFIX, s9.HYPOTHESIS_PREFIX)) and corrected["stage9_verdict"] == "FAIL"
    assert all(p["theory_claim_refs"] and p["interview_refs"] for p in kept)
    markdown = s9.paths(run)["md"].read_text(encoding="utf-8")
    assert "validé par l'étape 9" in markdown and "P8-001 ·" not in markdown


def test_model_unavailable_still_validates_deterministically_then_resumes_from_cache(tmp_path, monkeypatch):
    run = stage8_run(tmp_path, monkeypatch)
    use_fake_runtime(monkeypatch, {REPORT_VALIDATOR: validator}, available=False)
    assert trace_local.main(["stage9", run["output_dir"]]) == 0
    validation = load(run, "validation")
    assert validation["validation_outcome"] == "SUCCESS_WITH_WARNINGS"
    assert validation["validation_manifest"]["semantic"] == "indisponible"
    assert all(g["status"] == "SKIPPED_MODEL_UNAVAILABLE" for g in validation["semantic_groups"])
    assert load(run, "json")["validated"] is True

    ollama = use_fake_runtime(monkeypatch, {REPORT_VALIDATOR: validator})  # modèle revenu : vérification sémantique
    assert trace_local.main(["stage9", run["output_dir"]]) == 0
    assert ollama.calls_for(REPORT_VALIDATOR)
    first = load(run, "validation")
    ollama = use_fake_runtime(monkeypatch, {REPORT_VALIDATOR: validator})  # reprise : tout vient du cache
    assert trace_local.main(["stage9", run["output_dir"]]) == 0 and ollama.calls == []
    assert load(run, "validation")["paragraph_checks"] == first["paragraph_checks"]


def test_deterministic_checks_ids_quotes_generalization_provenance(tmp_path):
    run = si.make_ingested_run(tmp_path, [("ENT_A.txt", "Enquêteur : Tu utilises ChatGPT pour tes devoirs ?\nEnquêté : "
                                                         "Oui, pour réviser mes cours le soir.\n".encode("utf-8"))])
    analysis_dir = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME).write_text(json.dumps({"episodes": [{
        "episode_id": "ENT_A_E001", "speaker_warnings": [],
        "evidence": [{"turn_id": "ENT_A_T0002", "quote": "Oui, pour réviser mes cours le soir.",
                      "validation": {"valid": True}}]}]}), encoding="utf-8")
    claims = {"TH001": {"scope": "corpus_regularity", "needs_review": False, "supporting_interview_ids": ["ENT_A", "ENT_B"],
                        "supporting_cross_claim_ids": ["B01:XC001"], "representative_episode_ids": ["ENT_A_E001"],
                        "counterexamples": [{"interview_id": "ENT_C", "description": "un cas contraire"}],
                        "formulation": "f1"},
              "TH002": {"scope": INDIVIDUAL, "needs_review": True, "supporting_interview_ids": ["ENT_A"],
                        "supporting_cross_claim_ids": [], "representative_episode_ids": [], "counterexamples": [],
                        "formulation": "f2"}}
    known = {"interviews": {"ENT_A", "ENT_B"}, "cross_claims": {"B01:XC001"}, "episodes": {"ENT_A_E001"}}
    quotes = s9.QuoteChecker(run)
    good = {"episode_id": "ENT_A_E001", "interview_id": "ENT_A", "turn_id": "ENT_A_T0002",
            "quote": "Oui, pour réviser mes cours le soir."}
    base = {"paragraph_id": "P8-001", "theory_claim_refs": ["TH001", "TH999"], "interview_refs": ["ENT_A", "ENT_Z"],
            "supporting_cross_claim_ids": ["B01:XC001", "B09:XC404"], "episode_refs": ["ENT_A_E001"],
            "warnings": [], "needs_review": False, "scope": "corpus",
            "text": "Dans plusieurs entretiens, l'outil sert à réviser les cours à la maison le soir venu.",
            "quotes_used": [good, {**good, "quote": "Oui, pour réviser mes cours le matin."},
                            {**good, "turn_id": "ENT_A_T0001", "quote": "Tu utilises ChatGPT pour tes devoirs ?"}]}
    check = s9.check_paragraph(base, claims, known, quotes, 3)
    messages = " | ".join(i["message"] for i in check["issues"])
    assert "TH999" in messages and "ENT_Z" in messages and "B09:XC404" in messages  # identifiants inventés détectés
    paragraph = check["paragraph"]
    assert paragraph["theory_claim_refs"] == ["TH001"] and paragraph["interview_refs"] == ["ENT_A", "ENT_B"]
    assert [q["quote"] for q in paragraph["quotes_used"]] == [good["quote"]]  # citations invalides retirées
    reasons = [r["reason"] for r in check["removed_quotes"]]
    assert any("transcription" in r for r in reasons) and any("locuteur non autorisé" in r for r in reasons)
    assert "Contre-exemple connu : un cas contraire (ENT_C)." in paragraph["text"]  # contre-exemple connu ajouté
    assert check["verdict"] == "WARN" and check["action"] == "corrected"

    embedded = {**base, "theory_claim_refs": ["TH001"], "quotes_used": [{**good, "quote": "Je triche quand je suis pressé."}],
                "text": "Il le dit sans détour : « Je triche quand je suis pressé. » ce qui montre une limite."}
    check = s9.check_paragraph(embedded, claims, known, quotes, 3)
    assert check["verdict"] == "FAIL" and "Je triche" not in check["paragraph"]["text"]  # FAIL corrigé, pas reconstruit

    general = {**base, "theory_claim_refs": ["TH002"], "quotes_used": [], "episode_refs": [],
               "text": "Les étudiants vérifient souvent leurs réponses avant de rendre un devoir noté."}
    check = s9.check_paragraph(general, claims, known, quotes, 3)
    assert check["verdict"] == "FAIL" and check["action"] == "corrected"
    assert check["paragraph"]["text"].startswith(s9.HYPOTHESIS_PREFIX)  # hypothèse individuelle explicitée

    orphan = {**base, "theory_claim_refs": ["TH777"], "quotes_used": []}
    check = s9.check_paragraph(orphan, claims, known, quotes, 3)
    assert check["verdict"] == "FAIL" and check["action"] == "excluded" and check["paragraph"] is None  # sans provenance
