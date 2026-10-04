"""Étape 6 avec un seul entretien (non applicable) et étape 7 — rapport final DÉTERMINISTE (aucun appel au modèle).
Chemin critique : run synthétique jusqu'à l'étape 5 (agents simulés), rapport Markdown + JSON, CLI et Streamlit."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core import config, final_report
from core import local_pipeline as lp
from scripts import trace_local
from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests.fake_llm import use_fake_runtime

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def stage5_run(tmp_path):
    run, cache = S5.run_to_stage4(tmp_path, S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    run, _ = S5.run_stage5(run, cache, S5.scripted_mapper(S5.REFERENCE_PLAN))
    return run


def turns_of(run) -> dict:
    path = Path(run["files"][0]["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME
    return {t["turn_id"]: t for t in json.loads(path.read_text(encoding="utf-8"))["turns"]}


def test_stage6_on_a_single_interview_run_is_not_applicable_from_the_cli(tmp_path, monkeypatch, capsys):
    run = stage5_run(tmp_path)
    ollama = use_fake_runtime(monkeypatch, {}, available=False)  # Ollama arrêté : aucun appel n'est nécessaire
    assert trace_local.main(["stage6", run["output_dir"]]) == 0
    out = capsys.readouterr().out
    assert "NOT_APPLICABLE_SINGLE_INTERVIEW" in out
    assert "La comparaison inter-entretiens nécessite au moins deux entretiens exploitables." in out
    assert ollama.calls == []


def test_final_report_assembles_validated_outputs_without_any_model_call(tmp_path, monkeypatch):
    run = stage5_run(tmp_path)
    ollama = use_fake_runtime(monkeypatch, {}, available=False)
    result = final_report.generate(run)
    assert ollama.calls == []
    report = result["report"]
    assert result["md_path"].is_file() and result["json_path"].is_file()
    assert json.loads(result["json_path"].read_text(encoding="utf-8")) == report
    assert report["status"] == "COMPLETE" and report["llm_called"] is False and report["interview_count"] == 1
    assert report["stage6"]["status"] == "NOT_APPLICABLE_SINGLE_INTERVIEW"
    assert report["stage6"]["reason"] == "La comparaison inter-entretiens nécessite au moins deux entretiens exploitables."
    [item] = report["interviews"]
    assert item["configuration"]["type"] and item["trajectories"] and item["accountability_episodes"]
    assert item["counts"]["accountability_episodes"] == 4
    # citations : exactes, de l'enquêté·e, jamais un tour de l'enquêteur ni un tour à l'attribution douteuse
    turns = turns_of(run)
    quotes = item["representative_quotes"] + [q for e in item["accountability_episodes"] for q in e["quotes"]]
    assert quotes
    for q in quotes:
        assert q["quote"].removesuffix(" […]") in turns[q["turn_id"]]["text"]
        assert turns[q["turn_id"]]["speaker"] != "enqueteur" and not q["turn_id"].endswith("T0022")
    assert item["needs_review"]  # éléments à revoir signalés
    markdown = result["markdown"]
    for heading in ("### Résumé exécutif", "### Trajectoires et configuration (étape 5)",
                    "### Principaux épisodes d'accountability (étape 4)", "### Pratiques et tensions",
                    "### Citations représentatives (validées)", "### À revoir (`needs_review`)",
                    "## Encadré méthodologique", "### Limites"):
        assert heading in markdown
    assert "NOT_APPLICABLE_SINGLE_INTERVIEW" in markdown and "Un seul entretien" in markdown


def test_missing_stage5_is_declared_never_invented(tmp_path):
    run, _ = S5.run_to_stage4(tmp_path, S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    report = final_report.build(run)
    [item] = report["interviews"]
    assert report["status"] == "PARTIAL" and item["missing"] == ["Étape 5 : NOT_RUN"]
    assert item["trajectories"] == [] and item["configuration"] is None
    assert item["stage5_summary"] == final_report.MISSING
    assert final_report.MISSING in final_report.to_markdown(report)
    assert item["accountability_episodes"]  # l'étape 4, validée, reste rapportée


def test_cli_report_command(tmp_path, capsys):
    run = stage5_run(tmp_path)
    assert trace_local.main(["report", run["output_dir"]]) == 0
    out = capsys.readouterr().out
    assert "aucun appel au modèle" in out and "final_report.md" in out
    assert (Path(run["output_dir"]) / "final_report" / "final_report.json").is_file()


def test_streamlit_button_generates_the_final_report(tmp_path, monkeypatch):
    run = stage5_run(tmp_path)
    use_fake_runtime(monkeypatch, {}, available=False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    at.run()
    assert not at.exception and "Étape 7 — Rapport final" in [h.value for h in at.header]
    at.button(key="final_report_generate").click().run()
    assert not at.exception
    assert "Rapport final généré (aucun appel au modèle)." in [s.value for s in at.success]
    labels = [b.label for b in at.get("download_button")]
    assert "Télécharger le rapport (Markdown)" in labels and "Télécharger le rapport (JSON)" in labels
    assert (Path(run["output_dir"]) / "final_report" / "final_report.md").is_file()


def test_experimental_title_and_episodes_flagged_for_verification(tmp_path, capsys):
    run = stage5_run(tmp_path)
    assert trace_local.main(["report", run["output_dir"], "--verify", "E001, e3 ; E099"]) == 0
    out = capsys.readouterr().out
    assert "Interprétation à vérifier : E001, E003" in out and "inconnu(s) dans ce run (ignorés) : E099" in out
    existing = final_report.read_existing(run)
    report, markdown = existing["report"], existing["markdown"]
    assert markdown.startswith("# TRACE — Rapport final expérimental — résultats assistés par modèle local, à relire "
                               "qualitativement")
    assert report["experimental"] is True and report["interpretations_to_verify"] == ["E001", "E003", "E099"]
    [item] = report["interviews"]
    flagged = {e["episode_id"].rsplit("_", 1)[-1] for e in item["accountability_episodes"] + item["other_episodes"]
               if e["interpretation_to_verify"]}
    assert flagged == {"E001", "E003"}
    lines = [line for line in markdown.splitlines() if "⚠ **interprétation à vérifier**" in line]
    assert len(lines) == 2 and all(("E001" in l) or ("E003" in l) for l in lines)
    # régénération sans --verify (bouton Streamlit, CLI) : la liste enregistrée est reprise
    again = final_report.generate(run)
    assert again["report"]["interpretations_to_verify"] == ["E001", "E003", "E099"]
    assert final_report.generate(run, "")["report"]["interpretations_to_verify"] == []  # liste vidée explicitement
