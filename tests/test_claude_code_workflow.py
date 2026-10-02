"""Étape 3 en workflow Claude Code : paquets de tâche, réponses d'agent, validation, sorties, restauration.

Aucun appel API : en plus des garde-fous de tests/conftest.py (transport réel interdit, réseau refusé, aucune
clé), la construction d'un client d'API (LLMClient) est interdite dans les tests du workflow. Les « agents »
sont simulés comme dans les autres tests : leurs réponses (tests/synthetic_*.py) sont écrites dans
response.json, exactement là où un sous-agent Claude Code les écrirait.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.llm_client as llm_client
from agents.base import render_user_message
from core import accountability, analysis, config
from core import claude_code_workflow as wf
from core.stage3_restore import restore_stage3
from tests import synthetic_interviews as si
from tests.fake_llm import (AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, FakeTransport, agent_of, fake_settings,
                            text_response)

APP = str(config.PROJECT_ROOT / "app.py")
GOOD = {PRACTICE: si.GOOD_PRACTICES, INTERACTION: si.GOOD_SIGNALS}
EMPTY = {PRACTICE: {"practices": [], "extraction_notes": None}, INTERACTION: {"signals": [], "reading_notes": None},
         AUDITOR: {"assessments": [], "audit_notes": None}}


@pytest.fixture
def no_api_client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Client d'API construit pendant le workflow Claude Code : interdit.")
    monkeypatch.setattr(llm_client.LLMClient, "__init__", forbidden)


# --- Outils : jouer les agents comme un sous-agent Claude Code ---------------------------------------------

def packet(task_dir: Path) -> dict:
    task = json.loads((task_dir / wf.TASK_FILENAME).read_text(encoding="utf-8"))
    return {"task": task,
            "system_prompt": wf._resolve(task["system_prompt"]["path"]).read_text(encoding="utf-8"),
            "payload": wf._resolve(task["payload"]["path"]).read_text(encoding="utf-8"),
            "schema": json.loads(wf._resolve(task["schema"]["path"]).read_text(encoding="utf-8"))}


def params_of(task_dir: Path) -> dict:
    """Le paquet, sous la forme des paramètres que recevaient les agents simulés (tests/fake_llm.py)."""
    p = packet(task_dir)
    return {"system": [{"type": "text", "text": p["system_prompt"]}],
            "messages": [{"role": "user", "content": p["payload"]}],
            "output_config": {"format": {"type": "json_schema", "schema": p["schema"]}}}


def answer(status: dict, responders: dict) -> list[str]:
    """Écrit response.json pour chaque tâche en attente ou à corriger. Renvoie les agents joués."""
    played = []
    for task in status["pending"] + status["invalid"]:
        task_dir = wf._resolve(task["task_dir"])
        params = params_of(task_dir)
        agent = agent_of(params)
        handler = responders[agent]
        result = handler(params) if callable(handler) else handler
        text = result.content[0].text if hasattr(result, "content") else json.dumps(result, ensure_ascii=False)
        (task_dir / wf.RESPONSE_FILENAME).write_text(text, encoding="utf-8")
        played.append(agent)
    return played


def drive(run: dict, responders: dict, max_passes: int = 6) -> tuple[dict, list[list[str]]]:
    """Passages successifs jusqu'à la fin de l'étape 3. Renvoie (dernier résultat, agents joués par vague)."""
    waves = []
    for _ in range(max_passes):
        result = wf.run_stage3(run)
        run = result["metadata"]
        if result["status"]["status"] == wf.STAGE3_COMPLETE:
            return result, waves
        assert result["status"]["status"] == wf.STAGE3_AWAITING, result["status"]
        waves.append(sorted(answer(result["status"], responders)))
    raise AssertionError("L'étape 3 ne se termine pas")


def analysis_dir(run: dict, interview_id: str = si.INTERVIEW_ID) -> Path:
    return Path(run["output_dir"]) / config.INTERVIEWS_SUBDIR / interview_id / config.ANALYSIS_SUBDIR


def read(run: dict, filename: str, interview_id: str = si.INTERVIEW_ID) -> dict:
    return json.loads((analysis_dir(run, interview_id) / filename).read_text(encoding="utf-8"))


# --- Configuration ----------------------------------------------------------------------------------------

def test_backend_defaults_to_claude_code_and_api_mode_is_only_explicit():
    assert wf.stage3_backend({}) == wf.BACKEND_CLAUDE_CODE
    assert wf.stage3_backend({"TRACE_STAGE3_BACKEND": "n'importe quoi"}) == wf.BACKEND_CLAUDE_CODE
    assert wf.stage3_backend({"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_MODEL": "m"}) == wf.BACKEND_CLAUDE_CODE
    assert wf.stage3_backend({"TRACE_STAGE3_BACKEND": "anthropic"}) == wf.BACKEND_ANTHROPIC


def test_workflow_settings_need_no_key_and_keep_chunk_sizes():
    settings = wf.workflow_settings({"ANTHROPIC_API_KEY": "sk-ant-x", "TRACE_PRACTICE_CHUNK_TOKENS": "3000"})
    assert settings.api_key is None and settings.model == wf.WORKFLOW_MODEL
    assert settings.request_params() == {"effort": None, "temperature": None}
    assert settings.practice_chunk_tokens == 3000


# --- Entretien court : paquets, réponses, validation, sorties -----------------------------------------------

def test_first_pass_writes_exact_task_packets_and_no_output(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path)
    result = wf.run_stage3(run)
    status, run = result["status"], result["metadata"]
    assert status["status"] == wf.STAGE3_AWAITING and status["api_calls"] == 0
    assert sorted(t["agent"] for t in status["pending"]) == [INTERACTION, PRACTICE]  # aucun tour suspect : pas d'audit
    prepared = analysis.prepare_interview(run["files"][0]["ingestion"])
    for task in status["pending"]:
        p = packet(wf._resolve(task["task_dir"]))
        spec = wf.SPECS_BY_NAME[task["agent"]]
        assert p["task"]["system_prompt"]["path"] == f"prompts/{task['agent']}.md"
        assert p["system_prompt"] == spec.system_prompt  # le prompt du dépôt, source de vérité, inchangé
        assert p["payload"] == render_user_message(prepared.agent_input, prepared.transcript_json)
        assert p["schema"] == spec.output_schema
        assert p["task"]["check_command"].startswith("python scripts/trace_workflow.py check ")
        assert not (wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME).exists()
    agents = run["files"][0]["analysis"]["agents"]
    assert {a["error"]["code"] for a in agents.values()} == {wf.AWAITING_AGENT}
    assert not (analysis_dir(run) / "practice_extractor.json").exists()
    assert "2 tâche(s) en attente" in run["pipeline"][config.PRACTICE_STEP]
    assert wf.read_status(Path(run["output_dir"]))["status"] == wf.STAGE3_AWAITING


def test_answered_tasks_are_validated_and_saved_as_usual_stage3_outputs(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path)
    first = wf.run_stage3(run)
    assert sorted(answer(first["status"], GOOD)) == [INTERACTION, PRACTICE]
    for task in first["status"]["pending"]:
        report = wf.check_task(wf._resolve(task["task_dir"]))
        assert report["ok"] and not report["blocking"] and report["counts"]["invalid_evidence"] == 0

    result = wf.run_stage3(first["metadata"])
    status, run = result["status"], result["metadata"]
    assert status["status"] == wf.STAGE3_COMPLETE and status["answered_count"] == 2 and not status["pending"]
    agents = run["files"][0]["analysis"]["agents"]
    assert {a["status"] for a in agents.values()} == {analysis.STATUS_SUCCESS}
    assert all(a["api_calls"] == 0 and a["billed_this_run"] is False for a in agents.values())
    assert run["last_analysis"]["usage"]["api_calls"] == 0 and run["last_analysis"]["model"] == wf.WORKFLOW_MODEL
    assert run["pipeline"][config.PRACTICE_STEP] == "terminé (1/1 entretien(s))"
    practices = read(run, "practice_extractor.json")
    assert practices["model"] == wf.WORKFLOW_MODEL and practices["item_count"] == 6
    manifest = read(run, "practice_manifest.json")
    assert manifest["request_id"] == next(t["task_id"] for t in first["status"]["pending"] if t["agent"] == PRACTICE)
    assert read(run, "interaction_signals.json")["item_count"] == 10
    assert read(run, config.EVIDENCE_VALIDATION_FILENAME)["total_invalid_evidence"] == 0

    files = {name: (analysis_dir(run) / name).read_bytes() for name in ("practice_extractor.json",
                                                                        "interaction_signals.json")}
    again = wf.run_stage3(run)  # garde : étape complète et à jour, non rejouée, rien n'est réécrit
    assert again["status"]["status"] == wf.STAGE3_COMPLETE and again["status"]["skipped"] == [si.INTERVIEW_ID]
    assert again["status"]["interviews"][0]["phase"] == "déjà terminée — non rejouée" and not again["status"]["tasks"]
    assert {name: (analysis_dir(run) / name).read_bytes() for name in files} == files
    forced = wf.run_stage3(run, force=True)  # rejouer explicitement : mêmes tâches, aucune nouvelle
    assert forced["status"]["status"] == wf.STAGE3_COMPLETE
    assert [t["task_id"] for t in forced["status"]["tasks"]] == [t["task_id"] for t in status["tasks"]]


def test_workflow_outputs_equal_the_api_pipeline_outputs(tmp_path, monkeypatch):
    """Même requête, même réponse → mêmes sorties : seul le moyen d'obtenir la réponse change."""
    run_wf = si.make_ingested_run(tmp_path / "workflow")
    first = wf.run_stage3(run_wf)
    answer(first["status"], GOOD)
    run_wf = wf.run_stage3(first["metadata"])["metadata"]

    run_api = si.make_ingested_run(tmp_path / "api")
    transport = FakeTransport({PRACTICE: lambda p: text_response(si.GOOD_PRACTICES),
                               INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})
    run_api = analysis.analyze_run(run_api, settings=fake_settings(), transport=transport)

    payloads = {t["agent"]: packet(wf._resolve(t["task_dir"]))["payload"] for t in first["status"]["pending"]}
    for call in transport.calls:  # le paquet contient exactement le message que recevait l'API
        assert call["params"]["messages"][0]["content"] == payloads[call["agent"]]
    for filename, key in (("practice_extractor.json", "practices"), ("interaction_signals.json", "signals")):
        assert read(run_wf, filename)[key] == read(run_api, filename)[key]
    validation_wf, validation_api = (read(r, config.EVIDENCE_VALIDATION_FILENAME) for r in (run_wf, run_api))
    assert validation_wf["agents"] == validation_api["agents"]


def test_schema_invalid_response_is_never_accepted_until_corrected(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path)
    first = wf.run_stage3(run)
    practice_task = next(t for t in first["status"]["pending"] if t["agent"] == PRACTICE)
    practice_dir = wf._resolve(practice_task["task_dir"])
    answer(first["status"], GOOD)
    (practice_dir / wf.RESPONSE_FILENAME).write_text('{"practices": [{"summary": "incomplet"}]}', encoding="utf-8")

    report = wf.check_task(practice_dir)
    assert not report["ok"] and report["blocking"][0].startswith("SCHEMA_VALIDATION")
    second = wf.run_stage3(first["metadata"])
    assert second["status"]["status"] == wf.STAGE3_INVALID
    assert [t["task_id"] for t in second["status"]["invalid"]] == [practice_task["task_id"]]
    agents = second["metadata"]["files"][0]["analysis"]["agents"]
    assert agents[PRACTICE]["status"] == analysis.STATUS_FAILED
    assert agents[PRACTICE]["error"]["code"] == "SCHEMA_VALIDATION"
    assert agents[INTERACTION]["status"] == analysis.STATUS_SUCCESS  # l'autre agent n'est pas affecté
    assert not (analysis_dir(run) / "practice_extractor.json").exists()

    (practice_dir / wf.RESPONSE_FILENAME).write_text("pas du JSON", encoding="utf-8")
    assert wf.check_task(practice_dir)["blocking"][0].startswith("INVALID_JSON")

    (practice_dir / wf.RESPONSE_FILENAME).write_text(json.dumps(si.GOOD_PRACTICES), encoding="utf-8")  # correction
    assert wf.check_task(practice_dir)["ok"]
    assert wf.run_stage3(second["metadata"])["status"]["status"] == wf.STAGE3_COMPLETE


def test_check_blocks_a_fabricated_quote_and_the_pipeline_still_flags_it(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path)
    first = wf.run_stage3(run)
    answer(first["status"], {PRACTICE: si.with_fabricated_quote(si.GOOD_PRACTICES, "practices"),
                             INTERACTION: si.GOOD_SIGNALS})
    practice_dir = next(wf._resolve(t["task_dir"]) for t in first["status"]["pending"] if t["agent"] == PRACTICE)
    report = wf.check_task(practice_dir)
    assert not report["ok"] and report["counts"]["invalid_evidence"] == 1
    assert any(line.startswith("QUOTE_NOT_FOUND") for line in report["blocking"])
    # Si la réponse n'est pas corrigée, TRACE applique la règle habituelle : citation invalide, objet à revoir.
    run = wf.run_stage3(first["metadata"])["metadata"]
    practices = read(run, "practice_extractor.json")
    assert practices["status"] == analysis.STATUS_SUCCESS_WITH_WARNINGS and practices["invalid_evidence_count"] == 1
    first_practice = practices["practices"][0]
    assert first_practice["needs_review"] and "QUOTE_NOT_FOUND" in first_practice["review_reasons"]


def test_check_refuses_a_packet_whose_payload_was_modified(tmp_path):
    first = wf.run_stage3(si.make_ingested_run(tmp_path))
    answer(first["status"], GOOD)
    task_dir = wf._resolve(first["status"]["pending"][0]["task_dir"])
    payload = task_dir / wf.PAYLOAD_FILENAME
    payload.write_text(payload.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = wf.check_task(task_dir)
    assert not report["ok"] and "payload" in report["blocking"][0]


# --- Audit des locuteurs, puis agents avec les speaker_warning ----------------------------------------------

def test_agents_wait_for_the_speaker_audit_and_receive_its_warnings(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path, si.AUDIT_FILES)
    first = wf.run_stage3(run)
    assert [t["agent"] for t in first["status"]["pending"]] == [AUDITOR]  # aucun paquet d'agent avant l'audit
    assert first["status"]["interviews"][0]["phase"] == "audit des locuteurs"
    assert "analysis" not in first["metadata"]["files"][0]
    tasks_root = wf.tasks_dir(first["metadata"])
    assert [p.name.split("__")[1] for p in tasks_root.iterdir()] == [AUDITOR]
    answer(first["status"], {AUDITOR: si.AUDIT_ASSESSMENTS})
    assert wf.check_task(wf._resolve(first["status"]["pending"][0]["task_dir"]))["ok"]

    second = wf.run_stage3(first["metadata"])
    assert sorted(t["agent"] for t in second["status"]["pending"]) == [INTERACTION, PRACTICE]
    for task in second["status"]["pending"]:
        assert '"speaker_warning"' in packet(wf._resolve(task["task_dir"]))["payload"]
    answer(second["status"], EMPTY)
    third = wf.run_stage3(second["metadata"])
    assert third["status"]["status"] == wf.STAGE3_COMPLETE
    summary = third["metadata"]["files"][0]["analysis"]
    assert summary["speaker_audit"]["status"] in analysis.DONE_STATUSES and summary["speaker_audit"]["api_calls"] == 0
    assert summary["agents"][PRACTICE]["speaker_warning_count"] >= 1
    audit = read(third["metadata"], analysis.AUDITOR.output_filename, si.AUDIT_INTERVIEW_ID)
    assert audit["llm_called"] and audit["model"] == wf.WORKFLOW_MODEL and audit["transcript_modified"] is False


# --- Entretien long : blocs, puis lecture à longue distance --------------------------------------------------

def test_long_interview_reads_blocks_then_long_distance(tmp_path, no_api_client):
    from tests import synthetic_long_interview as L
    run = si.make_ingested_run(tmp_path, L.long_files())
    responders = {PRACTICE: EMPTY[PRACTICE], INTERACTION: L.simulated_chunk_reader,
                  LONG_DISTANCE: L.simulated_long_distance_reader, AUDITOR: EMPTY[AUDITOR]}
    result, waves = drive(run, responders)
    agent_waves = [w for w in waves if INTERACTION in w]
    assert len(agent_waves) == 1 and agent_waves[0].count(INTERACTION) > 1 and agent_waves[0].count(PRACTICE) > 1
    long_wave = next(i for i, w in enumerate(waves) if LONG_DISTANCE in w)
    assert long_wave > waves.index(agent_waves[0]) and waves[long_wave] == [LONG_DISTANCE]
    signals = read(result["metadata"], "interaction_signals.json", L.LONG_INTERVIEW_ID)
    assert signals["analysis_complete"] is True and signals["status"] in analysis.DONE_STATUSES
    long_distance = signals["chunking"]["long_distance"]
    assert long_distance["llm_called"] is True and long_distance["signals_kept"] >= 1
    assert result["status"]["api_calls"] == 0


# --- Restauration dans TRACE, consommation par l'étape 4 ----------------------------------------------------

def test_workflow_outputs_restore_into_trace_and_feed_stage4(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path / "workflow")
    first = wf.run_stage3(run)
    answer(first["status"], GOOD)
    run = wf.run_stage3(first["metadata"])["metadata"]
    assert accountability.stage3_state(analysis_dir(run))["status"] == accountability.STAGE3_COMPLETE

    names = ("practice_extractor.json", "interaction_signals.json", config.EVIDENCE_VALIDATION_FILENAME,
             analysis.AUDITOR.output_filename)
    uploads = [(f"{si.INTERVIEW_ID}_{name}", (analysis_dir(run) / name).read_bytes()) for name in names]
    fresh = si.make_ingested_run(tmp_path / "streamlit")  # autre machine : entretien réimporté, sans étape 3
    restored = restore_stage3(fresh, si.INTERVIEW_ID, uploads)
    assert restored["files"][0]["analysis"]["restored"] is True
    assert accountability.stage3_state(analysis_dir(restored))["status"] == accountability.STAGE3_COMPLETE
    for name in names[:2]:
        assert (analysis_dir(restored) / name).read_bytes() == (analysis_dir(run) / name).read_bytes()


# --- Commandes de Claude Code (scripts/trace_workflow.py) ---------------------------------------------------

@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    spec = importlib.util.spec_from_file_location("trace_workflow_cli",
                                                  config.PROJECT_ROOT / "scripts" / "trace_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_runs_stage3_end_to_end(tmp_path, cli, capsys, no_api_client):
    source = tmp_path / si.FILENAME
    source.write_text(si.TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(source)]) == 0
    assert "Étape suivante : python scripts/trace_workflow.py run run_" in capsys.readouterr().out

    assert cli.main(["stage3", "latest"]) == 3
    out = capsys.readouterr().out
    assert "AWAITING_AGENT" in out and "0 appel API" in out and out.count("task.json") == 2
    assert cli.main(["tasks", "latest", "--json"]) == 0
    status = {"pending": json.loads(capsys.readouterr().out)["pending"], "invalid": []}
    answer(status, {PRACTICE: si.with_fabricated_quote(si.GOOD_PRACTICES, "practices"), INTERACTION: si.GOOD_SIGNALS})

    practice_dir = next(t["task_dir"] for t in status["pending"] if t["agent"] == PRACTICE)
    assert cli.main(["check", practice_dir]) == 1
    assert "QUOTE_NOT_FOUND" in capsys.readouterr().out
    (wf._resolve(practice_dir) / wf.RESPONSE_FILENAME).write_text(json.dumps(si.GOOD_PRACTICES), encoding="utf-8")
    assert cli.main(["check", practice_dir]) == 0
    assert "OK — aucune anomalie bloquante" in capsys.readouterr().out

    assert cli.main(["stage3", "latest"]) == 0
    assert "COMPLETE" in capsys.readouterr().out
    assert cli.main(["status", "latest"]) == 0
    assert "Extraction des pratiques : terminé (1/1 entretien(s))" in capsys.readouterr().out
    assert cli.main(["stage3", "inconnu"]) == 2


# --- Interface Streamlit --------------------------------------------------------------------------------------

def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_app_runs_stage3_through_the_workflow_without_key(tmp_path, no_api_client):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = si.make_ingested_run(tmp_path)
    at.run()
    assert not at.exception
    pipeline = "🔵 **2. Extraction des pratiques** (IA) — prête, workflow Claude Code (0 appel API)"
    assert pipeline in texts(at.markdown)
    assert "Workflow Claude Code" in texts(at.info)
    assert "Analyse IA désactivée" not in texts(at.warning)
    assert not [b for b in at.button if b.key == "ai_launch"]  # l'ancien lanceur par API n'est pas proposé

    at.button(key="wf_stage3").click().run()
    assert not at.exception
    assert "tâche(s) d'agent en attente" in texts(at.warning)
    run = at.session_state["last_run"]
    assert "Exécute les tâches TRACE en attente du run" in " ".join(c.value for c in at.code)
    assert "⏳ EN ATTENTE (workflow Claude Code)" in list(at.table[-1].value["Practice Extractor"])

    answer(run["stage3_workflow"], GOOD)  # Claude Code joue les deux agents
    at.button(key="wf_stage3").click().run()
    assert not at.exception
    assert "Étape 3 terminée (workflow Claude Code, 0 appel API)." in texts(at.success)
    table = at.table[-1].value
    assert list(table["Practice Extractor"]) == ["✅ SUCCESS"] and list(table["Pratiques"]) == [6]
    assert "0 appel(s) API — 0 tokens entrée — 0 tokens sortie" in texts(at.markdown)
    labels = [b.label for b in at.get("download_button")]
    assert "Télécharger practice_extractor.json" in labels


def test_app_keeps_the_workflow_even_with_an_api_key(tmp_path, monkeypatch, no_api_client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = si.make_ingested_run(tmp_path)
    at.run()
    assert not at.exception and [b for b in at.button if b.key == "wf_stage3"]
    assert not [b for b in at.button if b.key == "ai_launch"]  # aucun repli vers l'API
    at.button(key="wf_stage3").click().run()
    assert not at.exception and at.session_state["last_run"]["stage3_workflow"]["api_calls"] == 0


def test_app_opens_a_run_prepared_in_claude_code(tmp_path, monkeypatch, no_api_client):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    run = si.make_ingested_run(tmp_path)
    first = wf.run_stage3(run)
    answer(first["status"], GOOD)
    run = wf.run_stage3(first["metadata"])["metadata"]
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "Aucun run lancé pendant cette session." in texts(at.info)
    at.selectbox(key="open_run_id").set_value(run["run_id"])
    at.button(key="open_run").click().run()
    assert not at.exception
    assert f"**Identifiant du run :** `{run['run_id']}`" in texts(at.markdown)
    assert list(at.table[-1].value["Practice Extractor"]) == ["✅ SUCCESS"]
