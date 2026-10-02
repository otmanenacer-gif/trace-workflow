"""Étape 4 en workflow Claude Code : paquets par bloc, réponses d'agent, validation, fusion, restauration, étape 5.

Aucun appel API : en plus des garde-fous de tests/conftest.py (transport réel interdit, réseau refusé, aucune
clé), la construction d'un client d'API (LLMClient) est interdite dans les tests du workflow (`no_api_client`).
L'Accountability Episode Builder est simulé par les lecteurs déterministes existants (tests/synthetic_stage4*.py),
dont les réponses sont écrites dans response.json, là où un sous-agent Claude Code les écrirait.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.llm_client as llm_client
from core import accountability, config, trajectory
from core import accountability_candidates as ac
from core import claude_code_workflow as wf
from core.analysis_cache import AnalysisCache
from core.stage3_restore import restore_stage3
from core.stage4_restore import restore_stage4
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S
from tests import synthetic_stage4_otmane as O
from tests import synthetic_stage5 as S5
from tests.fake_llm import ACCOUNTABILITY, FakeTransport, fake_settings
from tests.test_claude_code_workflow import APP, analysis_dir, answer, no_api_client, packet, texts  # noqa: F401


@pytest.fixture
def api_guard(monkeypatch):
    """Comme `no_api_client`, mais levable : les tests de comparaison exécutent ENSUITE l'ancien pipeline (LLM
    simulé) pour référence ; tout le workflow s'exécute avec la garde active."""
    state = {"forbid": True}
    original = llm_client.LLMClient.__init__

    def guarded(self, *args, **kwargs):
        if state["forbid"]:
            raise AssertionError("Client d'API construit pendant le workflow Claude Code : interdit.")
        original(self, *args, **kwargs)
    monkeypatch.setattr(llm_client.LLMClient, "__init__", guarded)
    return state

STAGE4_FILES = (config.ACCOUNTABILITY_EPISODES_FILENAME, config.ACCOUNTABILITY_VALIDATION_FILENAME,
                config.ACCOUNTABILITY_MANIFEST_FILENAME)
STAGE3_FILES = ("practice_extractor.json", "interaction_signals.json", config.EVIDENCE_VALIDATION_FILENAME,
                "speaker_attribution_audit.json")


def table(at, column: str):
    """Le dernier tableau qui a cette colonne (indépendant de l'ordre des sections)."""
    return [t.value for t in at.table if column in t.value.columns][-1]


def responders(stage3: dict, builder) -> dict:
    return {**stage3, ACCOUNTABILITY: builder}


def until_stage4_tasks(run: dict, agents: dict, max_passes: int = 8) -> dict:
    """Passages `run_until 4` (en jouant les agents de l'étape 3) jusqu'aux premières tâches de l'étape 4."""
    for _ in range(max_passes):
        result = wf.run_until(run, "4")
        run = result["metadata"]
        if result["stage"] == "4":
            return result
        assert result["status"]["status"] == wf.STAGE_AWAITING, result["status"]
        answer(result["status"], agents)
    raise AssertionError("L'étape 3 ne se termine pas")


def finish(result: dict, agents: dict, max_passes: int = 4) -> dict:
    for _ in range(max_passes):
        if result["status"]["status"] == wf.STAGE_COMPLETE:
            return result
        answer(result["status"], agents)
        result = wf.run_until(result["metadata"], "4")
    raise AssertionError("L'étape 4 ne se termine pas")


def episodes_doc(run: dict, interview_id: str) -> dict:
    return json.loads((analysis_dir(run, interview_id) / config.ACCOUNTABILITY_EPISODES_FILENAME).read_text())


def comparable(document: dict) -> dict:
    """Le contenu méthodologique d'accountability_episodes.json, sans horodatage, étiquette de moteur ni empreintes
    des fichiers de l'étape 3 (propres à chaque run : ils sont horodatés)."""
    keys = ("status", "analysis_complete", "candidate_count", "component_count", "episode_count",
            "accountability_episode_count", "ordinary_practice_count", "uncertain_count", "rejected_episode_count",
            "usable_episode_count", "unmarked_practice_count", "validation_error_count", "validation_warning_count",
            "candidates", "unmarked_practices", "episodes")
    return {k: document[k] for k in keys}


@pytest.fixture
def reference(tmp_path, api_guard):
    """Entretien de référence de l'étape 4 : étape 3 terminée par le workflow, premières tâches de l'étape 4."""
    agents = responders(S.stage3_responders(), S.REFERENCE_BUILDER)
    return until_stage4_tasks(si.make_ingested_run(tmp_path, S.FILES), agents), agents


# --- Paquets, réponses, sorties habituelles -------------------------------------------------------------

def test_stage4_runs_without_key_from_one_exact_packet(reference):
    first, agents = reference
    status, run = first["status"], first["metadata"]
    assert status["stage"] == "4" and status["status"] == wf.STAGE_AWAITING and status["api_calls"] == 0
    [task] = status["pending"]
    assert task["agent"] == ACCOUNTABILITY and task["stage"] == "4"
    assert task["task_dir"].startswith(str(Path(run["output_dir"]) / "workflow" / "stage4" / "tasks"))
    p = packet(wf._resolve(task["task_dir"]))
    prepared = accountability.prepare_stage4(run["files"][0]["ingestion"])
    assert p["task"]["stage"] == "4" and p["task"]["system_prompt"]["path"] == "prompts/accountability_episode_builder.md"
    assert p["system_prompt"] == accountability.SPEC.system_prompt  # le prompt du dépôt, inchangé
    assert p["payload"] == prepared.requests[0]["user_message"]    # le message que recevait l'API
    assert p["schema"] == accountability.SPEC.output_schema
    assert not (analysis_dir(run, S.INTERVIEW_ID) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()
    assert run["files"][0]["accountability"]["error"]["code"] == wf.AWAITING_AGENT
    assert "1 tâche(s) en attente" in run["pipeline"][config.ACCOUNTABILITY_STEP]

    answer(status, agents)
    report = wf.check_task(wf._resolve(task["task_dir"]))
    assert report["ok"] and report["counts"]["candidates"] == prepared.candidate_count and not report["blocking"]

    done = wf.run_until(run, "4")
    assert done["stage"] == "4" and done["status"]["status"] == wf.STAGE_COMPLETE
    run = done["metadata"]
    out = analysis_dir(run, S.INTERVIEW_ID)
    assert all((out / name).is_file() for name in STAGE4_FILES)
    manifest = json.loads((out / config.ACCOUNTABILITY_MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "SUCCESS" and manifest["api_calls"] == 0 and manifest["billed_this_run"] is False
    assert manifest["model"] == wf.WORKFLOW_MODEL and manifest["request_id"] == task["task_id"]
    document = episodes_doc(run, S.INTERVIEW_ID)
    assert (document["accountability_episode_count"], document["unmarked_practice_count"]) == (4, 4)
    assert document["validation_error_count"] == 0 and document["llm_called"] is True
    assert run["pipeline"][config.ACCOUNTABILITY_STEP] == "terminé (1/1 entretien(s))"
    assert run["last_accountability"]["usage"]["api_calls"] == 0
    assert run["stage4_workflow"]["status"] == wf.STAGE_COMPLETE


def test_run_until_never_replays_a_complete_stage3(reference):
    first, agents = reference
    out = analysis_dir(first["metadata"], S.INTERVIEW_ID)
    before = {name: (out / name).read_bytes() for name in STAGE3_FILES}
    answer(first["status"], agents)
    run = wf.run_until(first["metadata"], "4")["metadata"]
    run = wf.run_until(run, "4")["metadata"]
    assert {name: (out / name).read_bytes() for name in STAGE3_FILES} == before  # étape 3 intacte
    assert trajectory.stage4_state(out.parent)["status"] == trajectory.STAGE4_COMPLETE  # donc jamais périmée


def test_stage4_is_blocked_without_stage3_and_writes_no_task(tmp_path, no_api_client):
    run = si.make_ingested_run(tmp_path, S.FILES)
    result = wf.run_stage4(run)
    assert result["status"]["status"] == wf.STAGE_BLOCKED and result["status"]["task_count"] == 0
    assert result["metadata"]["files"][0]["accountability"]["status"] == accountability.STATUS_BLOCKED
    assert "Étape 4 non exécutée" in result["status"]["interviews"][0]["reason"]
    assert not wf.tasks_dir(run, "4").exists()


def test_same_outputs_as_the_api_pipeline_with_identical_responses(reference, tmp_path, api_guard):
    first, agents = reference
    answer(first["status"], agents)
    run_wf = wf.run_until(first["metadata"], "4")["metadata"]
    api_guard["forbid"] = False  # référence : l'ancien pipeline, LLM simulé

    run_api, _ = S5.run_to_stage4(tmp_path / "api", S.FILES, S.stage3_responders(), S.REFERENCE_BUILDER)
    assert comparable(episodes_doc(run_wf, S.INTERVIEW_ID)) == comparable(episodes_doc(run_api, S.INTERVIEW_ID))
    validation = [json.loads((analysis_dir(r, S.INTERVIEW_ID) / config.ACCOUNTABILITY_VALIDATION_FILENAME).read_text())
                  for r in (run_wf, run_api)]
    assert [{k: v for k, v in d.items() if k != "validated_at"} for d in validation][0] == \
           [{k: v for k, v in d.items() if k != "validated_at"} for d in validation][1]


# --- Réponses refusées ---------------------------------------------------------------------------------------

def test_invalid_json_and_schema_are_refused_until_corrected(reference):
    first, agents = reference
    [task] = first["status"]["pending"]
    task_dir = wf._resolve(task["task_dir"])
    response = task_dir / wf.RESPONSE_FILENAME
    for text, code in (("pas du JSON", "INVALID_JSON"), ('{"episodes": [{"candidate_ids": []}]}', "SCHEMA_VALIDATION")):
        response.write_text(text, encoding="utf-8")
        report = wf.check_task(task_dir)
        assert not report["ok"] and report["blocking"][0].startswith(code)
        result = wf.run_until(first["metadata"], "4")
        assert result["status"]["status"] == wf.STAGE_INVALID
        assert [t["task_id"] for t in result["status"]["invalid"]] == [task["task_id"]]
        summary = result["metadata"]["files"][0]["accountability"]
        assert summary["status"] == "FAILED" and summary["error"]["code"] == code
        assert not (analysis_dir(result["metadata"], S.INTERVIEW_ID) / config.ACCOUNTABILITY_EPISODES_FILENAME).exists()
    response.unlink()
    answer(first["status"], agents)  # correction
    assert wf.check_task(task_dir)["ok"]
    assert wf.run_until(first["metadata"], "4")["status"]["status"] == wf.STAGE_COMPLETE


def test_invalid_quote_is_blocked_by_check_and_flagged_by_the_validator(reference):
    first, agents = reference
    answer(first["status"], agents)
    [task] = first["status"]["pending"]
    response = wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME
    output = json.loads(response.read_text(encoding="utf-8"))
    output["episodes"][0]["evidence"][0]["quote"] = "Je fais tout écrire par ChatGPT sans jamais relire."
    response.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    report = wf.check_task(response.parent)
    assert not report["ok"] and any(line.startswith("QUOTE_NOT_FOUND") for line in report["blocking"])
    # Non corrigée, la réponse est jugée par la règle habituelle : citation invalide, épisode à revoir.
    run = wf.run_until(first["metadata"], "4")["metadata"]
    document = episodes_doc(run, S.INTERVIEW_ID)
    flagged = [e for e in document["episodes"] if "QUOTE_NOT_FOUND" in e["review_reasons"]]
    assert flagged and flagged[0]["needs_review"] and document["validation_error_count"] >= 1
    assert document["status"] == "SUCCESS_WITH_WARNINGS"


# --- Plusieurs paquets : un par bloc de composantes ----------------------------------------------------------

@pytest.fixture
def otmane_blocks(tmp_path, monkeypatch, no_api_client):
    """Entretien de régression long, seuil d'un appel abaissé : plusieurs blocs de composantes entières."""
    monkeypatch.setattr(ac, "SINGLE_CALL_MAX_INPUT_TOKENS", 2500)

    def start(name: str) -> tuple[dict, dict]:
        agents = responders(O.stage3_responders(), O.builder())
        return until_stage4_tasks(si.make_ingested_run(tmp_path / name, O.files()), agents), agents
    return start


def test_several_packets_are_independent_validated_separately_and_merged_in_block_order(otmane_blocks):
    first, agents = otmane_blocks("ordre_inverse")
    pending = sorted(first["status"]["pending"], key=lambda t: t["label"])
    prepared = accountability.prepare_stage4(first["metadata"]["files"][0]["ingestion"])
    assert len(pending) == len(prepared.requests) > 1
    assert [t["label"].rsplit("/", 1)[1] for t in pending] == [f"bloc{r['chunk']}" for r in prepared.requests]
    payloads = {packet(wf._resolve(t["task_dir"]))["payload"] for t in pending}
    assert payloads == {r["user_message"] for r in prepared.requests}  # un paquet par message, rien d'autre

    # Ordre d'exécution inversé : le dernier bloc d'abord, puis les autres.
    last = pending[-1]
    answer({"pending": [last], "invalid": []}, agents)
    assert wf.check_task(wf._resolve(last["task_dir"]))["ok"]  # validé seul, sur les candidats de son bloc
    partial = wf.run_until(first["metadata"], "4")
    assert partial["status"]["status"] == wf.STAGE_AWAITING and len(partial["status"]["pending"]) == len(pending) - 1
    summary = partial["metadata"]["files"][0]["accountability"]
    assert summary["status"] == "PARTIAL" and summary["analysis_complete"] is False  # jamais présenté comme complet
    answer(partial["status"], agents)
    for task in pending:
        report = wf.check_task(wf._resolve(task["task_dir"]))
        assert report["ok"] and "CANDIDATE_NOT_ADDRESSED" not in " ".join(report["warnings"])
    reversed_run = wf.run_until(partial["metadata"], "4")
    assert reversed_run["status"]["status"] == wf.STAGE_COMPLETE

    # Même entretien, toutes les réponses d'un coup : même résultat final.
    second, _ = otmane_blocks("en_une_fois")
    together = finish(second, agents)
    a, b = (episodes_doc(r["metadata"], O.INTERVIEW_ID) for r in (reversed_run, together))
    assert a["chunk_count"] == b["chunk_count"] == len(pending)
    assert comparable(a) == comparable(b)
    assert a["status"] == "SUCCESS" and a["usable_episode_count"] == a["candidate_count"] - 1


# --- Restauration et étape 5 ---------------------------------------------------------------------------------

def test_workflow_stage4_restores_into_trace_and_feeds_stage5(reference, tmp_path, api_guard):
    first, agents = reference
    answer(first["status"], agents)
    run = wf.run_until(first["metadata"], "4")["metadata"]
    out = analysis_dir(run, S.INTERVIEW_ID)

    # Autre machine : entretien réimporté, étape 3 puis étape 4 restaurées depuis les fichiers (0 appel).
    fresh = si.make_ingested_run(tmp_path / "streamlit", S.FILES)
    fresh = restore_stage3(fresh, S.INTERVIEW_ID, [(n, (out / n).read_bytes()) for n in STAGE3_FILES])
    stage4 = [(n, (out / n).read_bytes()) for n in STAGE4_FILES[:2]]
    fresh = restore_stage4(fresh, S.INTERVIEW_ID, stage4)
    restored_dir = analysis_dir(fresh, S.INTERVIEW_ID)
    assert trajectory.stage4_state(restored_dir.parent)["status"] == trajectory.STAGE4_COMPLETE
    assert fresh["files"][0]["accountability"]["restored"] is True

    # L'étape 5 (pas encore migrée : LLM simulé) reçoit EXACTEMENT le même matériau qu'après l'ancien pipeline.
    api_guard["forbid"] = False
    run_api, cache = S5.run_to_stage4(tmp_path / "api", S.FILES, S.stage3_responders(), S.REFERENCE_BUILDER)
    mapper = S5.scripted_mapper(S5.REFERENCE_PLAN)
    _, from_api = S5.run_stage5(run_api, AnalysisCache(tmp_path / "c1"), mapper)
    run, from_workflow = S5.run_stage5(run, AnalysisCache(tmp_path / "c2"), mapper)
    _, from_restore = S5.run_stage5(fresh, AnalysisCache(tmp_path / "c3"), mapper)
    sent = [t.calls[0]["params"]["messages"][0]["content"] for t in (from_api, from_workflow, from_restore)]
    assert sent[0] == sent[1] == sent[2]
    assert run["files"][0]["trajectory"]["status"] == "SUCCESS"


# --- Commandes de Claude Code ---------------------------------------------------------------------------------

def test_cli_runs_trace_until_stage4(tmp_path, monkeypatch, capsys, no_api_client):
    import importlib.util
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    spec = importlib.util.spec_from_file_location("trace_workflow_cli4",
                                                  config.PROJECT_ROOT / "scripts" / "trace_workflow.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = tmp_path / S.FILENAME
    source.write_text(S.TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(source)]) == 0
    capsys.readouterr()
    agents = responders(S.stage3_responders(), S.REFERENCE_BUILDER)
    stages = []
    for _ in range(6):
        code = cli.main(["run", "latest", "--until", "4", "--json"])
        status = json.loads(capsys.readouterr().out)
        stages.append((status["stage"], code))
        if code == 0:
            break
        assert code == 3
        answer(status, agents)
    assert stages[-1] == ("4", 0) and ("3", 3) in stages and ("4", 3) in stages
    assert cli.main(["tasks", "latest", "--stage", "4", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"pending": [], "invalid": []}
    task_dir = wf.read_status(cli.resolve_run("latest"), "4")["tasks"][0]["task_dir"]
    assert cli.main(["check", task_dir]) == 0
    assert cli.main(["stage4", "latest"]) == 0
    assert "étape 4 (workflow Claude Code, 0 appel API) : COMPLETE" in capsys.readouterr().out
    assert cli.main(["status", "latest"]) == 0
    out = capsys.readouterr().out
    assert "Construction des épisodes d'accountability : terminé (1/1 entretien(s))" in out and "étape 3" in out


# --- Interface Streamlit -----------------------------------------------------------------------------------

def test_app_runs_stage4_through_the_workflow_without_key(reference, monkeypatch):
    first, agents = reference
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = first["metadata"]
    at.run()
    assert not at.exception
    assert "Analyse IA désactivée" not in texts(at.warning)
    assert not [b for b in at.button if b.key == "acc_launch"]  # l'ancien lanceur par API n'est pas proposé
    assert "⏳ EN ATTENTE (workflow Claude Code)" in list(table(at, "Étape 4")["Étape 4"])
    assert f"Exécute TRACE sur le run {first['metadata']['run_id']} jusqu'à l'étape 4" in " ".join(
        c.value for c in at.code)

    [task] = first["status"]["pending"]  # réponse invalide : erreur affichée, aucun appel
    (wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME).write_text("pas du JSON", encoding="utf-8")
    at.button(key="wf_stage4").click().run()
    assert not at.exception and "réponse(s) d'agent à corriger" in texts(at.warning)
    assert list(table(at, "État")["État"]) == ["réponse à corriger"]
    assert "JSON invalide" in " ".join(table(at, "Erreur")["Erreur"])

    (wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME).unlink()
    answer(first["status"], agents)  # Claude Code joue l'agent
    at.button(key="wf_stage4").click().run()
    assert not at.exception
    assert "Étape 4 terminée (workflow Claude Code, 0 appel API)." in texts(at.success)
    results = table(at, "Étape 4")
    assert list(results["Étape 4"]) == ["✅ SUCCESS"] and list(results["Épisodes accountability"]) == [4]
    assert list(results["Appels API"]) == [0]
    assert "réponse(s) d'agent du workflow Claude Code (0 appel API)" in texts(at.markdown)
    assert "🟢 **4. Construction des épisodes d'accountability** (IA) — terminé (1/1 entretien(s))" in texts(at.markdown)
    labels = [b.label for b in at.get("download_button")]
    assert f"Télécharger {config.ACCOUNTABILITY_EPISODES_FILENAME}" in labels
    assert f"Télécharger {config.ACCOUNTABILITY_VALIDATION_FILENAME}" in labels


def test_app_shows_stage4_ready_and_keeps_the_workflow_even_with_a_key(reference, monkeypatch):
    first, _ = reference
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    run = dict(first["metadata"])
    run.pop("stage4_workflow")
    run["files"] = [{k: v for k, v in f.items() if k != "accountability"} for f in run["files"]]
    run["pipeline"] = {**run["pipeline"], config.ACCOUNTABILITY_STEP: "inactive"}
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    at.run()
    assert not at.exception
    assert ("🔵 **4. Construction des épisodes d'accountability** (IA) — prête, workflow Claude Code (0 appel API)"
            in texts(at.markdown))
    assert [b for b in at.button if b.key == "wf_stage4"] and not [b for b in at.button if b.key == "acc_launch"]
    at.button(key="wf_stage4").click().run()  # aucun repli vers l'API, même avec une clé
    assert not at.exception and at.session_state["last_run"]["stage4_workflow"]["api_calls"] == 0
