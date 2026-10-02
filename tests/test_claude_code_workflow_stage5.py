"""Étape 5 en workflow Claude Code : paquet unique, réponse d'agent, validation, requalifications, garde, étape 6.

Aucun appel API : en plus des garde-fous de tests/conftest.py (transport réel interdit, réseau refusé, aucune
clé), la construction d'un client d'API (LLMClient) est interdite pendant tout le workflow (`api_guard`). Les tests
de comparaison exécutent ENSUITE l'ancien pipeline (LLM simulé) pour référence. Le Trajectory Mapper est simulé
par les lecteurs scriptés existants (tests/synthetic_stage5.py), dont les réponses sont écrites dans response.json,
là où un sous-agent Claude Code les écrirait.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import config, cross_interview, trajectory
from core import claude_code_workflow as wf
from core import cross_interview_corpus as corpus
from core import trajectory_candidates as tc
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S
from tests import synthetic_stage5 as S5
from tests.fake_llm import ACCOUNTABILITY, TRAJECTORY, text_response
from tests.test_claude_code_workflow import APP, analysis_dir, answer, packet, texts
from tests.test_claude_code_workflow_stage4 import api_guard, table  # noqa: F401

STAGE5_FILES = (config.STUDENT_TRAJECTORY_FILENAME, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME,
                config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)
EARLIER_FILES = ("practice_extractor.json", "interaction_signals.json", config.EVIDENCE_VALIDATION_FILENAME,
                 "speaker_attribution_audit.json", config.ACCOUNTABILITY_EPISODES_FILENAME,
                 config.ACCOUNTABILITY_VALIDATION_FILENAME)


class Pipeline:
    """Un entretien synthétique et ses agents simulés, pour le workflow ET pour l'ancien pipeline."""

    def __init__(self, interview_id, files, stage3, builder, mapper):
        self.interview_id, self.files, self.stage3, self.builder, self.mapper = (interview_id, files, stage3, builder,
                                                                                 mapper)

    @property
    def agents(self) -> dict:
        return {**self.stage3, ACCOUNTABILITY: self.builder, TRAJECTORY: self.mapper}

    @classmethod
    def reference(cls, mapper=None):
        return cls(S.INTERVIEW_ID, S.FILES, S.stage3_responders(), S.REFERENCE_BUILDER,
                   mapper or S5.scripted_mapper(S5.REFERENCE_PLAN))

    @classmethod
    def case(cls, case, mode="good", mapper=None):
        return cls(case.interview_id, case.files, case.stage3_responders(), case.builder(), mapper or case.mapper(mode))

    def api(self, tmp_path) -> dict:
        """L'ancien pipeline (LLM simulé) jusqu'à l'étape 5, pour comparaison."""
        run, cache = S5.run_to_stage4(tmp_path, self.files, self.stage3, self.builder)
        run, _ = S5.run_stage5(run, AnalysisCache(tmp_path / "cache5"), self.mapper)
        return run


def altered(mapper, mutate):
    """La même réponse simulée, modifiée de façon déterministe (identique pour le workflow et l'ancien pipeline)."""
    def respond(params):
        output = json.loads(mapper(params).content[0].text)
        mutate(output)
        return text_response(output)
    return respond


def until(run: dict, agents: dict, stage: str, max_passes: int = 10) -> dict:
    """Passages `run_until` jusqu'aux tâches (ou à la fin) de l'étape `stage`, en jouant les étapes précédentes."""
    for _ in range(max_passes):
        result = wf.run_until(run, stage)
        run = result["metadata"]
        if result["stage"] == stage:
            return result
        assert result["status"]["status"] == wf.STAGE_AWAITING, result["status"]
        answer(result["status"], agents)
    raise AssertionError(f"L'étape {stage} n'est pas atteinte")


def finish(run: dict, agents: dict, stage: str = "5", max_passes: int = 10) -> dict:
    for _ in range(max_passes):
        result = wf.run_until(run, stage)
        run = result["metadata"]
        if result["stage"] == stage and result["status"]["status"] == wf.STAGE_COMPLETE:
            return result
        assert result["status"]["status"] == wf.STAGE_AWAITING, result["status"]
        answer(result["status"], agents)
    raise AssertionError(f"L'étape {stage} ne se termine pas")


def load(run: dict, name: str, interview_id: str) -> dict:
    return json.loads((analysis_dir(run, interview_id) / name).read_text(encoding="utf-8"))


def comparable(document: dict) -> dict:
    """student_trajectory.json sans horodatage, étiquette de moteur ni empreintes des fichiers (propres au run)."""
    return {k: v for k, v in document.items() if k not in ("generated_at", "model", "source_hashes", "cache_hit")}


def both(tmp_path, pipeline: Pipeline, api_guard) -> tuple[dict, dict, dict]:
    """(run du workflow, document du workflow, document de l'ancien pipeline), réponses identiques."""
    run = finish(si.make_ingested_run(tmp_path / "workflow", pipeline.files), pipeline.agents)["metadata"]
    api_guard["forbid"] = False  # référence : l'ancien pipeline, LLM simulé
    api_run = pipeline.api(tmp_path / "api")
    iid = pipeline.interview_id
    for name in (config.STUDENT_TRAJECTORY_VALIDATION_FILENAME,):
        a, b = (load(r, name, iid) for r in (run, api_run))
        assert {k: v for k, v in a.items() if k != "validated_at"} == {k: v for k, v in b.items() if k != "validated_at"}
    return run, load(run, config.STUDENT_TRAJECTORY_FILENAME, iid), load(api_run, config.STUDENT_TRAJECTORY_FILENAME, iid)


@pytest.fixture
def reference(tmp_path, api_guard):
    """Entretien de référence : étapes 3 et 4 terminées par le workflow, première tâche de l'étape 5."""
    pipeline = Pipeline.reference()
    return until(si.make_ingested_run(tmp_path, S.FILES), pipeline.agents, "5"), pipeline


# --- Paquet, réponse, sorties habituelles ----------------------------------------------------------------

def test_stage5_runs_without_key_from_one_exact_packet(reference):
    first, pipeline = reference
    status, run = first["status"], first["metadata"]
    assert status["stage"] == "5" and status["status"] == wf.STAGE_AWAITING and status["api_calls"] == 0
    [task] = status["pending"]
    assert task["agent"] == TRAJECTORY and task["stage"] == "5"
    p = packet(wf._resolve(task["task_dir"]))
    prepared = trajectory.prepare_stage5(run["files"][0]["ingestion"])
    assert p["task"]["system_prompt"]["path"] == "prompts/trajectory_mapper.md"
    assert p["system_prompt"] == trajectory.SPEC.system_prompt  # le prompt du dépôt, inchangé
    assert p["payload"] == prepared.request["user_message"]      # le message que recevait l'API
    assert p["schema"] == trajectory.SPEC.output_schema
    [interview] = status["interviews"]
    assert interview["estimated_input_tokens"] == prepared.estimated_input_tokens < tc.SINGLE_CALL_MAX_INPUT_TOKENS
    assert interview["over_single_call_threshold"] is False and interview["warnings"] == []
    assert not (analysis_dir(run, S.INTERVIEW_ID) / config.STUDENT_TRAJECTORY_FILENAME).exists()
    assert run["files"][0]["trajectory"]["error"]["code"] == wf.AWAITING_AGENT
    assert "1 tâche(s) en attente" in run["pipeline"][config.TRAJECTORY_STEP]

    answer(status, pipeline.agents)
    report = wf.check_task(wf._resolve(task["task_dir"]))
    assert report["ok"] and not report["blocking"]
    assert report["counts"]["kept"] == 3 and report["counts"]["criteria"] == 2

    done = wf.run_until(run, "5")
    assert done["stage"] == "5" and done["status"]["status"] == wf.STAGE_COMPLETE
    run = done["metadata"]
    assert all((analysis_dir(run, S.INTERVIEW_ID) / name).is_file() for name in STAGE5_FILES)
    manifest = load(run, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME, S.INTERVIEW_ID)
    assert manifest["status"] == "SUCCESS" and manifest["api_calls"] == 0 and manifest["billed_this_run"] is False
    assert manifest["model"] == wf.WORKFLOW_MODEL and manifest["request_id"] == task["task_id"]
    assert manifest["llm_called"] is True
    document = load(run, config.STUDENT_TRAJECTORY_FILENAME, S.INTERVIEW_ID)
    assert document["configuration_type"] == "contextual_configuration" and document["kept_claim_count"] == 3
    assert run["pipeline"][config.TRAJECTORY_STEP] == "terminé (1/1 entretien(s))"
    assert run["last_trajectory"]["usage"]["api_calls"] == 0
    assert wf.stage5_state(analysis_dir(run, S.INTERVIEW_ID).parent)["status"] == wf.STAGE5_COMPLETE


# --- Garde : run --until 5 ne rejoue pas les étapes terminées ----------------------------------------------

def test_run_until_5_never_replays_completed_stages(reference):
    first, pipeline = reference
    answer(first["status"], pipeline.agents)
    run = wf.run_until(first["metadata"], "5")["metadata"]
    out = analysis_dir(run, S.INTERVIEW_ID)
    before = {name: (out / name).read_bytes() for name in EARLIER_FILES + STAGE5_FILES}

    again = wf.run_until(run, "5")
    assert again["stage"] == "5" and again["status"]["status"] == wf.STAGE_COMPLETE
    assert again["status"]["skipped"] == [S.INTERVIEW_ID] and not again["status"]["tasks"]
    for stage in wf.STAGES:  # chaque étape, appelée seule, est elle aussi gardée
        assert wf.run_stage(stage, run)["status"]["skipped"] == [S.INTERVIEW_ID]
    assert {name: (out / name).read_bytes() for name in before} == before  # rien n'a été réécrit

    # Rejouer EXPLICITEMENT une étape complète est possible (réponses relues, aucune nouvelle tâche).
    forced = wf.run_stage3(run, force=True)
    assert forced["status"]["status"] == wf.STAGE_COMPLETE and not forced["status"]["skipped"]
    assert not forced["status"]["pending"]

    # Une étape 3 réécrite (ici : mêmes données, autres octets) rend les étapes 4 et 5 périmées : la garde les
    # rejoue alors, sans nouvelle tâche (mêmes paquets, réponses relues).
    practices = out / "practice_extractor.json"
    practices.write_text(json.dumps(json.loads(practices.read_text(encoding="utf-8")), ensure_ascii=False, indent=1),
                         encoding="utf-8")
    assert trajectory.stage4_state(out.parent)["status"] == trajectory.STAGE4_STALE
    assert wf.stage5_state(out.parent)["status"] == wf.STAGE5_STALE
    redone = wf.run_until(forced["metadata"], "5")
    assert redone["status"]["status"] == wf.STAGE_COMPLETE and not redone["status"]["skipped"]
    assert [t["state"] for t in redone["status"]["tasks"]] == [wf.TASK_ANSWERED]  # réponse relue, aucune nouvelle
    assert trajectory.stage4_state(out.parent)["status"] == trajectory.STAGE4_COMPLETE
    assert wf.stage5_state(out.parent)["status"] == wf.STAGE5_COMPLETE


# --- Réponses refusées ---------------------------------------------------------------------------------------

def test_invalid_json_and_schema_are_refused_until_corrected(reference):
    first, pipeline = reference
    [task] = first["status"]["pending"]
    task_dir = wf._resolve(task["task_dir"])
    for text, code in (("pas du JSON", "INVALID_JSON"), ('{"configuration_type": "mixed", "claims": []}',
                                                        "SCHEMA_VALIDATION")):
        (task_dir / wf.RESPONSE_FILENAME).write_text(text, encoding="utf-8")
        report = wf.check_task(task_dir)
        assert not report["ok"] and report["blocking"][0].startswith(code)
        result = wf.run_until(first["metadata"], "5")
        assert result["stage"] == "5" and result["status"]["status"] == wf.STAGE_INVALID
        summary = result["metadata"]["files"][0]["trajectory"]
        assert summary["status"] == "FAILED" and summary["error"]["code"] == code
        assert not (analysis_dir(result["metadata"], S.INTERVIEW_ID) / config.STUDENT_TRAJECTORY_FILENAME).exists()
    (task_dir / wf.RESPONSE_FILENAME).unlink()
    answer(first["status"], pipeline.agents)  # correction
    assert wf.check_task(task_dir)["ok"]
    assert wf.run_until(first["metadata"], "5")["status"]["status"] == wf.STAGE_COMPLETE


def test_invented_identifier_is_refused_by_check_and_rejected_by_the_validator(tmp_path, api_guard):
    def invent(output):
        output["claims"][0]["episode_ids"] = ["E999"]
    pipeline = Pipeline.reference(altered(S5.scripted_mapper(S5.REFERENCE_PLAN), invent))
    first = until(si.make_ingested_run(tmp_path, S.FILES), pipeline.agents, "5")
    answer(first["status"], pipeline.agents)
    report = wf.check_task(wf._resolve(first["status"]["pending"][0]["task_dir"]))
    assert not report["ok"] and any(line.startswith("UNKNOWN_EPISODE_ID") for line in report["blocking"])
    run = wf.run_until(first["metadata"], "5")["metadata"]  # non corrigée : règle habituelle du validateur
    claims = load(run, config.STUDENT_TRAJECTORY_FILENAME, S.INTERVIEW_ID)["trajectory_claims"]
    invented = [c for c in claims if "UNKNOWN_EPISODE_ID" in c["review_reasons"]]
    assert invented and not any(f"{S.INTERVIEW_ID}_E999" in c["support_ids"] for c in claims)


def test_invalid_anchor_is_blocked_by_check_and_treated_as_before(tmp_path, api_guard):
    def break_anchor(output):
        claim = next(c for c in output["claims"] if c["temporal_anchors"])
        claim["temporal_anchors"][0]["text"] = "pendant les vacances de Noël"
    pipeline = Pipeline.case(S5.TEMPORAL, mapper=altered(S5.TEMPORAL.mapper(), break_anchor))
    first = until(si.make_ingested_run(tmp_path / "check", pipeline.files), pipeline.agents, "5")
    answer(first["status"], pipeline.agents)
    report = wf.check_task(wf._resolve(first["status"]["pending"][0]["task_dir"]))
    assert not report["ok"] and any(line.startswith("ANCHOR_NOT_FOUND") for line in report["blocking"])
    _, workflow_doc, api_doc = both(tmp_path, pipeline, api_guard)
    assert comparable(workflow_doc) == comparable(api_doc)
    assert any("ANCHOR_NOT_FOUND" in c["review_reasons"] for c in workflow_doc["trajectory_claims"])


# --- Mêmes sorties, requalifications et needs_review que l'ancien pipeline ----------------------------------

def test_same_outputs_as_the_api_pipeline_with_identical_responses(tmp_path, api_guard):
    _, workflow_doc, api_doc = both(tmp_path, Pipeline.reference(), api_guard)
    assert comparable(workflow_doc) == comparable(api_doc)
    assert workflow_doc["status"] == api_doc["status"] == "SUCCESS"


def test_requalifications_are_identical_to_the_api_pipeline(tmp_path, api_guard):
    pipeline = Pipeline.case(S5.CONTEXT, "adversarial")
    first = until(si.make_ingested_run(tmp_path / "check", pipeline.files), pipeline.agents, "5")
    answer(first["status"], pipeline.agents)
    report = wf.check_task(wf._resolve(first["status"]["pending"][0]["task_dir"]))
    assert report["counts"]["requalified"] >= 1 and any("requalifiée" in line for line in report["info"])
    _, workflow_doc, api_doc = both(tmp_path, pipeline, api_guard)
    assert comparable(workflow_doc) == comparable(api_doc)
    requalified = [c for c in workflow_doc["trajectory_claims"] if "model_claim_type" in c]
    assert requalified and workflow_doc["requalified_claim_count"] == len(requalified)
    assert all(c["needs_review"] and any(r.endswith("_REQUALIFIED") for r in c["review_reasons"])
               for c in requalified)


def test_needs_review_is_kept_as_in_the_api_pipeline(tmp_path, api_guard):
    _, workflow_doc, api_doc = both(tmp_path, Pipeline.case(S5.REVIEW), api_guard)
    assert comparable(workflow_doc) == comparable(api_doc)
    assert any(c["needs_review"] and not c["model_needs_review"] for c in workflow_doc["trajectory_claims"])


# --- Paquet trop long : mesuré, signalé, jamais découpé ni confié à une API ---------------------------------

def test_oversized_packet_stays_single_and_is_flagged(reference, monkeypatch):
    first, pipeline = reference
    monkeypatch.setattr(tc, "SINGLE_CALL_MAX_INPUT_TOKENS", 500)
    result = wf.run_stage5(first["metadata"])
    [task] = result["status"]["pending"]  # toujours UN paquet
    [interview] = result["status"]["interviews"]
    assert interview["over_single_call_threshold"] is True and "au-delà du seuil" in interview["warnings"][0]
    answer(result["status"], pipeline.agents)
    report = wf.check_task(wf._resolve(task["task_dir"]))
    assert report["ok"] and any(line.startswith("PAYLOAD_OVER_THRESHOLD") for line in report["warnings"])
    done = wf.run_stage5(result["metadata"])
    assert done["status"]["status"] == wf.STAGE_COMPLETE and done["status"]["interviews"][0]["warnings"]
    document = load(done["metadata"], config.STUDENT_TRAJECTORY_FILENAME, S.INTERVIEW_ID)
    validation = load(done["metadata"], config.STUDENT_TRAJECTORY_VALIDATION_FILENAME, S.INTERVIEW_ID)
    assert document["status"] == "SUCCESS_WITH_WARNINGS"
    assert "PAYLOAD_OVER_THRESHOLD" in {i["code"] for i in validation["issues"]}
    assert load(done["metadata"], config.STUDENT_TRAJECTORY_MANIFEST_FILENAME, S.INTERVIEW_ID)["api_calls"] == 0


# --- Restauration (import des triplets) et étape 6 -----------------------------------------------------------

def test_stage5_outputs_restore_and_are_recognized_by_stage6(tmp_path, api_guard):
    run = finish(si.make_ingested_run(tmp_path / "workflow", S.FILES), Pipeline.reference().agents)["metadata"]
    out = analysis_dir(run, S.INTERVIEW_ID)
    downloaded = [(f"{S.INTERVIEW_ID}_{name}", (out / name).read_bytes()) for name in STAGE5_FILES]
    assert sorted(d for _, d in downloaded) == sorted(d for _, d in corpus.run_stage5_uploads(run))

    api_guard["forbid"] = False  # second entretien, ancien pipeline (LLM simulé) : un corpus de deux entretiens
    other = Pipeline.case(S5.TEMPORAL).api(tmp_path / "other")
    checked = corpus.check_corpus(downloaded + corpus.run_stage5_uploads(other))
    rows = {r["interview_id"]: r for r in checked["rows"]}
    assert rows[S.INTERVIEW_ID]["status"] == corpus.STATUS_USABLE and checked["n_usable"] == 2
    prepared = cross_interview.prepare_stage6(downloaded + corpus.run_stage5_uploads(other))
    assert not prepared.blocked and prepared.material is not None and prepared.request is not None
    assert S.INTERVIEW_ID in prepared.input_hashes


# --- Commandes de Claude Code --------------------------------------------------------------------------------

def test_cli_runs_trace_until_stage5_and_guards_completed_stages(tmp_path, monkeypatch, capsys, api_guard):
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    spec = importlib.util.spec_from_file_location("trace_workflow_cli5",
                                                  config.PROJECT_ROOT / "scripts" / "trace_workflow.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = tmp_path / S.FILENAME
    source.write_text(S.TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(source)]) == 0
    assert "--until 5" in capsys.readouterr().out
    agents, stages = Pipeline.reference().agents, []
    for _ in range(8):
        code = cli.main(["run", "latest", "--until", "5", "--json"])
        status = json.loads(capsys.readouterr().out)
        stages.append((status["stage"], code))
        if code == 0:
            break
        assert code == 3
        answer(status, agents)
    assert stages[-1] == ("5", 0) and {("3", 3), ("4", 3), ("5", 3)} <= set(stages)
    assert cli.main(["run", "latest", "--until", "5"]) == 0  # rien à rejouer
    assert "phase déjà terminée — non rejouée" in capsys.readouterr().out
    assert cli.main(["stage5", "latest", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped"] == [S.INTERVIEW_ID]
    assert cli.main(["stage5", "latest", "--force", "--json"]) == 0  # rejouée explicitement : réponse relue
    assert json.loads(capsys.readouterr().out)["skipped"] == []
    task_dir = wf.read_status(cli.resolve_run("latest"), "5")["tasks"][0]["task_dir"]
    assert cli.main(["check", task_dir]) == 0
    assert "OK — aucune anomalie bloquante" in capsys.readouterr().out
    assert cli.main(["status", "latest"]) == 0
    assert "Configuration et trajectoire intra-entretien : terminé (1/1 entretien(s))" in capsys.readouterr().out


# --- Interface Streamlit -----------------------------------------------------------------------------------

def test_app_runs_stage5_through_the_workflow_without_key(reference):
    first, pipeline = reference
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = first["metadata"]
    at.run()
    assert not at.exception
    assert "Analyse IA désactivée" not in texts(at.warning)
    assert not [b for b in at.button if b.key == "traj_launch"]  # l'ancien lanceur par API n'est pas proposé
    assert "⏳ EN ATTENTE (workflow Claude Code)" in list(table(at, "Étape 5")["Étape 5"])
    assert f"Exécute TRACE sur le run {first['metadata']['run_id']} jusqu'à l'étape 5" in " ".join(
        c.value for c in at.code)

    [task] = first["status"]["pending"]  # réponse invalide : erreur affichée, aucun appel
    (wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME).write_text("pas du JSON", encoding="utf-8")
    at.button(key="wf_stage5").click().run()
    assert not at.exception and "réponse(s) d'agent à corriger" in texts(at.warning)
    assert list(table(at, "État")["État"]) == ["réponse à corriger"]

    (wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME).unlink()
    answer(first["status"], pipeline.agents)  # Claude Code joue le Trajectory Mapper
    at.button(key="wf_stage5").click().run()
    assert not at.exception
    assert "Étape 5 terminée (workflow Claude Code, 0 appel API)." in texts(at.success)
    results = table(at, "Étape 5")
    assert list(results["Étape 5"]) == ["✅ SUCCESS"] and list(results["Affirmations retenues"]) == [3]
    assert list(results["Appels API"]) == [0]
    assert "réponse d'agent du workflow Claude Code (0 appel API)" in texts(at.markdown)
    assert ("🟢 **5. Configuration et trajectoire intra-entretien** (IA) — terminé (1/1 entretien(s))"
            in texts(at.markdown))
    labels = [b.label for b in at.get("download_button")]
    assert all(f"Télécharger {name}" in labels for name in STAGE5_FILES)


def test_app_shows_stage5_ready_and_keeps_the_workflow_even_with_a_key(tmp_path, monkeypatch, api_guard):
    run = finish(si.make_ingested_run(tmp_path, S.FILES), Pipeline.reference().agents, "4")["metadata"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    at.run()
    assert not at.exception
    assert ("🔵 **5. Configuration et trajectoire intra-entretien** (IA) — prête, workflow Claude Code (0 appel API)"
            in texts(at.markdown))
    assert [b for b in at.button if b.key == "wf_stage5"] and not [b for b in at.button if b.key == "traj_launch"]
    at.button(key="wf_stage5").click().run()  # aucun repli vers l'API, même avec une clé
    assert not at.exception and at.session_state["last_run"]["stage5_workflow"]["api_calls"] == 0
    assert at.session_state["last_run"]["stage5_workflow"]["status"] == wf.STAGE_AWAITING
