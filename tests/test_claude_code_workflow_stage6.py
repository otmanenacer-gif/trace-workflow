"""Étape 6 en workflow Claude Code : corpus, paquet unique, validation, garde par identité du corpus, Streamlit.

Aucun appel API : en plus des garde-fous de tests/conftest.py (transport réel interdit, réseau refusé, aucune
clé), la construction d'un client d'API (LLMClient) est interdite pendant tout le workflow (`api_guard`). Les
triplets de l'étape 5 viennent des fixtures existantes (tests/synthetic_stage6.py, produits par le vrai
orchestrateur de l'étape 5 avant que la garde ne soit levée) ; le Cross-Interview Comparator est simulé par
`S6.scripted_comparator`, dont les réponses sont écrites dans response.json, là où un sous-agent les écrirait.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import config
from core import claude_code_workflow as wf
from core import cross_interview as X
from core.analysis_cache import AnalysisCache
from tests import synthetic_stage6 as S6
from tests.fake_llm import COMPARATOR, FakeTransport, fake_settings
from tests.test_claude_code_workflow import APP, answer, packet, texts
from tests.test_claude_code_workflow_stage4 import api_guard, table  # noqa: F401

OUTPUT_NAMES = (config.CROSS_INTERVIEW_COMPARISON_FILENAME, config.CROSS_INTERVIEW_VALIDATION_FILENAME,
                config.CROSS_INTERVIEW_MANIFEST_FILENAME)


@pytest.fixture
def stage5(api_guard):
    """Triplets de l'étape 5 des fixtures existantes (ancien pipeline simulé), PUIS garde active."""
    def build(ids) -> list[tuple[str, bytes]]:
        api_guard["forbid"] = False
        uploads = S6.uploads(ids)
        api_guard["forbid"] = True
        return uploads
    return build


def comparator(plan=S6.GOOD_PLAN, mutate=None) -> dict:
    return {COMPARATOR: S6.scripted_comparator(plan, mutate=mutate)}


def finish(uploads, base: Path, agents=None) -> dict:
    first = wf.run_stage6(uploads, base_dir=base)
    assert first["status"]["status"] == wf.STAGE_AWAITING, first["status"]
    answer(first["status"], agents or comparator())
    done = wf.run_stage6(uploads, base_dir=base)
    assert done["status"]["status"] == wf.STAGE_COMPLETE, done["status"]
    return done


def outputs(result: dict) -> dict:
    out = Path(result["corpus_dir"]) / config.ANALYSIS_SUBDIR
    return {name: json.loads((out / name).read_text(encoding="utf-8")) for name in OUTPUT_NAMES}


def comparable(document: dict) -> dict:
    return {k: v for k, v in document.items() if k not in ("generated_at", "model", "cache_hit", "validated_at")}


def same_as_api(tmp_path, uploads, result, api_guard, agents=None) -> dict:
    """Mêmes fichiers, même réponse, ancien pipeline (LLM simulé) : mêmes sorties. Renvoie le document du workflow."""
    api_guard["forbid"] = False
    manifest = X.run_stage6(uploads, settings=fake_settings(), transport=FakeTransport(agents or comparator()),
                            cache=AnalysisCache(tmp_path / "cache"), base_dir=tmp_path / "api")
    api_guard["forbid"] = True
    workflow_out, api_out = outputs(result), outputs({"corpus_dir": manifest["corpus_dir"]})
    assert manifest["corpus_id"] == result["status"]["corpus_id"]
    for name in OUTPUT_NAMES[:2]:
        assert comparable(workflow_out[name]) == comparable(api_out[name]), name
    return workflow_out[config.CROSS_INTERVIEW_COMPARISON_FILENAME]


# --- Paquet unique, réponse, sorties habituelles ----------------------------------------------------------

def test_stage6_runs_without_key_on_two_interviews_from_one_exact_packet(tmp_path, stage5):
    uploads = stage5(S6.IDS_2)
    first = wf.run_stage6(uploads, base_dir=tmp_path)
    status = first["status"]
    assert status["status"] == wf.STAGE_AWAITING and status["api_calls"] == 0 and status["mode"] == "exploratory"
    assert [(r["interview_id"], r["status"]) for r in status["rows"]] == [("ENT_A", "exploitable"),
                                                                        ("ENT_B", "exploitable")]
    [task] = status["pending"]
    assert task["agent"] == COMPARATOR and task["stage"] == "6"
    corpus_dir = Path(first["corpus_dir"])
    assert corpus_dir == tmp_path / status["corpus_id"]
    assert task["task_dir"].startswith(str(corpus_dir / "workflow" / "stage6" / "tasks"))
    p = packet(wf._resolve(task["task_dir"]))
    prepared = X.prepare_stage6(uploads)
    assert p["task"]["corpus_id"] == status["corpus_id"] and p["task"]["run_id"] is None
    assert p["task"]["system_prompt"]["path"] == "prompts/cross_interview_comparator.md"
    assert p["system_prompt"] == X.SPEC.system_prompt and p["schema"] == X.SPEC.output_schema
    assert p["payload"] == prepared.request["user_message"]  # le message que recevait l'API
    assert sorted(wf.read_inputs(corpus_dir)) == sorted(uploads)  # copie octet pour octet des fichiers importés
    assert first["manifest"]["error"]["code"] == wf.AWAITING_AGENT
    assert not (corpus_dir / config.ANALYSIS_SUBDIR / config.CROSS_INTERVIEW_COMPARISON_FILENAME).exists()

    answer(status, comparator())
    report = wf.check_task(wf._resolve(task["task_dir"]))
    assert report["ok"] and report["counts"]["interviews"] == 2 and not report["blocking"]
    done = wf.run_stage6(uploads, base_dir=tmp_path)
    assert done["status"]["status"] == wf.STAGE_COMPLETE and done["corpus_dir"] == first["corpus_dir"]
    files = outputs(done)
    document, manifest = files[config.CROSS_INTERVIEW_COMPARISON_FILENAME], files[config.CROSS_INTERVIEW_MANIFEST_FILENAME]
    assert document["exploratory"] is True and document["corpus_n_usable"] == 2
    assert manifest["api_calls"] == 0 and manifest["billed_this_run"] is False and manifest["llm_called"] is True
    assert manifest["model"] == wf.WORKFLOW_MODEL and manifest["request_id"] == task["task_id"]
    assert wf.stage6_state(prepared, corpus_dir)["status"] == wf.STAGE6_COMPLETE


def test_corpus_of_seventeen_preserves_not_observed_negative_cases_and_review_as_the_api(tmp_path, stage5,
                                                                                         api_guard):
    uploads = stage5(S6.IDS_17)
    done = finish(uploads, tmp_path / "workflow")
    document = same_as_api(tmp_path, uploads, done, api_guard)
    assert document["corpus_n_usable"] == 17 and document["mode"] == "comparative"
    effort = next(p for p in document["student_role_criterion_patterns"] if p["criterion_label"] == "effort")
    assert {"ENT_B", "ENT_D"} <= set(effort["positions"]["not_observed"])  # non observé, jamais « absent »
    assert "ENT_H" in effort["positions"]["explicit_refusal"]
    assert any(n["interview_id"] == "ENT_D" for n in document["negative_cases"])  # cas négatif conservé
    verification = next(c for c in document["cross_case_claims"] if c["interview_ids"] == ["ENT_O", "ENT_P"])
    assert verification["needs_review"] is True and verification["confidence"] == "low"  # needs_review propagé


def test_invalid_triplet_is_excluded_with_its_reason(tmp_path, stage5, api_guard):
    uploads = stage5([*S6.IDS_4, "ENT_X_INVALIDE"])
    first = wf.run_stage6(uploads, base_dir=tmp_path / "workflow")
    rows = {r["interview_id"]: r for r in first["status"]["rows"]}
    assert rows["ENT_X_INVALIDE"]["status"] == "exclu" and rows["ENT_X_INVALIDE"]["reasons"]
    assert first["status"]["n_usable"] == 4 and '"ENT_X_INVALIDE"' not in packet(
        wf._resolve(first["status"]["pending"][0]["task_dir"]))["payload"]
    answer(first["status"], comparator())
    done = wf.run_stage6(uploads, base_dir=tmp_path / "workflow")
    document = same_as_api(tmp_path, uploads, done, api_guard)
    assert [e["interview_id"] for e in document["excluded_interviews"]] == ["ENT_X_INVALIDE"]


# --- Réponses refusées ou corrigées par le validateur ---------------------------------------------------------

def test_invalid_json_and_schema_are_refused_until_corrected(tmp_path, stage5):
    uploads = stage5(S6.IDS_2)
    first = wf.run_stage6(uploads, base_dir=tmp_path)
    [task] = first["status"]["pending"]
    response = wf._resolve(task["task_dir"]) / wf.RESPONSE_FILENAME
    for text, code in (("pas du JSON", "INVALID_JSON"), ('{"cross_case_claims": [{"claim_type": "x"}]}',
                                                        "SCHEMA_VALIDATION")):
        response.write_text(text, encoding="utf-8")
        report = wf.check_task(response.parent)
        assert not report["ok"] and report["blocking"][0].startswith(code)
        result = wf.run_stage6(uploads, base_dir=tmp_path)
        assert result["status"]["status"] == wf.STAGE_INVALID and result["status"]["invalid"][0]["task_id"] == \
            task["task_id"]
        assert result["manifest"]["status"] == "FAILED" and result["manifest"]["error"]["code"] == code
        assert not (Path(result["corpus_dir"]) / config.ANALYSIS_SUBDIR /
                    config.CROSS_INTERVIEW_COMPARISON_FILENAME).exists()
    response.unlink()
    answer(first["status"], comparator())  # correction
    assert wf.check_task(response.parent)["ok"]
    assert wf.run_stage6(uploads, base_dir=tmp_path)["status"]["status"] == wf.STAGE_COMPLETE


def test_invented_identifier_is_refused_by_check_and_rejected_as_by_the_api(tmp_path, stage5, api_guard):
    def invent(output, material):
        output["cross_case_claims"][0]["support"][0]["claim_ids"] = ["TC999"]
        output["cross_case_claims"][0]["support"][0]["criterion_ids"] = []
        return output
    uploads = stage5(S6.IDS_4)
    first = wf.run_stage6(uploads, base_dir=tmp_path / "workflow")
    answer(first["status"], comparator(mutate=invent))
    report = wf.check_task(wf._resolve(first["status"]["pending"][0]["task_dir"]))
    assert not report["ok"] and any(line.startswith("UNKNOWN_OBJECT_ID") for line in report["blocking"])
    done = wf.run_stage6(uploads, base_dir=tmp_path / "workflow")  # non corrigée : règle habituelle
    document = same_as_api(tmp_path, uploads, done, api_guard, comparator(mutate=invent))
    assert "UNKNOWN_OBJECT_ID" in document["cross_case_claims"][0]["review_reasons"]


def test_counts_are_recalculated_as_by_the_api(tmp_path, stage5, api_guard):
    def inflate(output, material):
        output["cross_case_claims"][0]["n_supporting_interviews"] = 99
        return output
    uploads = stage5(S6.IDS_8)
    first = wf.run_stage6(uploads, base_dir=tmp_path / "workflow")
    answer(first["status"], comparator(mutate=inflate))
    report = wf.check_task(wf._resolve(first["status"]["pending"][0]["task_dir"]))
    assert report["ok"] and any(line.startswith("COUNT_MISMATCH") for line in report["warnings"])
    done = wf.run_stage6(uploads, base_dir=tmp_path / "workflow")
    claim = same_as_api(tmp_path, uploads, done, api_guard, comparator(mutate=inflate))["cross_case_claims"][0]
    assert claim["model_n_supporting_interviews"] == 99
    assert claim["n_supporting_interviews"] == len(claim["interview_ids"]) < 99  # recalculé en entretiens distincts


# --- Garde : identité du corpus ---------------------------------------------------------------------------------

def test_guard_is_based_on_corpus_identity_not_on_time(tmp_path, stage5):
    uploads = stage5(S6.IDS_4)
    done = finish(uploads, tmp_path)
    out = Path(done["corpus_dir"]) / config.ANALYSIS_SUBDIR
    before = {name: (out / name).read_bytes() for name in OUTPUT_NAMES}

    again = wf.run_stage6(list(reversed(uploads)), base_dir=tmp_path)  # même corpus, autre ordre d'import
    assert again["status"]["status"] == wf.STAGE_COMPLETE and again["status"]["skipped"] is True
    assert again["status"]["phase"] == "déjà terminée — non rejouée" and not again["status"]["tasks"]
    assert {name: (out / name).read_bytes() for name in OUTPUT_NAMES} == before  # rien n'est réécrit

    forced = wf.run_stage6(uploads, base_dir=tmp_path, force=True)  # rejouée explicitement : réponse relue
    assert forced["status"]["status"] == wf.STAGE_COMPLETE and forced["status"]["skipped"] is False
    assert [t["state"] for t in forced["status"]["tasks"]] == [wf.TASK_ANSWERED]

    # Même corpus mais autre identité d'analyse (ici : empreinte du prompt enregistrée différente) : rejoué.
    manifest_path = out / config.CROSS_INTERVIEW_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps({**manifest, "prompt_sha256": "autre-prompt"}), encoding="utf-8")
    assert wf.stage6_state(X.prepare_stage6(uploads), Path(done["corpus_dir"]))["status"] == wf.STAGE6_STALE
    redone = wf.run_stage6(uploads, base_dir=tmp_path)
    assert redone["status"]["status"] == wf.STAGE_COMPLETE and redone["status"]["skipped"] is False


def test_adding_removing_or_modifying_an_interview_gives_a_new_corpus(tmp_path, stage5):
    uploads = stage5(S6.IDS_4)
    base = finish(uploads, tmp_path)["status"]["corpus_id"]
    added = wf.run_stage6(uploads + stage5(["ENT_E"]), base_dir=tmp_path)["status"]
    removed = wf.run_stage6(stage5(S6.IDS_4[:3]), base_dir=tmp_path)["status"]
    triplet = dict(stage5(["ENT_A"]))
    name = f"ENT_A_{config.STUDENT_TRAJECTORY_FILENAME}"
    document = json.loads(triplet[name])
    document["trajectory_claims"][0]["description"] += " (formulation révisée)"
    triplet[name] = json.dumps(document, ensure_ascii=False).encode()
    modified = wf.run_stage6(list(triplet.items()) + stage5(S6.IDS_4[1:]), base_dir=tmp_path)["status"]
    assert len({base, added["corpus_id"], removed["corpus_id"], modified["corpus_id"]}) == 4
    for status in (added, removed, modified):  # nouvelle analyse : une tâche à exécuter, rien n'est repris
        assert status["status"] == wf.STAGE_AWAITING and status["skipped"] is False and len(status["pending"]) == 1


def test_fewer_than_two_usable_interviews_blocks_without_any_task(tmp_path, stage5):
    result = wf.run_stage6(stage5(S6.IDS_2[:1]), base_dir=tmp_path)
    assert result["status"]["status"] == wf.STAGE_BLOCKED and result["status"]["task_count"] == 0
    assert "moins de deux entretiens exploitables" in result["status"]["reason"]


# --- Commandes de Claude Code ---------------------------------------------------------------------------------

def test_cli_runs_stage6_on_stage5_outputs(tmp_path, monkeypatch, capsys, stage5):
    monkeypatch.setattr(config, "CROSS_INTERVIEW_DIR", tmp_path / "cross_interview")
    spec = importlib.util.spec_from_file_location("trace_workflow_cli6",
                                                  config.PROJECT_ROOT / "scripts" / "trace_workflow.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    folder = tmp_path / "telechargements"
    stage5(S6.IDS_4)  # triplets construits avant la garde
    S6.write_corpus(folder, S6.IDS_4)

    assert cli.main(["stage6", str(folder), "--json"]) == 3
    status = json.loads(capsys.readouterr().out)
    assert status["n_usable"] == 4 and len(status["pending"]) == 1
    answer(status, comparator())
    assert cli.main(["check", status["pending"][0]["task_dir"]]) == 0
    assert "OK — aucune anomalie bloquante" in capsys.readouterr().out
    assert cli.main(["stage6", "--corpus", status["corpus_id"]]) == 0
    out = capsys.readouterr().out
    assert "étape 6 (workflow Claude Code, 0 appel API) : COMPLETE" in out and "ENT_A : exploitable" in out
    assert cli.main(["stage6", str(folder)]) == 0  # même corpus : non rejoué
    assert "phase déjà terminée — non rejouée" in capsys.readouterr().out
    assert cli.main(["stage6", str(tmp_path / "absent")]) == 2


# --- Interface Streamlit -----------------------------------------------------------------------------------

def test_app_runs_stage6_through_the_workflow_without_key(tmp_path, monkeypatch, stage5):
    monkeypatch.setattr(config, "CROSS_INTERVIEW_DIR", tmp_path / "cross_interview")
    uploads = stage5([*S6.IDS_2, "ENT_X_INVALIDE"])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert "prête, workflow Claude Code (0 appel API), sur import des sorties de l'étape 5" in texts(at.markdown)
    at.file_uploader(key="stage6_files").set_value([(n, d, "application/json") for n, d in uploads]).run()
    assert not at.exception
    assert "N exploitables : 2" in texts(at.markdown) and "ENT_X_INVALIDE exclu" in texts(at.error)
    assert [b for b in at.button if b.key == "wf_stage6"] and not [b for b in at.button if b.key == "stage6_launch"]

    at.button(key="wf_stage6").click().run()
    assert not at.exception and "tâche(s) d'agent en attente" in texts(at.warning)
    assert list(table(at, "État")["État"]) == ["en attente"]
    assert "Exécute TRACE Stage 6 sur le corpus corpus_" in " ".join(c.value for c in at.code)
    assert "⏳ EN ATTENTE (workflow Claude Code)" in texts(at.markdown)

    status = wf.read_corpus_status(X.corpus_dir_for(X.prepare_stage6(uploads)))
    response = wf._resolve(status["pending"][0]["task_dir"]) / wf.RESPONSE_FILENAME
    response.write_text("pas du JSON", encoding="utf-8")  # réponse invalide : erreur affichée, aucun appel
    at.button(key="wf_stage6").click().run()
    assert not at.exception and list(table(at, "État")["État"]) == ["réponse à corriger"]

    response.unlink()
    answer(status, comparator())  # Claude Code joue le Comparator
    at.button(key="wf_stage6").click().run()
    assert not at.exception
    assert "Étape 6 terminée (workflow Claude Code, 0 appel API)." in texts(at.success)
    assert "0 appel(s) API — réponse d'agent du workflow Claude Code" in texts(at.markdown)
    assert "Comparaison EXPLORATOIRE" in texts(at.warning)
    labels = [b.label for b in at.get("download_button")]
    assert all(f"Télécharger {name}" in labels for name in OUTPUT_NAMES)
