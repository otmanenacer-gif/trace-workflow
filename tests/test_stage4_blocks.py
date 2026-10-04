"""Étape 4.2 — blocs de candidats bornés par la réponse : chaque bloc est un appel indépendant, mis en cache dès sa
validation ; fusion déterministe puis validation habituelle ; une troncature ou un arrêt ne fait rejouer que les blocs
manquants. Vrai runner local sur un faux Ollama : aucun modèle réel, aucun réseau.

Origine : sur OTMANE_NACER, 12 candidats en UN appel demandaient ≈ 7 200 tokens de réponse (les citations recopiées
font 77 % du JSON) pour une réserve de 6 144 : réponse tronquée, rejouée en entier, de façon reproductible.
"""

import json
from pathlib import Path

import pytest

from core import accountability, config
from core import accountability_candidates as ac
from core import local_pipeline as lp
from core.llm_client import LLMError
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests.fake_llm import (ACCOUNTABILITY, AUDITOR, INTERACTION, PRACTICE, OllamaRaw, text_response,
                            use_fake_runtime)
from tests.test_stage3_refilter_restore import assert_stage4_receives_30_22, refiltered, spy_candidate_builder  # noqa: F401

# --- Entretien synthétique à 15 candidats indépendants ---------------------------------------------------------

ID = "ENTRETIEN_QUINZE"
TASKS = ("les dissertations", "les exposés", "les fiches de lecture", "les mails", "les traductions",
         "les commentaires de texte", "les synthèses", "les mémoires", "les QCM", "les exercices de statistiques",
         "les lettres de motivation", "les bibliographies", "les études de cas", "les notes de synthèse", "les oraux")
TURNS = []
for task in TASKS:
    TURNS += [("Enquêteur", f"Et pour {task} ?"),
              ("Enquêté", f"Pour {task}, je demande à ChatGPT des pistes, mais pas d'écrire à ma place.")]
FILES = [("Entretien_quinze.txt", ("\n".join(f"{s} : {t}" for s, t in TURNS) + "\n").encode("utf-8"))]


def tid(number: int) -> str:
    return f"{ID}_T{number:04d}"


def answer_turn(index: int) -> int:
    return 2 * index + 2


PRACTICES = {"practices": [si.practice(
    summary=f"L'étudiant demande des pistes à ChatGPT pour {task}, jamais d'écrire à sa place.",
    turn_start=tid(answer_turn(i) - 1), turn_end=tid(answer_turn(i)), use_status="use", academic_task=task,
    ai_tool=["ChatGPT"], ai_action=["propose des pistes"], explicit_constraints=["pas d'écrire à ma place"],
    evidence=[{"turn_id": tid(answer_turn(i)), "quote": TURNS[answer_turn(i) - 1][1]}])
    for i, task in enumerate(TASKS)], "extraction_notes": None}
SIGNALS = {"signals": [], "reading_notes": None}
RULES = {answer_turn(i): S4.decision(
    "accountability_episode", f"L'étudiant borne l'usage de ChatGPT pour {task} : des pistes, pas l'écriture.",
    moves=[S4.move("restriction", "L'étudiant limite l'usage aux pistes.", answer_turn(i)),
           S4.move("refusal", "L'étudiant refuse que l'outil écrive à sa place.", answer_turn(i))],
    boundary=["écrire à ma place"]) for i, task in enumerate(TASKS)}
BUILDER = S4.scripted_builder(RULES)


def stage3_agents() -> dict:
    return {PRACTICE: lambda p: text_response(PRACTICES), INTERACTION: lambda p: text_response(SIGNALS),
            AUDITOR: lambda p: text_response({"assessments": []})}


def first_candidate(params: dict) -> str:
    return S4.sent_payload(params)["candidates"][0]["candidate_id"]


def stage3_run(tmp_path, monkeypatch, name="run") -> dict:
    use_fake_runtime(monkeypatch, stage3_agents())
    result = lp.run_stage("3", si.make_ingested_run(tmp_path / name, FILES))
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    return result["metadata"]


def episodes_doc(run: dict, interview_id: str = ID) -> dict:
    info = next(f for f in run["files"] if f["ingestion"]["interview_id"] == interview_id)
    path = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / config.ACCOUNTABILITY_EPISODES_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def comparable(document: dict) -> dict:
    """Le document de l'étape 4 sans ce qui dépend de l'exécution : horodatage, découpage, statut de cache, et
    empreintes des fichiers de l'étape 3 (deux runs distincts : leurs fichiers portent leur propre horodatage)."""
    skip = {"generated_at", "chunk_count", "cache_hit", "status", "source_hashes"}
    out = {k: v for k, v in document.items() if k not in skip}
    out["episodes"] = [{k: v for k, v in e.items() if k != "validated_at"} for e in document["episodes"]]
    return out


def builder_calls(ollama) -> list[str]:
    """Premier candidat de chaque appel de l'Accountability Episode Builder (le bloc demandé)."""
    return [first_candidate(c["params"]) for c in ollama.calls_for(ACCOUNTABILITY)]


# --- Découpage ---------------------------------------------------------------------------------------------------

def test_fifteen_candidates_are_split_into_blocks_each_candidate_exactly_once(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    prepared = accountability.prepare_stage4(run["files"][0]["ingestion"])
    assert prepared.candidate_count == 15 and ac.BLOCK_MAX_CANDIDATES == 4
    blocks = [r["candidate_ids"] for r in prepared.requests]
    assert [len(b) for b in blocks] == [4, 4, 4, 3]
    flat = [cid for block in blocks for cid in block]
    assert flat == [c["candidate_id"] for c in prepared.built["candidates"]]  # tous, une fois, dans l'ordre

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: BUILDER})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    calls = ollama.calls_for(ACCOUNTABILITY)
    assert len(calls) == 4
    sent = [[ID + "_" + c["candidate_id"] for c in S4.sent_payload(call["params"])["candidates"]] for call in calls]
    assert sorted(map(tuple, sent)) == sorted(map(tuple, blocks))  # chaque bloc ne reçoit que ses candidats
    for call in calls:  # « Bloc i/4 » : consigne existante du gabarit, prompt et schéma inchangés
        assert call["params"]["system"][0]["text"] == accountability.SPEC.system_prompt
        assert "Bloc " in call["params"]["messages"][0]["content"]
    document = episodes_doc(result["metadata"])
    assert document["chunk_count"] == 4 and document["candidate_count"] == document["episode_count"] == 15
    covered = [cid for e in document["episodes"] for cid in e["candidate_ids"]]
    assert sorted(covered) == sorted(flat) and len(covered) == len(set(covered))  # ni perdu, ni dupliqué


def test_components_are_never_split_across_blocks(refiltered, monkeypatch):  # noqa: F811
    run, _, _ = refiltered
    prepared = accountability.prepare_stage4(run["files"][0]["ingestion"])
    component = {c["candidate_id"]: c["component_id"] for c in prepared.built["candidates"]}
    owners = {}
    for index, request in enumerate(prepared.requests):
        for cid in request["candidate_ids"]:
            owners.setdefault(component[cid], set()).add(index)
    assert len(prepared.requests) > 1 and all(len(blocks) == 1 for blocks in owners.values())
    assert any(sum(1 for c in component.values() if c == k) > 1 for k in owners)  # au moins une composante de 2


def test_fusion_of_blocks_equals_the_single_call_format(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch, "blocks")
    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: BUILDER})
    blocks = episodes_doc(lp.run_stage("4", run)["metadata"])

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "single_cache")
    monkeypatch.setattr(ac, "BLOCK_MAX_CANDIDATES", ac.SINGLE_CALL_MAX_CANDIDATES)  # ancien découpage : un appel
    single_run = stage3_run(tmp_path, monkeypatch, "single")
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: BUILDER})
    single = episodes_doc(lp.run_stage("4", single_run)["metadata"])
    assert len(ollama.calls_for(ACCOUNTABILITY)) == 1 and single["chunk_count"] == 1 and blocks["chunk_count"] == 4

    assert list(blocks) == list(single)  # mêmes clés, même ordre : format de l'étape 4 inchangé
    assert comparable(blocks) == comparable(single)
    assert [e["episode_id"] for e in blocks["episodes"]] == [f"{ID}_E{n:03d}" for n in range(1, 16)]
    for episode in blocks["episodes"]:  # rien de perdu : citations, opérations, objets de frontière, identifiants
        assert episode["validation_status"] == "valid" and episode["usable_for_next_stages"] is True
        assert [m["type"] for m in episode["accounting_moves"]] == ["restriction", "refusal"]
        assert episode["boundary_objects"] == ["écrire à ma place"]
        assert episode["evidence"] and all(e["validation"]["valid"] for e in episode["evidence"])
        assert episode["candidate_ids"] and episode["practice_ids"]


# --- Reprise ----------------------------------------------------------------------------------------------------

def test_resume_after_ollama_stops_mid_stage_redoes_only_the_missing_blocks(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)

    def stops_after_two_blocks(params):
        if first_candidate(params) in ("C009", "C013"):
            return LLMError("OLLAMA_UNAVAILABLE", "faux Ollama arrêté pendant l'étape 4")
        return BUILDER(params)

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: stops_after_two_blocks})
    first = lp.run_stage("4", run)
    assert first["status"]["status"] == lp.STAGE_FAILED
    partial = episodes_doc(first["metadata"])
    assert partial["status"] == "PARTIAL" and partial["analysis_complete"] is False
    assert sorted(c for e in partial["episodes"] for c in e["candidate_ids"]) == [f"{ID}_C{n:03d}" for n in range(1, 9)]
    assert not lp.stage_complete("4", first["metadata"]["files"][0])
    cached = list((config.CACHE_DIR / ACCOUNTABILITY).glob("*.json"))
    assert len(cached) == 2  # les deux blocs validés sont persistés immédiatement

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: BUILDER})
    second = lp.run_stage("4", first["metadata"])
    assert second["status"]["status"] == lp.STAGE_COMPLETE
    assert sorted(builder_calls(ollama)) == ["C009", "C013"]  # seuls les blocs manquants
    fresh_tmp = tmp_path / "fresh"
    monkeypatch.setattr(config, "CACHE_DIR", fresh_tmp / "cache")
    reference_run = stage3_run(fresh_tmp, monkeypatch)
    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: BUILDER})
    reference = episodes_doc(lp.run_stage("4", reference_run)["metadata"])
    assert comparable(episodes_doc(second["metadata"])) == comparable(reference)


def test_a_truncated_block_is_replayed_alone_never_the_validated_blocks(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)

    def truncates_block_three(params):
        if first_candidate(params) == "C009":
            text = json.dumps(S4.scripted_output(params, RULES), ensure_ascii=False)
            return OllamaRaw(text[: len(text) // 2], done_reason="length")  # réserve de sortie atteinte
        return BUILDER(params)

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: truncates_block_three})
    first = lp.run_stage("4", run)
    assert first["status"]["status"] == lp.STAGE_FAILED
    calls = builder_calls(ollama)
    assert calls.count("C001") == calls.count("C005") == calls.count("C013") == 1  # blocs valides : un appel chacun
    assert calls.count("C009") > 1  # le bloc tronqué est redemandé (réserve doublée), seul
    document = episodes_doc(first["metadata"])
    assert document["status"] == "PARTIAL" and len(document["episodes"]) == 11
    manifest_chunks = json.loads((Path(first["metadata"]["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
                                  / config.ACCOUNTABILITY_MANIFEST_FILENAME).read_text(encoding="utf-8"))["chunks"]
    assert [c["status"] for c in manifest_chunks].count("FAILED") == 1

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: BUILDER})
    second = lp.run_stage("4", first["metadata"])
    assert second["status"]["status"] == lp.STAGE_COMPLETE
    assert builder_calls(ollama) == ["C009"]  # les 3 blocs validés viennent du cache
    assert episodes_doc(second["metadata"])["episode_count"] == 15


# --- Stage 3 corrigé (30 pratiques / 22 signaux) --------------------------------------------------------------

def test_stage4_blocks_receive_exactly_the_corrected_stage3(refiltered, monkeypatch):  # noqa: F811
    run, _, _ = refiltered
    received = spy_candidate_builder(monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    assert_stage4_receives_30_22(received)
    prepared = accountability.prepare_stage4(result["metadata"]["files"][0]["ingestion"])
    calls = ollama.calls_for(ACCOUNTABILITY)
    assert len(calls) == len(prepared.requests) == 3
    sent = sorted(c["candidate_id"] for call in calls for c in S4.sent_payload(call["params"])["candidates"])
    assert sent == sorted(c["candidate_id"].removeprefix(S4.INTERVIEW_ID + "_") for c in prepared.built["candidates"])
    document = episodes_doc(result["metadata"], S4.INTERVIEW_ID)
    assert document["candidates"]["summary"]["practice_count"] == 30
    assert document["candidates"]["summary"]["signal_count"] == 22


@pytest.mark.parametrize("candidates, expected", [(1, [1]), (4, [4]), (5, [4, 1]), (12, [4, 4, 4])])
def test_block_sizes(candidates, expected, tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    prepared = accountability.prepare_stage4(run["files"][0]["ingestion"])
    built = {**prepared.built, "candidates": prepared.built["candidates"][:candidates]}
    chunks = ac.plan_payload_chunks(prepared.transcript, built, prepared.speaker_warnings)
    assert [len(c) for c in chunks] == expected
