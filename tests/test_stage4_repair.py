"""Étape 4.3 — réparations CIBLÉES : un candidat sans disposition, un identifiant d'un candidat voisin, un épisode
d'accountability sans opération ne font jamais régénérer un bloc ; seul l'objet concerné est réparé (au plus une fois),
les blocs et réparations validés restent en cache. Vrai runner local sur un faux Ollama : aucun modèle, aucun réseau.

Reproduit, sur un entretien SYNTHÉTIQUE (12 candidats, 3 blocs de 4), les trois anomalies du vrai run OTMANE_NACER :
C011 absent de la réponse de son bloc (CANDIDATE_NOT_ADDRESSED), E005 (C005) portant le signal de C006
(SIGNAL_NOT_IN_CANDIDATES), E009 (C009) `accountability_episode` sans opération (NO_ACCOUNTING_MOVES).
"""

import json
from pathlib import Path

from core import accountability, config, stage4_repair
from core import accountability_episode_validator as episode_validator
from core import local_pipeline as lp
from core.llm_client import LLMError
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests.fake_llm import ACCOUNTABILITY, AUDITOR, INTERACTION, PRACTICE, text_response, use_fake_runtime

ID = "ENTRETIEN_DOUZE"
TASKS = ("les dissertations", "les exposés", "les fiches de lecture", "les mails", "les traductions",
         "les commentaires de texte", "les synthèses", "les mémoires", "les QCM", "les exercices de statistiques",
         "les lettres de motivation", "les bibliographies")
BOUNDARY = "mais pas d'écrire à ma place"
TURNS = []
for task in TASKS:
    TURNS += [("Enquêteur", f"Et pour {task} ?"),
              ("Enquêté", f"Pour {task}, je demande à ChatGPT des pistes, {BOUNDARY}.")]
FILES = [("Entretien_douze.txt", ("\n".join(f"{s} : {t}" for s, t in TURNS) + "\n").encode("utf-8"))]


def tid(number: int) -> str:
    return f"{ID}_T{number:04d}"


def answer(i: int) -> int:
    return 2 * i + 2


PRACTICES = {"practices": [si.practice(
    summary=f"L'étudiant demande des pistes à ChatGPT pour {task}.", turn_start=tid(answer(i) - 1),
    turn_end=tid(answer(i)), use_status="use", academic_task=task, ai_tool=["ChatGPT"],
    ai_action=["propose des pistes"], evidence=[{"turn_id": tid(answer(i)), "quote": TURNS[answer(i) - 1][1]}])
    for i, task in enumerate(TASKS)], "extraction_notes": None}
SIGNALS = {"signals": [si.signal(
    turn_ids=[tid(answer(i))], signal_type="restriction", surface_form=BOUNDARY,
    description=f"L'enquêté limite l'usage de l'outil pour {task}.",
    evidence=[{"turn_id": tid(answer(i)), "quote": BOUNDARY}]) for i, task in enumerate(TASKS)], "reading_notes": None}


def stage3_agents() -> dict:
    return {PRACTICE: lambda p: text_response(PRACTICES), INTERACTION: lambda p: text_response(SIGNALS),
            AUDITOR: lambda p: text_response({"assessments": []})}


def episode_for(candidate: dict, payload: dict, status="accountability_episode") -> dict:
    """Épisode correct d'un candidat (identifiants abrégés, comme le modèle) : sa pratique, son signal, une opération
    appuyée sur le tour de l'enquêté·e, ses citations exactes."""
    number = min(S4.turn_number(t) for t in candidate["turn_ids"] if S4.turn_number(t) % 2 == 0)
    quotes = [q for pid in candidate["practice_ids"] for q in S4.practice_quotes(payload, pid)]
    quotes += [payload["evidence_by_id"][ref] for sid in candidate["signal_ids"]
               for ref in payload["signals_by_id"][sid]["evidence"]]
    turns = sorted(candidate["turn_ids"], key=S4.turn_number)
    episode = {"candidate_ids": [candidate["candidate_id"]], "turn_start": turns[0], "turn_end": turns[-1],
               "practice_ids": candidate["practice_ids"], "signal_ids": candidate["signal_ids"],
               "episode_status": status, "accountability_problem": "Jusqu'où l'outil peut-il intervenir ?",
               "accounting_moves": [{"type": "restriction", "description": "L'étudiant limite l'usage aux pistes.",
                                     "evidence_turn_ids": [f"T{number:04d}"]}],
               "boundary_objects": ["écrire à ma place"], "student_role_reference": None, "external_reference": None,
               "episode_summary": "L'étudiant borne l'usage de ChatGPT : des pistes, pas l'écriture.",
               "confidence": "high", "needs_review": False,
               "evidence": [{"turn_id": q["turn_id"], "quote": q["quote"]} for q in quotes]}
    if status == "ordinary_practice":
        episode.update(accountability_problem=None, accounting_moves=[], boundary_objects=[],
                       episode_summary="Aucune restriction, justification ou évaluation explicite n'est relevée dans "
                                       "ce passage.")
    return episode


def is_repair(params: dict) -> bool:
    return stage4_repair.REPAIR_NOTE.strip() in params["messages"][0]["content"]


def candidates_of(params: dict) -> list[str]:
    return [c["candidate_id"] for c in S4.sent_payload(params)["candidates"]]


def faulty_builder(*, c005_cites_c006=False, repairs=None):
    """Blocs : C005 porte aussi le signal de C006 (et, au besoin, une citation de C006), C009 est un épisode
    d'accountability sans opération, C011 est omis. Réparations : `repairs[candidat](params)` ou une réponse correcte
    (C009 reclassé en pratique ordinaire : aucune opération inventée)."""
    repairs = repairs or {}

    def respond(params):
        payload = S4.sent_payload(params)
        by_id = {c["candidate_id"]: c for c in payload["candidates"]}
        if is_repair(params):
            [cid] = by_id
            if cid in repairs:
                return repairs[cid](params)
            status = "ordinary_practice" if cid == "C009" else "accountability_episode"
            return text_response({"episodes": [episode_for(by_id[cid], payload, status)], "builder_notes": None})
        episodes = []
        for cid, candidate in by_id.items():
            if cid == "C011":
                continue
            episode = episode_for(candidate, payload)
            if cid == "C005":
                episode["signal_ids"] = episode["signal_ids"] + by_id["C006"]["signal_ids"]
                if c005_cites_c006:
                    episode["evidence"].append({"turn_id": "T0012", "quote": BOUNDARY})
                    episode["turn_end"] = "T0012"
            if cid == "C009":
                episode["accounting_moves"] = []
            episodes.append(episode)
        return text_response({"episodes": episodes, "builder_notes": None})
    return respond


def stage3_run(tmp_path, monkeypatch) -> dict:
    use_fake_runtime(monkeypatch, stage3_agents())
    result = lp.run_stage("3", si.make_ingested_run(tmp_path, FILES))
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    return result["metadata"]


def analysis_file(run: dict, name: str) -> dict:
    path = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def by_candidate(document: dict) -> dict:
    return {e["candidate_ids"][0].removeprefix(ID + "_"): e for e in document["episodes"]}


def calls_summary(ollama) -> list[tuple[str, tuple]]:
    return [("réparation" if is_repair(c["params"]) else "bloc", tuple(candidates_of(c["params"])))
            for c in ollama.calls_for(ACCOUNTABILITY)]


def assert_invariant(document: dict) -> None:
    candidates = {c["candidate_id"]: c for c in document["candidates"]["candidates"]}
    for episode in document["episodes"]:
        practices = {p for c in episode["candidate_ids"] for p in candidates[c]["practice_ids"]}
        signals = {s for c in episode["candidate_ids"] for s in candidates[c]["signal_ids"]}
        assert set(episode["practice_ids"]) <= practices and set(episode["signal_ids"]) <= signals, episode["episode_id"]


BLOCKS = [("bloc", ("C001", "C002", "C003", "C004")), ("bloc", ("C005", "C006", "C007", "C008")),
          ("bloc", ("C009", "C010", "C011", "C012"))]


# --- 1 + 2 + 3 + 4 : réparations ciblées, sans recalcul des autres épisodes ---------------------------------------

def test_missing_candidate_foreign_signal_and_moveless_episode_are_repaired_alone(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder()})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    # 3 blocs, puis UNE réparation pour C009 et UNE pour C011, chacune sur ce seul candidat ; C005 : aucun appel
    assert calls_summary(ollama) == BLOCKS + [("réparation", ("C009",)), ("réparation", ("C011",))]

    document = analysis_file(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert document["status"] == "SUCCESS" and document["analysis_complete"] is True
    episodes = by_candidate(document)
    assert sorted(episodes) == [f"C{n:03d}" for n in range(1, 13)]  # chaque candidat : exactement une disposition
    assert document["episode_count"] == document["usable_episode_count"] == 12
    # 1. C011 : réparé seul, disposition explicite
    assert episodes["C011"]["episode_status"] == "accountability_episode"
    assert episodes["C011"]["trace_repair"]["kind"] == stage4_repair.KIND_MISSING
    # 2. C005 : le signal de C006, sans appui dans l'épisode, est retiré et noté — aucune contamination
    c006_signals = [s for s in episodes["C006"]["signal_ids"]]
    assert not set(c006_signals) & set(episodes["C005"]["signal_ids"])
    assert episodes["C005"]["removed_foreign_ids"] == c006_signals
    assert "FOREIGN_ID_REMOVED" in episodes["C005"]["review_reasons"]
    assert episodes["C005"]["validation_status"] == "valid" and "trace_repair" not in episodes["C005"]
    assert_invariant(document)
    # 3. C009 : reclassé (pratique ordinaire), aucune opération inventée
    assert episodes["C009"]["episode_status"] == "ordinary_practice" and episodes["C009"]["accounting_moves"] == []
    assert "NO_ACCOUNTING_MOVES" in episodes["C009"]["trace_repair"]["problems"][0]
    # 4. les autres épisodes sont ceux des blocs, jamais recalculés
    for cid in ("C001", "C002", "C003", "C004", "C006", "C007", "C008", "C010", "C012"):
        assert "trace_repair" not in episodes[cid] and episodes[cid]["validation_status"] == "valid"
    manifest = analysis_file(result["metadata"], config.ACCOUNTABILITY_MANIFEST_FILENAME)
    assert [(r["kind"], r["candidate_ids"], r["outcome"]) for r in manifest["repairs"]] == [
        ("episode", [f"{ID}_C009"], "repaired"), ("missing_candidate", [f"{ID}_C011"], "repaired")]
    validation = analysis_file(result["metadata"], config.ACCOUNTABILITY_VALIDATION_FILENAME)
    assert validation["error_count"] == 0 and validation["unaddressed_candidate_ids"] == []


def test_repair_messages_carry_only_the_concerned_candidate_and_the_unchanged_prompt(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder()})
    lp.run_stage("4", run)
    repairs = [c for c in ollama.calls_for(ACCOUNTABILITY) if is_repair(c["params"])]
    for call in repairs:
        params = call["params"]
        assert params["system"][0]["text"] == accountability.SPEC.system_prompt  # prompt système inchangé
        assert call["payload"]["format"] == accountability.SPEC.output_schema       # schéma inchangé
        payload = S4.sent_payload(params)
        assert len(payload["candidates"]) == 1 and len(payload["practices_by_id"]) == 1
    c009 = next(c["params"]["messages"][0]["content"] for c in repairs if candidates_of(c["params"]) == ["C009"])
    assert "NO_ACCOUNTING_MOVES" in c009 and "<previous_episodes>" in c009 and "n'en invente pas" in c009
    c011 = next(c["params"]["messages"][0]["content"] for c in repairs if candidates_of(c["params"]) == ["C011"])
    assert "CANDIDATE_NOT_ADDRESSED" in c011 and "<previous_episodes>" not in c011


def test_a_foreign_signal_the_episode_relies_on_is_repaired_never_silently_dropped(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder(c005_cites_c006=True)})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    repairs = [k for k in calls_summary(ollama) if k[0] == "réparation"]
    assert ("réparation", ("C005",)) in repairs  # la citation du tour de C006 appuie l'épisode : réparation ciblée
    document = analysis_file(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME)
    c005 = by_candidate(document)["C005"]
    assert "SIGNAL_NOT_IN_CANDIDATES" in c005["trace_repair"]["problems"][0]
    assert "removed_foreign_ids" not in c005 and {e["turn_id"] for e in c005["evidence"]} == {tid(10)}
    assert_invariant(document)


def test_a_failed_repair_keeps_the_original_rejected_and_an_unaddressed_candidate_blocks_completion(tmp_path,
                                                                                                    monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    still_wrong = {
        "C009": lambda p: text_response({"episodes": [{**episode_for(S4.sent_payload(p)["candidates"][0],
                                                                     S4.sent_payload(p)), "accounting_moves": []}],
                                         "builder_notes": None}),
        "C011": lambda p: text_response({"episodes": [], "builder_notes": None}),
    }
    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder(repairs=still_wrong)})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_FAILED  # étape 4 non terminée : un candidat sans disposition
    document = analysis_file(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert document["status"] == "PARTIAL" and document["analysis_complete"] is False
    episodes = by_candidate(document)
    assert "C011" not in episodes
    assert episodes["C009"]["validation_status"] == "rejected" and "trace_repair" not in episodes["C009"]
    validation = analysis_file(result["metadata"], config.ACCOUNTABILITY_VALIDATION_FILENAME)
    [missing] = [i for i in validation["issues"] if i["code"] == "CANDIDATE_NOT_ADDRESSED"]
    assert missing["severity"] == episode_validator.ERROR and missing["object_id"] == f"{ID}_C011"
    manifest = analysis_file(result["metadata"], config.ACCOUNTABILITY_MANIFEST_FILENAME)
    assert "sans disposition" in manifest["error"]["message"]
    assert [r["outcome"] for r in manifest["repairs"]] == ["unresolved", "unresolved"]


# --- 5 : reprise après interruption ------------------------------------------------------------------------------

def test_resume_after_interruption_redoes_only_the_missing_repair(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    stopped = {"C011": lambda p: LLMError("OLLAMA_UNAVAILABLE", "faux Ollama arrêté pendant la réparation")}
    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder(repairs=stopped)})
    first = lp.run_stage("4", run)
    assert first["status"]["status"] == lp.STAGE_FAILED
    manifest = analysis_file(first["metadata"], config.ACCOUNTABILITY_MANIFEST_FILENAME)
    assert [(r["candidate_ids"], r["outcome"]) for r in manifest["repairs"]] == [
        ([f"{ID}_C009"], "repaired"), ([f"{ID}_C011"], "failed")]

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder()})
    second = lp.run_stage("4", first["metadata"])
    assert second["status"]["status"] == lp.STAGE_COMPLETE
    assert calls_summary(ollama) == [("réparation", ("C011",))]  # blocs et réparation de C009 : cache
    document = analysis_file(second["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert document["episode_count"] == 12 and document["analysis_complete"] is True

    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder()})
    assert lp.run_stage("4", second["metadata"])["status"]["skipped"] == [ID] and ollama.calls == []


# --- 6 : citations exactes intactes ; informations sans appel ---------------------------------------------------

def test_exact_quotes_of_the_episodes_are_kept_verbatim(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    builder, answered = faulty_builder(), {}

    def recording(params):  # réponses des BLOCS, telles que le modèle les a données
        reply = builder(params)
        if not is_repair(params):
            for episode in json.loads(reply.text)["episodes"]:
                answered[episode["candidate_ids"][0]] = [(f"{ID}_{e['turn_id']}", e["quote"]) for e in episode["evidence"]]
        return reply

    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: recording})
    result = lp.run_stage("4", run)
    document = analysis_file(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME)
    for cid, episode in by_candidate(document).items():
        if "trace_repair" not in episode:  # épisode d'un bloc : citations reprises caractère pour caractère
            assert [(e["turn_id"], e["quote"]) for e in episode["evidence"]] == answered[cid], cid
    quotes = [e for episode in document["episodes"] for e in episode["evidence"]]
    assert len(quotes) == 24 and all(e["validation"]["valid"] for e in quotes)  # 12 épisodes × 2 citations exactes
    transcript = {t["turn_id"]: t["text"] for t in json.loads(
        (Path(result["metadata"]["files"][0]["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME)
        .read_text(encoding="utf-8"))["turns"]}
    assert all(e["quote"] in transcript[e["turn_id"]] for e in quotes)  # mot pour mot, jamais retouchées
    validation = analysis_file(result["metadata"], config.ACCOUNTABILITY_VALIDATION_FILENAME)
    assert validation["valid_evidence_count"] == validation["evidence_count"] == 24
    assert validation["invalid_evidence_count"] == 0


def test_speaker_warning_stays_informative_and_never_triggers_a_call(tmp_path, monkeypatch):
    use_fake_runtime(monkeypatch, S4.stage3_responders())
    run = lp.run_stage("3", si.make_ingested_run(tmp_path, S4.FILES))["metadata"]
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    validation = json.loads((Path(result["metadata"]["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
                             / config.ACCOUNTABILITY_VALIDATION_FILENAME).read_text(encoding="utf-8"))
    assert any(i["code"] == "SPEAKER_WARNING_PROPAGATED" and i["severity"] == "info" for i in validation["issues"])
    assert validation["repairs"] == [] and len(ollama.calls_for(ACCOUNTABILITY)) == 2  # les deux blocs, rien d'autre


def test_a_stage4_validated_by_an_older_validator_is_replayed_from_the_cache(tmp_path, monkeypatch):
    run = stage3_run(tmp_path, monkeypatch)
    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder()})
    run = lp.run_stage("4", run)["metadata"]
    path = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / config.ACCOUNTABILITY_EPISODES_FILENAME
    document = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**document, "validator_version": "1.1"}, ensure_ascii=False), encoding="utf-8")
    state = lp.stage_state("4", run["files"][0])
    assert state["status"] == "STALE" and "1.1" in state["reasons"][0]
    assert lp.stage_state("3", run["files"][0])["status"] == "COMPLETE"  # l'étape 3 n'est pas concernée
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: faulty_builder()})
    result = lp.run_stage("4", run)
    assert result["status"]["status"] == lp.STAGE_COMPLETE and ollama.calls == []  # blocs et réparations : cache
    assert lp.stage_complete("4", result["metadata"]["files"][0])
