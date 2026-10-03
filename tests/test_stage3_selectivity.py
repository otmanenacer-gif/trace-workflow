"""Étape 3 : sélectivité déterministe renforcée (pratiques liées à une IAG, contradictions démontrées, normalisation de
non_use_reason) et resélection d'un run existant SANS aucun appel au modèle. Faux Ollama, aucun réseau."""

import copy
import json
from pathlib import Path

import pytest

from core import config, practice_selectivity, signal_selectivity
from core import local_pipeline as lp
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests.fake_llm import (ACCOUNTABILITY, AUDITOR, INTERACTION, PRACTICE, text_response, use_fake_runtime)

S4_ID = S4.INTERVIEW_ID


def t4(number: int) -> str:
    return f"{S4_ID}_T{number:04d}"


def contradiction(first: tuple[int, str], second: tuple[int, str], tid=t4) -> dict:
    return si.signal(turn_ids=[tid(first[0]), tid(second[0])], signal_type="cross_turn_contradiction",
                     surface_form="contradiction", description="Deux passages présentés comme contradictoires.",
                     topic="usage", cross_turn_reference="Rapprochement de deux tours.",
                     evidence=[{"turn_id": tid(first[0]), "quote": first[1]},
                               {"turn_id": tid(second[0]), "quote": second[1]}])


# S002 observé sur OTMANE_NACER : deux citations réelles, aucune contradiction
FALSE_CONTRADICTION = contradiction((8, "Je lui demande des idées de lectures"),
                                    (14, "J'aurais peur que le professeur pense que je n'ai rien fait."))
ROUTINE = si.practice(summary="L'étudiant indique faire ses plans lui-même.", turn_start=t4(5), turn_end=t4(6),
                      use_status="use", academic_task="plans", evidence=[{"turn_id": t4(6), "quote":
                                                                          "Normalement je fais mes plans moi-même."}])
PARASITE = {**copy.deepcopy(S4.STAGE3_PRACTICES["practices"][0]), "non_use_reason": "not_stated"}


def analysis_file(run: dict, name: str, interview_id: str = S4_ID) -> dict:
    info = next(f for f in run["files"] if f["ingestion"]["interview_id"] == interview_id)
    return json.loads((Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / name).read_text(encoding="utf-8"))


# --- Contradictions entre tours -----------------------------------------------------------------------------------

def test_a_contradiction_without_a_shared_object_is_set_aside():
    assert signal_selectivity.contradiction_problem(FALSE_CONTRADICTION) == signal_selectivity.REASON_CONTRADICTION_OBJECT
    odd = contradiction((8, "je m'énerve devant mon ordinateur"), (14, "Il sera clément avec moi"))
    assert signal_selectivity.contradiction_problem(odd) == signal_selectivity.REASON_CONTRADICTION_OBJECT


@pytest.mark.parametrize("first, second", [
    ("je lui ai demandé un plan", "le plan de la dissert, c'est moi qui l'ai fait, toute seule"),
    ("Normalement je fais mes plans moi-même.", "Une fois je lui ai demandé un plan"),
    ("Je ne lui fais jamais écrire mes dissertations.", "je lui ai fait écrire la conclusion"),
    ("Je vérifie toujours tout ce qu'il me donne", "je ne les vérifie pas"),
    ("Je ne l'utilise jamais en dehors des cours.", "Le week-end je lui demande toujours des recettes de cuisine."),
])
def test_demonstrable_contradictions_are_kept(first, second):
    assert signal_selectivity.contradiction_problem(contradiction((2, first), (12, second))) is None


def test_same_object_without_any_opposition_is_set_aside():
    same = contradiction((8, "Je lui demande des idées de lectures"), (16, "Je lui demande des résumés des textes."))
    assert signal_selectivity.contradiction_problem(same) == signal_selectivity.REASON_CONTRADICTION_OPPOSITION
    single = {**FALSE_CONTRADICTION, "evidence": FALSE_CONTRADICTION["evidence"][:1]}
    assert signal_selectivity.contradiction_problem(single) == signal_selectivity.REASON_CONTRADICTION_TURNS


# --- Pratiques : lien explicite avec une IAG ----------------------------------------------------------------------

ROUTINE_TEXT = ("Enquêteur : Tu utilises ChatGPT ?\n"
                "Enquêté : Oui, je lui demande des définitions. Et le soir je révise à la bibliothèque avec mes amis.\n"
                "Enquêteur : Et pour les exposés ?\n"
                "Enquêté : Pour les exposés aussi, quand je manque de temps.\n"
                "Enquêteur : Et le sport ?\n"
                "Enquêté : Je fais du foot le mercredi.\n"
                "Enquêteur : Tu l'utilises pour tes mails ?\n"
                "Enquêté : Pour mes mails oui, souvent.\n")


def routine_transcript(tmp_path) -> dict:
    run = si.make_ingested_run(tmp_path, [("Entretien_routines.txt", ROUTINE_TEXT.encode("utf-8"))])
    return json.loads((Path(run["files"][0]["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME)
                      .read_text(encoding="utf-8"))


def rp(number: int, quote: str, **fields) -> dict:
    tid = f"ENTRETIEN_ROUTINES_T{number:04d}"
    return si.practice(summary=fields.pop("summary", "Pratique."), turn_start=tid, turn_end=tid,
                       evidence=[{"turn_id": tid, "quote": quote}], **fields)


def test_routines_without_an_explicit_ai_link_are_set_aside_others_are_kept(tmp_path):
    transcript = routine_transcript(tmp_path)
    practices = [
        rp(2, "je lui demande des définitions", use_status="use"),                     # s'adresse à l'outil
        rp(2, "le soir je révise à la bibliothèque avec mes amis", use_status="use"),  # autre phrase : routine
        rp(4, "Pour les exposés aussi, quand je manque de temps.", use_status="use"),  # reprise explicite (« aussi »)
        rp(6, "Je fais du foot le mercredi.", use_status="use"),                       # routine
        rp(8, "Pour mes mails oui, souvent.", use_status="use"),                       # réponse à une question sur l'IAG
        rp(6, "Je fais du foot le mercredi.", use_status="use", ai_action=["propose un programme"]),  # champ déclaré
        rp(6, "Je fais du foot le mercredi.", use_status="non_use", non_use_reason="not_stated"),    # statut seul
    ]
    result = practice_selectivity.apply(practices, transcript)
    aside = [(p["evidence"][0]["quote"], p["set_aside_reason"]) for p in result["set_aside"]]
    assert aside == [("le soir je révise à la bibliothèque avec mes amis", practice_selectivity.REASON_NO_AI_LINK),
                     ("Je fais du foot le mercredi.", practice_selectivity.REASON_NO_AI_LINK),
                     ("Je fais du foot le mercredi.", practice_selectivity.REASON_NO_AI_LINK)]
    assert len(result["practices"]) == 4 and result["summary"]["set_aside_by_reason"] == {"NO_AI_LINK": 3}
    assert all(p["evidence"][0]["validation"]["valid"] for p in result["set_aside"])  # citations intactes, vérifiées


CAMPUS_TEXT = ("Enquêteur : Tu utilises ChatGPT pour réviser ?\n"
               "Enquêté : Oui, je lui demande des fiches.\n"
               "Enquêteur : Et tes dissertations ?\n"
               "Enquêté : Non, mes dissertations je les écris moi-même.\n"
               "Enquêteur : Et pour les plans, tu lui demandes ?\n"
               "Enquêté : Je ne veux pas qu'il réfléchisse à ma place.\n"
               "Enquêteur : Tu travailles où, d'habitude ?\n"
               "Enquêté : Je travaille surtout sur le campus, à la bibliothèque, je ne travaille pas chez moi.\n"
               "Enquêteur : Et le week-end ?\n"
               "Enquêté : Le week-end je vais à la bibliothèque municipale.\n")


def test_a_campus_routine_coded_non_use_without_any_ai_link_is_set_aside(tmp_path):
    """Cas observé sur OTMANE_NACER : routine de travail (campus, bibliothèque) codée non_use, sans outil, sans
    mention d'une IAG ni dans la citation ni dans le contexte immédiat → NO_AI_LINK. Le statut seul ne suffit pas."""
    run = si.make_ingested_run(tmp_path, [("Entretien_campus.txt", CAMPUS_TEXT.encode("utf-8"))])
    transcript = json.loads((Path(run["files"][0]["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME)
                            .read_text(encoding="utf-8"))

    def cp(number: int, quote: str, **fields) -> dict:
        tid = f"ENTRETIEN_CAMPUS_T{number:04d}"
        return si.practice(summary="Pratique.", turn_start=tid, turn_end=tid, ai_tool=[], ai_action=[],
                           evidence=[{"turn_id": tid, "quote": quote}], **fields)

    campus = cp(8, "Je travaille surtout sur le campus, à la bibliothèque", use_status="non_use",
                non_use_reason="not_stated")
    weekend = cp(10, "Le week-end je vais à la bibliothèque municipale.", use_status="non_use",
                 non_use_reason="preference")  # même avec une raison, toujours sans lien avec une IAG
    written = cp(4, "mes dissertations je les écris moi-même", use_status="non_use", non_use_reason="not_stated")
    refusal = cp(6, "Je ne veux pas qu'il réfléchisse à ma place.", use_status="refusal",
                 non_use_reason="personal_rule")
    result = practice_selectivity.apply([campus, weekend, written, refusal], transcript)
    assert [(p["evidence"][0]["quote"], p["set_aside_reason"]) for p in result["set_aside"]] == [
        (campus["evidence"][0]["quote"], practice_selectivity.REASON_NO_AI_LINK),
        (weekend["evidence"][0]["quote"], practice_selectivity.REASON_NO_AI_LINK)]
    # vrais non-usages : réponse à une question sur l'outil, ou pronom qui le désigne (« qu'il réfléchisse à ma place »)
    assert result["practices"] == [written, refusal]


def test_a_practice_with_an_invalid_quote_is_never_set_aside_the_validator_reports_it(tmp_path):
    transcript = routine_transcript(tmp_path)
    invented = rp(6, "Je joue au tennis le samedi.", use_status="use")
    result = practice_selectivity.apply([invented], transcript)
    assert result["practices"] == [invented] and not result["set_aside"]


# --- Pipeline : nouvelles sorties, puis resélection d'un run existant sans modèle --------------------------------

def stage3_responders(practices: dict, signals: dict) -> dict:
    return {PRACTICE: lambda p: text_response(practices), INTERACTION: lambda p: text_response(signals),
            AUDITOR: lambda p: text_response(S4.STAGE3_AUDIT)}


def noisy_outputs() -> tuple[dict, dict]:
    practices = copy.deepcopy(S4.STAGE3_PRACTICES)
    practices["practices"] = [PARASITE, *practices["practices"][1:], ROUTINE]
    signals = copy.deepcopy(S4.STAGE3_SIGNALS)
    signals["signals"].append(FALSE_CONTRADICTION)
    return practices, signals


def test_stage3_outputs_set_aside_noise_and_normalize_non_use_reason(tmp_path, monkeypatch):
    practices, signals = noisy_outputs()
    use_fake_runtime(monkeypatch, stage3_responders(practices, signals))
    run = lp.run_stage("3", si.make_ingested_run(tmp_path, S4.FILES))["metadata"]
    document = analysis_file(run, "practice_extractor.json")
    assert [p["summary"] for p in document["set_aside_practices"]] == [ROUTINE["summary"]]
    assert document["practices"][0]["non_use_reason"] is None  # « not_stated » parasite sur un usage
    assert not any("NON_USE_REASON_MISMATCH" in p["review_reasons"] for p in document["practices"])
    interaction = analysis_file(run, "interaction_signals.json")
    [aside] = [s for s in interaction["set_aside_signals"] if s["signal_type"] == "cross_turn_contradiction"]
    assert aside["set_aside_reason"] == signal_selectivity.REASON_CONTRADICTION_OBJECT
    assert aside["evidence"][0]["quote"] == FALSE_CONTRADICTION["evidence"][0]["quote"]  # citations non modifiées
    quotes = [e["quote"] for e in FALSE_CONTRADICTION["evidence"]]
    assert all([e["quote"] for e in s["evidence"]] != quotes for s in interaction["signals"])  # rien en aval


def test_refilter_applies_the_new_selectivity_to_an_existing_run_without_any_model_call(tmp_path, monkeypatch):
    practices, signals = noisy_outputs()
    # sorties « anciennes » : produites avant le filtre renforcé
    with monkeypatch.context() as old:
        old.setattr(practice_selectivity, "relevance_problem", lambda *a: None)
        old.setattr(practice_selectivity, "normalize_non_use_reason", lambda p: (p, None))
        old.setattr(signal_selectivity, "contradiction_problem", lambda s: None)
        use_fake_runtime(monkeypatch, {**stage3_responders(practices, signals), ACCOUNTABILITY: S4.REFERENCE_BUILDER})
        run = lp.run_until(si.make_ingested_run(tmp_path, S4.FILES), "4")["metadata"]
    before_signals = analysis_file(run, "interaction_signals.json")["signals"]
    quotes = [e["quote"] for e in FALSE_CONTRADICTION["evidence"]]
    old_id = next(s["signal_id"] for s in before_signals if [e["quote"] for e in s["evidence"]] == quotes)

    ollama = use_fake_runtime(monkeypatch, {})  # aucun répondeur : tout appel au modèle échouerait
    result = lp.refilter_stage3(run)
    assert ollama.calls == [] and result["report"]["model_calls"] == 0
    [interview] = result["report"]["interviews"]
    fates = {row["previous_id"]: row for row in interview["previous_signals"]}
    assert fates[old_id]["fate"] == "set_aside"
    assert fates[old_id]["reason"] == signal_selectivity.REASON_CONTRADICTION_OBJECT
    assert interview["practices_kept"] == interview["practices_before"] - 1
    assert [p["reason"] for p in interview["practices_set_aside"]] == [practice_selectivity.REASON_NO_AI_LINK]
    assert [(n["from"], n["to"]) for n in interview["non_use_reason_normalized"]] == [("not_stated", None)]
    assert interview["remaining_non_use_reason_mismatch"] == []
    report_path = Path(run["output_dir"]) / "local_runs" / "stage3" / lp.REFILTER_REPORT_FILENAME
    assert json.loads(report_path.read_text(encoding="utf-8"))["interviews"][0]["interview_id"] == S4_ID

    # étape 4 : périmée, rejouée normalement ; même résultat qu'une étape 3 sans bruit
    use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    stage4 = lp.run_stage("4", result["metadata"])
    assert stage4["status"]["status"] == lp.STAGE_COMPLETE and stage4["status"]["skipped"] == []
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "clean_cache")
    use_fake_runtime(monkeypatch, {**S4.stage3_responders(), ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    clean = lp.run_until(si.make_ingested_run(tmp_path / "clean", S4.FILES), "4")["metadata"]
    strip = lambda e: {k: v for k, v in e.items() if k not in ("validated_at", "generated_at")}  # noqa: E731
    episodes = [analysis_file(r, config.ACCOUNTABILITY_EPISODES_FILENAME)["episodes"]
                for r in (stage4["metadata"], clean)]
    assert [strip(e) for e in episodes[0]] == [strip(e) for e in episodes[1]]


def test_refilter_refuses_and_writes_nothing_when_a_response_is_missing_from_the_cache(tmp_path, monkeypatch):
    practices, signals = noisy_outputs()
    use_fake_runtime(monkeypatch, stage3_responders(practices, signals))
    run = lp.run_stage("3", si.make_ingested_run(tmp_path, S4.FILES))["metadata"]
    path = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / "practice_extractor.json"
    before = path.read_bytes()
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "empty_cache")
    with pytest.raises(ValueError, match="absente"):
        lp.refilter_stage3(run)
    assert path.read_bytes() == before


# --- Cas réels OTMANE_NACER (refilter) : P023 et P025 conservées, routine campus / bibliothèque écartée ---------

OTMANE_TEXT = ("Enquêteur : Tu utilises ChatGPT pour tes études ?\n"
               "Enquêté : Oui, pour réviser surtout.\n"
               "Enquêteur : Et pour tes devoirs ?\n"
               "Enquêté : Non, je vais pas lui faire faire mes devoirs à ma place.\n"
               "Enquêteur : Et pour des choses plus personnelles ?\n"
               "Enquêté : Des conseils perso, ça je demande pas, c'est pas son rôle.\n"
               "Enquêteur : Et tu travailles où, en général ?\n"
               "Enquêté : Je travaille sur le campus, je ne travaille pas chez moi.\n")


def otmane_case(tmp_path):
    run = si.make_ingested_run(tmp_path, [("Entretien_otmane_cas.txt", OTMANE_TEXT.encode("utf-8"))])
    transcript = json.loads((Path(run["files"][0]["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME)
                            .read_text(encoding="utf-8"))

    def op(number: int, quote: str, summary: str, **fields) -> dict:
        tid = f"ENTRETIEN_OTMANE_CAS_T{number:04d}"
        values = {"use_status": "refusal", "non_use_reason": "not_stated", "ai_tool": [], "ai_action": []}
        values.update(fields)
        return si.practice(summary=summary, turn_start=tid, turn_end=tid, evidence=[{"turn_id": tid, "quote": quote}],
                           **values)
    return transcript, op


def test_p023_refusal_naming_chatgpt_is_kept(tmp_path):
    transcript, op = otmane_case(tmp_path)
    p023 = op(4, "je vais pas lui faire faire mes devoirs à ma place",
              "L'étudiant ne demande pas à ChatGPT de lui faire des devoirs à sa place.")
    quote_only = op(4, "je vais pas lui faire faire mes devoirs", "Refus de délégation des devoirs.")  # « lui faire »
    result = practice_selectivity.apply([p023, quote_only], transcript)
    assert result["practices"] == [p023, quote_only] and not result["set_aside"]


def test_p025_non_use_naming_chatgpt_in_its_summary_is_kept(tmp_path):
    transcript, op = otmane_case(tmp_path)
    p025 = op(6, "Des conseils perso, ça je demande pas", use_status="non_use",
              summary="L'étudiant ne demande pas à ChatGPT de lui donner des conseils personnels.")
    result = practice_selectivity.apply([p025], transcript)
    assert result["practices"] == [p025] and not result["set_aside"]


def test_campus_routine_stays_set_aside_despite_negation_question_series_or_unrelated_fields(tmp_path):
    """Les deux voies qui la conservaient : simple négation (« je ne travaille pas chez moi ») dans une série de
    questions sur l'IAG, et `student_action_after` rempli d'une action sans lien avec l'outil."""
    transcript, op = otmane_case(tmp_path)
    campus = op(8, "Je travaille sur le campus, je ne travaille pas chez moi.", use_status="non_use",
                summary="L'étudiant travaille sur le campus plutôt que chez lui.")
    with_field = {**campus, "student_action_after": ["travaille sur le campus"],
                  "verification_or_control": ["relit ses notes"]}
    result = practice_selectivity.apply([campus, with_field], transcript)
    assert [p["set_aside_reason"] for p in result["set_aside"]] == [practice_selectivity.REASON_NO_AI_LINK] * 2
    assert not result["practices"]
