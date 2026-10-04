"""Étape 4 — contrôle sémantique (grounding) des épisodes, RAPPORT SEULEMENT. Faux Ollama, aucun réseau.

Cas réels OTMANE_NACER (extraits de la transcription, comme tests/test_stage3_selectivity.py) :
- E011 : routine campus / bibliothèque (T0307, T0309) devenue « limitation de l'usage de Google » ;
- E005 : récit sur d'autres étudiants (T0350–T0356) présenté comme une restriction personnelle ;
- E003 : « je remercié… Merci beaucoup » (T0116) typé `appeal_to_authorship` ;
- E001 : « dernière minute », proposé par l'enquêteur (T0042) et récusé par l'étudiant (T0043).
Ces tests vérifient ce que le vérificateur reçoit (citations brutes, jamais les étiquettes de l'étape 3), le verdict
déterministe tiré de sa réponse, le mode rapport seulement, le cache et la comparaison avec l'audit manuel. La
qualité du jugement du modèle réel se mesure par le diagnostic sur un vrai run, pas ici.
"""

import asyncio
import hashlib
import json
from pathlib import Path

from core import config, stage4_grounding as g
from core import local_pipeline as lp
from scripts import stage4_blocks_report
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests.fake_llm import ACCOUNTABILITY, GROUNDING, text_response, use_fake_runtime

ID = "OTMANE_NACER"


def t(n: int) -> str:
    return f"{ID}_T{n:04d}"


def transcript(turns: dict[int, tuple[str, str]]) -> dict:
    return {"interview_id": ID, "source": {"sha256": "0" * 64},
            "turns": [{"turn_id": t(n), "speaker": speaker, "text": text} for n, (speaker, text) in sorted(turns.items())]}


def episode(eid: str, status="accountability_episode", problem=None, summary="", moves=(), evidence=(), boundary=(),
            role=None, external=None) -> dict:
    return {"episode_id": f"{ID}_{eid}", "candidate_ids": [f"{ID}_C0{eid[-2:]}"], "episode_status": status,
            "accountability_problem": problem, "episode_summary": summary, "confidence": "high",
            "accounting_moves": [{"type": k, "description": d, "evidence_turn_ids": [t(n) for n in ns]}
                                 for k, d, ns in moves],
            "boundary_objects": list(boundary), "student_role_reference": role, "external_reference": external,
            "evidence": [{"turn_id": t(n), "quote": q} for n, q in evidence], "practice_ids": [], "signal_ids": []}


def output(claims: dict[str, tuple[str, list[str]]], types: dict[str, bool], third_party=False, framing=False,
           status_fit="consistent") -> dict:
    return {"claims": [{"claim_id": c, "verdict": v, "quote_ids": q, "reason": f"raison {c}"} for c, (v, q) in claims.items()],
            "move_types": [{"move_id": m, "type_fits_definition": ok, "reason": f"type {m}"} for m, ok in types.items()],
            "third_party_as_student": third_party, "interviewer_framing_as_student": framing,
            "status_fit": status_fit, "notes": None}


# --- E011 réel ---------------------------------------------------------------------------------------------------

T0309 = ("semaine type eh bien disons que je vais je me rends sur le campus, je je travaille Soit j'ai un cours le "
         "matin, donc je je m'y rends pour aller à ce cours. Soit je je je m'y rends avant que le cours ait lieu pour "
         "travailler à la bibliothèque. Hm hm. Je fais quelques pauses où soit je suis sur le téléphone, soit je lis "
         "quelque chose entre mes mes sessions de travail.")
E011_TRANSCRIPT = transcript({
    305: ("enquete", "non, j'utilise plus Google."),
    306: ("enqueteur", "D'accord. OK. Maintenant, j'aimerais bien comprendre un peu comment tu travailles toi."),
    307: ("enquete", "C'est-à-dire comment est-ce que je travaille ?"),
    308: ("enqueteur", "Bah une semaine type par exemple pour les moments les plus difficiles."),
    309: ("enquete", T0309)})
E011 = episode("E011", problem="jusqu'où l'outil peut-il intervenir dans le travail d'étudiant?",
               summary="L'étudiant limite l'usage de Google à son travail personnel et définit clairement le cadre de "
                       "son utilisation.",
               moves=[("restriction", "L'étudiant limite l'usage à son travail personnel.", (307, 309))],
               boundary=["cours", "bibliothèque"], role="je travaille",
               evidence=[(307, "C'est-à-dire comment est-ce que je travaille ?"),
                         (309, T0309.split(" Hm hm.")[0])])


def test_e011_request_shows_only_raw_quotes_and_the_campus_routine_cannot_become_a_tool_restriction():
    request = g.checker_request(E011_TRANSCRIPT, E011)
    message = request["message"]
    assert [c["claim_id"] for c in request["claims"]] == ["P", "S", "M1", "B1", "B2", "R"]
    assert "« C'est-à-dire comment est-ce que je travaille ? »" in message and "pour travailler à la bibliothèque" in message
    assert f"[{t(309)}, enquete]" in message
    assert f"contexte [{t(308)}, enqueteur, avant {t(309)} — jamais une preuve]" in message
    assert "tour complet de Q2" in message  # citation extraite d'un tour plus long : le tour brut est montré
    # ni l'étiquette de l'étape 3 (« c'est seulement pour mon travail »), ni le tour T0305 (« Google ») non cité
    assert "seulement pour" not in message and "Google." not in message
    # définitions : celles du prompt de l'Accountability Episode Builder, mot pour mot
    assert "`appeal_to_authorship`" in message and "## Les trois statuts" in message
    judged = output({"P": ("not_supported", []), "S": ("not_supported", ["Q1", "Q2"]),
                     "M1": ("not_supported", ["Q2"]), "B1": ("not_supported", []), "B2": ("not_supported", []),
                     "R": ("supported", ["Q1"])}, {"M1": False}, status_fit="too_strong")
    verdict = g.episode_verdict(E011, request, judged)
    assert verdict["verdict"] == g.UNSUPPORTED
    top = verdict["findings"][0]
    assert top["problem"] == "aucune opération soutenue par les citations" and top["claim_ids"] == ["M1"]
    assert top["quote_ids"] == ["Q2"]
    assert g.formal_warnings(E011_TRANSCRIPT, E011) == ["BOUNDARY_NOT_A_PAIR"]


# --- E005 réel ---------------------------------------------------------------------------------------------------

E005_TRANSCRIPT = transcript({
    349: ("enqueteur", "Est-ce que tu sais s'il y a des répercussions pour"),
    350: ("enquete", "euh bah par exemple donc pour de Redamerida dans l'English track il y a des il y a des il y a "
                     "des élèves qui ont raté leur matière c'est-à-dire"),
    351: ("enqueteur", "c'est-à-dire que ils ont pas eu les crédits"),
    352: ("enquete", "par rapport à l'utilisation de l'IA"),
    354: ("enquete", "et parce qu'elle l'avait en fait si je me souviens bien c'était un groupe de trois personnes qui "
                     "n'avaient pas fait leurs lectures et qui n'avaient pas fait leur exposé h et donc elles ont essayé "
                     "de s'y prendre à la dernière minute et elles ont utilisé l'IA sauf que c'était ridiculement mal "
                     "fait et donc le – le prof les a cramées tout de suite."),
    356: ("enquete", "Euh je sais plus si ça a été le cas pour toutes ces filles-là, mais il y en a au moins une partie "
                     "qui n'ont pas eu les crédits de ce cours.")})
E005 = episode("E005", problem="L'étudiant limite l'usage de Lia à des situations spécifiques et évalue les risques associés.",
               summary="L'étudiant limite l'usage de Lia à des situations spécifiques et évalue les risques associés.",
               moves=[("restriction", "L'étudiant limite l'usage de Lia à des situations spécifiques et évalue les "
                                      "risques associés.", (350, 351, 352, 354, 356))],
               evidence=[(350, "il y a des élèves qui ont raté leur matière"), (351, "ils ont pas eu les crédits"),
                         (354, "le – le prof les a cramées tout de suite.")])


def test_e005_a_story_about_other_students_cannot_by_itself_prove_a_personal_restriction():
    request = g.checker_request(E005_TRANSCRIPT, E005)
    assert f"[{t(351)}, enqueteur]" in request["message"]  # le locuteur brut de chaque citation est montré
    judged = output({"P": ("not_supported", []), "S": ("not_supported", []), "M1": ("not_supported", ["Q1", "Q3"])},
                    {"M1": False}, third_party=True, status_fit="too_strong")
    verdict = g.episode_verdict(E005, request, judged)
    assert verdict["verdict"] == g.UNSUPPORTED
    problems = [f["problem"] for f in verdict["findings"]]
    assert "récit sur des tiers présenté comme position de l'enquêté·e" in problems
    assert "MOVE_CITES_INTERVIEWER_TURN M1" in g.formal_warnings(E005_TRANSCRIPT, E005)
    assert {"PROBLEM_NOT_A_QUESTION", "PROBLEM_EQUALS_SUMMARY"} <= set(g.formal_warnings(E005_TRANSCRIPT, E005))


# --- E003 réel ---------------------------------------------------------------------------------------------------

T0087 = ("Bah je lui demande pas de de faire du du travail pour lequel j'ai le temps de de le faire. C'est-à-dire que "
         "si il y a quelque chose pour lequel s'il y a quelque chose que je peux faire par moi-même, par exemple H je "
         "sais pas euh écrire je sais pas j'aime bien écrire des des poèmes par exemple je pas demander à d'écrire un "
         "poème à ma place")
T0116 = ("Est-ce que je je sais plus si je lui demandé des de corriger certaines choses suite à ça ? Non. Euh après je "
         "lui ai demandé je remercié je lui ai demandé de me donner la la bibliographie enfin la liste des sources de "
         "me dresser une liste plus clair. Merci merci beaucoup Euh oui, merci beaucoup. C'est parfait comme ça.")
E003_TRANSCRIPT = transcript({86: ("enqueteur", "Où est-ce que tu mets la limite dans ton utilisation ?"),
                              87: ("enquete", T0087), 115: ("enqueteur", "Et après ?"), 116: ("enquete", T0116)})
E003 = episode("E003", summary="L'étudiant ne demande pas à Lia de faire des devoirs pour lui et remercie Lia après "
                               "avoir reçu un travail.",
               moves=[("restriction", "L'étudiant ne demande pas à Lia de faire des devoirs pour lui.", (87,)),
                      ("appeal_to_authorship", "L'étudiant remercie Lia après avoir reçu un travail.", (116,))],
               evidence=[(87, T0087), (116, T0116)])


def test_e003_thanking_the_tool_is_never_appeal_to_authorship():
    request = g.checker_request(E003_TRANSCRIPT, E003)
    assert '"move_type": "appeal_to_authorship"' in request["message"]
    assert "à qui a fait ou écrit le travail" in request["message"]  # la définition méthodologique est fournie
    judged = output({"S": ("inferred", ["Q1"]), "M1": ("inferred", ["Q1"]), "M2": ("supported", ["Q2"])},
                    {"M1": True, "M2": False})
    verdict = g.episode_verdict(E003, request, judged)
    assert verdict["verdict"] == g.DOUBTFUL  # M1 tient ; le type de M2 est hors définition
    [finding] = verdict["findings"]
    assert finding["problem"] == "M2 : type « appeal_to_authorship » hors définition" and finding["quote_ids"] == ["Q2"]


# --- E001 réel : formulation de l'enquêteur récusée -------------------------------------------------------------

T0043 = ("Soit pas pour les maths, pas forcément que je suis à la dernière minute. C'est parce que comme je suis très "
         "mauvais en maths, parfois je ne comprends pas l'exercice. Euh mais comme il faut bien que je le fasse, je "
         "demande à chat GPT de de le faire à ma place. Faute de mieux.")
E001_TRANSCRIPT = transcript({42: ("enqueteur", "D'accord. Et ça c'est parce que tu es à la dernière minute si j'ai bien compris."),
                              43: ("enquete", T0043)})
E001 = episode("E001", summary="L'étudiant limite l'usage de ChatGPT à des devoirs de mathématiques difficiles et à la "
                               "dernière minute.",
               moves=[("restriction", "L'étudiant limite l'usage à des devoirs de mathématiques difficiles et à la "
                                      "dernière minute.", (43,))],
               evidence=[(43, T0043.split("minute. ", 1)[1])])


def test_e001_the_full_turn_shows_that_the_student_rejects_the_interviewer_suggestion():
    request = g.checker_request(E001_TRANSCRIPT, E001)
    message = request["message"]
    assert "tour complet de Q1 : « Soit pas pour les maths, pas forcément que je suis à la dernière minute." in message
    assert f"contexte [{t(42)}, enqueteur, avant {t(43)} — jamais une preuve]" in message
    judged = output({"S": ("contradicted", ["Q1"]), "M1": ("contradicted", ["Q1"])}, {"M1": False}, framing=True)
    verdict = g.episode_verdict(E001, request, judged)
    assert verdict["verdict"] == g.CONTRADICTED
    assert {f["problem"] for f in verdict["findings"]} >= {"S contredit par les citations", "M1 contredit par les citations",
                                                          "formulation de l'enquêteur attribuée à l'enquêté·e"}


# --- Paraphrase admise ; absence de verdict jamais comptée comme soutien ------------------------------------------

def test_a_faithful_paraphrase_is_supported_and_a_missing_verdict_is_never_support():
    turns = transcript({333: ("enqueteur", "Est-ce que tu le délègues ?"),
                        334: ("enquete", "Bah j'ai jamais demandé à – à l'IA de faire un plan pour moi, je lui donne "
                                         "toujours les plans – que je veux qu'il remplisse.")})
    ep = episode("E012", problem="Jusqu'où l'outil peut-il intervenir dans la construction du plan ?",
                 summary="L'étudiant ne délègue jamais le plan à l'IA : il lui fournit le plan à remplir.",
                 moves=[("refusal", "L'étudiant dit ne jamais demander de plan à l'IA.", (334,))],
                 boundary=["faire le plan / remplir le plan"],
                 evidence=[(334, "j'ai jamais demandé à – à l'IA de faire un plan pour moi")])
    request = g.checker_request(turns, ep)
    paraphrase = output({"P": ("inferred", ["Q1"]), "S": ("inferred", ["Q1"]), "M1": ("supported", ["Q1"]),
                         "B1": ("inferred", ["Q1"])}, {"M1": True})
    assert g.episode_verdict(ep, request, paraphrase)["verdict"] == g.SUPPORTED
    assert g.formal_warnings(turns, ep) == []
    silent = output({"P": ("inferred", ["Q1"]), "M1": ("supported", ["Q1"])}, {"M1": True})  # S et B1 sans verdict
    verdict = g.episode_verdict(ep, request, silent)
    assert verdict["verdict"] == g.DOUBTFUL and verdict["claim_verdicts"]["S"] == g.NOT_CHECKED


# --- Comparaison avec l'audit manuel -----------------------------------------------------------------------------

MANUAL = {"solide": ["E007", "E010", "E012"], "douteux": ["E001", "E002", "E003", "E004", "E008"],
          "faux": ["E005", "E006", "E009", "E011"]}


def test_comparison_with_the_manual_audit():
    automatic = {"E001": "contradicted", "E002": "doubtful", "E003": "doubtful", "E004": "doubtful",
                 "E005": "unsupported", "E006": "unsupported", "E007": "supported", "E008": "supported",
                 "E009": "contradicted", "E010": "supported", "E011": "unsupported", "E012": "doubtful"}
    rows = [{"episode_id": f"{ID}_{k}", "verdict": v} for k, v in automatic.items()]
    comparison = g.compare(rows, MANUAL)
    assert (comparison["compared"], comparison["agreements"]) == (12, 9)
    assert [l["episode_id"] for l in comparison["lines"] if not l["agree"]] == ["E001", "E008", "E012"]
    assert comparison["matrix"]["faux"] == {"supported": 0, "doubtful": 0, "unsupported": 3, "contradicted": 1,
                                            "not_checked": 0}


# --- Bout en bout : rapport seulement, un appel par épisode, cache -------------------------------------------------

def all_supported(params: dict) -> object:
    content = params["messages"][0]["content"]
    shown = json.loads(content.split("<episode>\n", 1)[1].split("\n</episode>", 1)[0])
    claims = {c["claim_id"]: ("supported", ["Q1"]) for c in shown["claims"]}
    return text_response(output(claims, {c: True for c in claims if c.startswith("M")}))


def stage4_run(tmp_path, monkeypatch) -> dict:
    use_fake_runtime(monkeypatch, {**S4.stage3_responders(), ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    return lp.run_until(si.make_ingested_run(tmp_path, S4.FILES), "4")["metadata"]


def test_report_only_one_call_per_episode_nothing_modified_and_cached(tmp_path, monkeypatch):
    run = stage4_run(tmp_path, monkeypatch)
    interview_dir = Path(run["files"][0]["ingestion"]["output_dir"])
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in analysis_dir.glob("*.json")}
    assert lp.stage_complete("4", run["files"][0])

    ollama = use_fake_runtime(monkeypatch, {GROUNDING: all_supported})
    assert stage4_blocks_report.main(["grounding", run["output_dir"], "--solid", "E001,E002"]) == 0
    episodes = json.loads((analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME).read_text(encoding="utf-8"))
    calls = ollama.calls_for(GROUNDING)
    assert len(calls) == len(episodes["episodes"]) == 6 and ollama.calls_for(ACCOUNTABILITY) == []
    for call in calls:  # jamais d'étiquette ni de résumé de l'étape 3 dans la requête
        content = call["params"]["messages"][0]["content"]
        for practice in S4.STAGE3_PRACTICES["practices"]:
            assert practice["summary"] not in content
        for signal in S4.STAGE3_SIGNALS["signals"]:
            assert signal["description"] not in content
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in analysis_dir.glob("*.json")}
    assert after == before and lp.stage_complete("4", run["files"][0])  # rapport seulement : rien de modifié

    report = json.loads((lp.stage_dir(run, "4") / g.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["mode"] == "report-only" and report["model"] == "qwen2.5:7b"
    [interview] = report["interviews"]
    assert interview["counts"]["supported"] == 6
    assert report["manual_comparison"]["compared"] == 2

    ollama = use_fake_runtime(monkeypatch, {GROUNDING: all_supported})  # diagnostic relancé : tout vient du cache
    assert stage4_blocks_report.main(["grounding", run["output_dir"]]) == 0 and ollama.calls == []


def test_plan_mode_calls_nothing(tmp_path, monkeypatch, capsys):
    run = stage4_run(tmp_path, monkeypatch)
    ollama = use_fake_runtime(monkeypatch, {})
    assert stage4_blocks_report.main(["grounding", run["output_dir"], "--plan"]) == 0
    assert ollama.calls == [] and not (lp.stage_dir(run, "4") / g.REPORT_FILENAME).exists()
    assert "6 appel(s) à faire" in capsys.readouterr().out


def test_a_checker_failure_leaves_the_episode_not_checked(tmp_path, monkeypatch):
    run = stage4_run(tmp_path, monkeypatch)
    use_fake_runtime(monkeypatch, {GROUNDING: lambda p: text_response({"pas": "conforme"})})
    runner = lp.make_runner(lp.runtime_settings())
    result = asyncio.run(g.check_interview(Path(run["files"][0]["ingestion"]["output_dir"]), runner,
                                           lp.runtime_settings()))
    assert result["counts"]["not_checked"] == 6 and all(r["error"] for r in result["rows"])
