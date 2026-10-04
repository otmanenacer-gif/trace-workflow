"""Étape 4 — contrôle sémantique (grounding) des épisodes, en mode RAPPORT SEULEMENT.

Une citation littéralement exacte peut ne pas soutenir l'interprétation qu'on lui prête (épisode E011 réel : une
routine campus / bibliothèque devenue « limitation de l'usage de Google »). Le validateur de l'étape 4 vérifie la
forme ; ce module vérifie le fond, sans rien modifier :

    épisodes de l'étape 4 (accountability_episodes.json, inchangé)
      → requête par épisode (`checker_request`) : définitions méthodologiques extraites du prompt de l'Accountability
        Episode Builder (une seule source), affirmations de l'épisode (P, S, M1…, B1…, R, X), citations exactes de
        l'épisode avec leur locuteur, texte brut des tours cités par une opération, texte brut du tour de chaque
        citation (une citation coupée de son sens : négation, récusation), et — marquée « contexte », jamais une
        preuve — la question de l'enquêteur qui précède un tour cité. JAMAIS les étiquettes, résumés ou
        descriptions de l'étape 3 (types de signaux, surface_form, summary…).
      → Episode Grounding Checker (agents/episode_grounding_checker.py), UN appel par épisode, modèle local, réponse
        mise en cache (clé : la requête exacte) : un nouveau diagnostic du même run ne rappelle pas le modèle
      → verdict de chaque affirmation DÉDUIT par TRACE (`claim_verdict`, version 1.1) de l'analyse du vérificateur
        (qui l'affirme, proposition établie, changement de sujet, effet du contexte, relation) — jamais déclaré
        directement par le modèle : equivalent / direct_paraphrase → supported ; immediate_inference → inferred ;
        stronger_than_evidence / different → not_supported ; ambiguous → ambiguous ; contradicted, ou contexte qui
        inverse le sens → contradicted ; une relation favorable devient not_supported si l'affirmation n'est pas
        énoncée par l'enquêté·e lui-même, change de sujet, ou si le contexte en limite le sens
      → verdict d'épisode déterministe (`episode_verdict`). Affirmations CENTRALES : la question pratique (P) et les
        opérations (M…) d'un épisode d'accountability ; le résumé (S) des autres statuts. Les autres (S d'un épisode
        d'accountability, B…, R, X) sont secondaires.
           contradicted  une affirmation centrale contredite ;
           unsupported   P non établie (épisode d'accountability), aucune opération établie, S non établi (autres
                         statuts), récit sur des tiers présenté comme position de l'enquêté·e, statut trop fort ;
           doubtful      une affirmation secondaire non établie ou contredite, une opération non établie à côté d'une
                         autre établie, un type d'opération hors définition, une formulation de l'enquêteur attribuée
                         à l'enquêté·e, un statut trop faible, une affirmation sans verdict ;
           supported     sinon (paraphrase et inférence immédiate admises).
      → défauts formels, déterministes, signalés comme avertissements (aucune réparation) : question pratique qui
        n'est pas une question ou qui répète le résumé, frontière qui n'est pas « A / B », chaîne « null »,
        opération appuyée aussi sur des tours de l'enquêteur
      → rapport local_runs/stage4/grounding_report.json ; comparaison facultative avec un audit manuel.

Aucune réparation, aucune décision bloquante : les sorties de l'étape 4 et la garde des étapes restent inchangées.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from agents import episode_grounding_checker as checker
from agents.base import sha256_text
from core import config
from core.accountability_candidates import excerpt
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.llm_client import LLMError, LLMSettings
from core.schemas import SPEAKER_INTERVIEWER

SPEC = checker.SPEC
GROUNDING_VERSION = "1.1"  # 1.1 : verdicts déduits de l'analyse, affirmations centrales, citations Q… normalisées
REPORT_FILENAME = "grounding_report.json"
SUPPORTED, DOUBTFUL, UNSUPPORTED, CONTRADICTED = "supported", "doubtful", "unsupported", "contradicted"
VERDICTS = (SUPPORTED, DOUBTFUL, UNSUPPORTED, CONTRADICTED)
NOT_CHECKED = "not_checked"
HOLDS = ("supported", "inferred")
NOT_ESTABLISHED = ("not_supported", "ambiguous")
RELATION_VERDICTS = {"equivalent": "supported", "direct_paraphrase": "supported", "immediate_inference": "inferred",
                     "stronger_than_evidence": "not_supported", "different": "not_supported",
                     "contradicted": "contradicted", "ambiguous": "ambiguous"}
CLAIM_LABELS = {"not_supported": "non établi par les citations", "ambiguous": "ambigu (plusieurs lectures)",
                "contradicted": "contredit par les citations", NOT_CHECKED: "non vérifié (sans verdict)"}
TURN_TEXT_RATIO = 0.9  # tour complet montré sous une citation qui en couvre moins de 90 %
# Sections du prompt de l'Accountability Episode Builder reprises telles quelles comme définitions
DEFINITION_HEADINGS = ("## Ce que « accountability » veut dire ici", "## Les trois statuts",
                       "## Opérations (`accounting_moves`)", "## Frontières (`boundary_objects`)", "## L'enquêteur")
CONTEXT_MAX_CHARS = 400
MANUAL_CLASSES = {"solide": (SUPPORTED,), "douteux": (DOUBTFUL,), "faux": (UNSUPPORTED, CONTRADICTED)}


# --- Requête : définitions, affirmations, citations brutes --------------------------------------------------

def definitions(builder_prompt: str | None = None) -> str:
    """Définitions méthodologiques : sections du prompt de l'Accountability Episode Builder, mot pour mot."""
    if builder_prompt is None:
        from core.accountability import SPEC as BUILDER
        builder_prompt = BUILDER.system_prompt
    sections = re.split(r"(?m)^(?=## )", builder_prompt)
    kept = [s.strip() for s in sections if any(s.startswith(h) for h in DEFINITION_HEADINGS)]
    if len(kept) != len(DEFINITION_HEADINGS):
        raise ValueError("Définitions introuvables dans le prompt de l'Accountability Episode Builder.")
    return "\n\n".join(kept)


def _real(value) -> str | None:
    return None if value in (None, "", "null") else value


def claims_of(episode: dict) -> list[dict]:
    """Affirmations analytiques de l'épisode, identifiées (P, S, M1…, B1…, R, X)."""
    claims = []
    if _real(episode.get("accountability_problem")):
        claims.append({"claim_id": "P", "field": "accountability_problem", "text": episode["accountability_problem"]})
    claims.append({"claim_id": "S", "field": "episode_summary", "text": episode.get("episode_summary") or ""})
    for n, move in enumerate(episode.get("accounting_moves", []), start=1):
        claims.append({"claim_id": f"M{n}", "field": "accounting_moves", "move_type": move.get("type"),
                       "text": move.get("description") or "", "turn_ids": move.get("evidence_turn_ids", [])})
    for n, boundary in enumerate(episode.get("boundary_objects", []), start=1):
        claims.append({"claim_id": f"B{n}", "field": "boundary_objects", "text": boundary})
    if _real(episode.get("student_role_reference")):
        claims.append({"claim_id": "R", "field": "student_role_reference", "text": episode["student_role_reference"]})
    if _real(episode.get("external_reference")):
        claims.append({"claim_id": "X", "field": "external_reference", "text": episode["external_reference"]})
    return claims


def quotes_of(transcript: dict, episode: dict) -> tuple[list[dict], list[dict]]:
    """(citations Q1…, contextes). Citations : celles de l'épisode, puis le texte brut des tours cités par une opération
    sans citation ; contexte : la question de l'enquêteur qui précède immédiatement un tour cité de l'enquêté·e."""
    turns = transcript["turns"]
    position = {t["turn_id"]: i for i, t in enumerate(turns)}
    quotes = []
    for e in episode.get("evidence", []):
        turn = turns[position[e["turn_id"]]] if e.get("turn_id") in position else None
        item = {"quote_id": f"Q{len(quotes) + 1}", "turn_id": e.get("turn_id"),
                "speaker": turn["speaker"] if turn else "unknown", "text": e.get("quote", ""), "kind": "citation"}
        if turn and len(item["text"].strip()) < TURN_TEXT_RATIO * len(turn["text"].strip()):  # citation coupée ?
            item["turn_text"] = excerpt(turn["text"], [item["text"]])
        quotes.append(item)
    quoted = {q["turn_id"] for q in quotes}
    for move in episode.get("accounting_moves", []):
        for turn_id in move.get("evidence_turn_ids", []):
            if turn_id in position and turn_id not in quoted:
                turn = turns[position[turn_id]]
                quotes.append({"quote_id": f"Q{len(quotes) + 1}", "turn_id": turn_id, "speaker": turn["speaker"],
                               "text": excerpt(turn["text"], []), "kind": "tour cité par une opération"})
                quoted.add(turn_id)
    contexts = []
    for turn_id in sorted(quoted, key=lambda t: position.get(t, -1)):
        index = position.get(turn_id)
        if not index or turns[index]["speaker"] == SPEAKER_INTERVIEWER:
            continue
        previous = turns[index - 1]
        if previous["speaker"] == SPEAKER_INTERVIEWER and previous["turn_id"] not in quoted:
            contexts.append({"turn_id": previous["turn_id"], "before": turn_id,
                             "text": previous["text"][:CONTEXT_MAX_CHARS]})
    return quotes, contexts


def checker_request(transcript: dict, episode: dict, builder_prompt: str | None = None) -> dict:
    """Message de l'Episode Grounding Checker pour UN épisode, et ce qu'il contient (affirmations, citations)."""
    claims = claims_of(episode)
    quotes, contexts = quotes_of(transcript, episode)
    short = lambda turn_id: (turn_id or "?").rsplit("_", 1)[-1]  # noqa: E731
    shown_claims = []
    for c in claims:
        item = {k: v for k, v in c.items() if k in ("claim_id", "field", "move_type", "text")}
        if "turn_ids" in c:  # une opération renvoie aux citations Q… de ses tours, jamais à des identifiants de tour
            item["quote_ids"] = [q["quote_id"] for q in quotes if q["turn_id"] in c["turn_ids"]]
        shown_claims.append(item)
    shown = {"episode_status": episode.get("episode_status"), "confidence": episode.get("confidence"),
             "claims": shown_claims}
    lines = [f'{q["quote_id"]} [{short(q["turn_id"])}, {q["speaker"]}{", " + q["kind"] if q["kind"] != "citation" else ""}] '
             f'« {q["text"]} »' + (f'\n    tour complet de {q["quote_id"]} : « {q["turn_text"]} »'
                                   if q.get("turn_text") else "")
             for q in quotes]
    lines += [f'contexte [{short(c["turn_id"])}, enqueteur, avant {short(c["before"])} — jamais une preuve] '
              f'« {c["text"]} »' for c in contexts]
    message = SPEC.user_template.format(
        interview_id=transcript["interview_id"], episode_id=episode.get("episode_id"),
        definitions=definitions(builder_prompt), episode_json=json.dumps(shown, ensure_ascii=False, indent=1),
        quotes="\n".join(lines), claim_ids=", ".join(c["claim_id"] for c in claims),
        move_ids=", ".join(c["claim_id"] for c in claims if c["claim_id"].startswith("M")) or "aucune")
    return {"message": message, "claims": claims, "quotes": quotes, "contexts": contexts}


# --- Défauts formels (avertissements, déterministes) -------------------------------------------------------

def formal_warnings(transcript: dict, episode: dict) -> list[str]:
    speaker = {t["turn_id"]: t["speaker"] for t in transcript["turns"]}
    warnings = []
    problem = _real(episode.get("accountability_problem"))
    if problem and not problem.rstrip().endswith("?"):
        warnings.append("PROBLEM_NOT_A_QUESTION")
    if problem and problem.strip().lower() == (episode.get("episode_summary") or "").strip().lower():
        warnings.append("PROBLEM_EQUALS_SUMMARY")
    if any(" / " not in b for b in episode.get("boundary_objects", [])):
        warnings.append("BOUNDARY_NOT_A_PAIR")
    if "null" in (episode.get("student_role_reference"), episode.get("external_reference")):
        warnings.append("NULL_STRING")
    for n, move in enumerate(episode.get("accounting_moves", []), start=1):
        if any(speaker.get(t) == SPEAKER_INTERVIEWER for t in move.get("evidence_turn_ids", [])):
            warnings.append(f"MOVE_CITES_INTERVIEWER_TURN M{n}")
    return warnings


# --- Verdicts (déterministes) -----------------------------------------------------------------------------

def claim_verdict(claim: dict) -> str:
    """Verdict d'UNE affirmation. Réponse 1.1 : déduit de l'analyse du vérificateur, jamais déclaré par lui ; réponse
    1.0 (rapport antérieur) : le verdict déclaré."""
    if "relation" not in claim:
        return claim.get("verdict", NOT_CHECKED)
    if claim.get("context_effect") == "reverses":
        return "contradicted"
    verdict = RELATION_VERDICTS.get(claim["relation"], NOT_CHECKED)
    if verdict in HOLDS and (claim.get("asserted_by") != "student" or claim.get("subject_shift")
                             or claim.get("context_effect") == "limits"):
        return "not_supported"
    return verdict


def quote_ids(raw: list[str], request: dict) -> list[str]:
    """Citations Q… : un identifiant de tour renvoyé à la place (réponse réelle fréquente) devient la ou les citations
    de ce tour ; un identifiant inconnu est gardé, marqué « ? »."""
    known = {q["quote_id"] for q in request["quotes"]}
    by_turn: dict[str, list[str]] = {}
    for q in request["quotes"]:
        for key in (q["turn_id"], (q["turn_id"] or "").rsplit("_", 1)[-1]):
            by_turn.setdefault(key, []).append(q["quote_id"])
    out = []
    for value in raw or []:
        out += [value] if value in known else by_turn.get(value, [f"{value}?"])
    return list(dict.fromkeys(out))


def episode_verdict(episode: dict, request: dict, output: dict) -> dict:
    """Verdict de l'épisode (voir l'en-tête du module). Une affirmation sans verdict n'est jamais comptée comme
    établie (doubtful)."""
    by_id = {c["claim_id"]: c for c in output.get("claims", [])}
    types = {m["move_id"]: m for m in output.get("move_types", [])}
    claims = request["claims"]
    verdict_of = {c["claim_id"]: claim_verdict(by_id[c["claim_id"]]) if c["claim_id"] in by_id else NOT_CHECKED
                  for c in claims}
    moves = [c["claim_id"] for c in claims if c["claim_id"].startswith("M")]
    is_episode = episode.get("episode_status") == "accountability_episode"
    central = (["P"] if is_episode and "P" in verdict_of else []) + moves + ([] if is_episode else ["S"])
    held = [m for m in moves if verdict_of[m] in HOLDS]
    findings = []

    def add(level: str, problem: str, claim_ids=(), reason: str = "") -> None:
        ids = sorted({q for cid in claim_ids for q in quote_ids((by_id.get(cid) or {}).get("quote_ids", []), request)})
        reasons = [" — ".join(x for x in ((by_id.get(cid) or {}).get("evidence_proposition"),
                                         (by_id.get(cid) or {}).get("reason")) if x) for cid in claim_ids]
        findings.append({"level": level, "problem": problem, "claim_ids": list(claim_ids), "quote_ids": ids,
                         "reason": reason or " | ".join(r for r in reasons if r)})

    for cid in central:
        if verdict_of[cid] == "contradicted":
            add(CONTRADICTED, f"{cid} {CLAIM_LABELS['contradicted']}", [cid])
    if is_episode and "P" in central and verdict_of["P"] in NOT_ESTABLISHED:
        add(UNSUPPORTED, f"P (question pratique) {CLAIM_LABELS[verdict_of['P']]}", ["P"])
    if is_episode and not moves:
        add(UNSUPPORTED, "épisode d'accountability sans opération", [])
    elif is_episode and not held and any(verdict_of[m] in NOT_ESTABLISHED for m in moves):
        add(UNSUPPORTED, "aucune opération établie par les citations",
            [m for m in moves if verdict_of[m] in NOT_ESTABLISHED])
    if not is_episode and verdict_of.get("S") in NOT_ESTABLISHED:
        add(UNSUPPORTED, f"résumé (S) {CLAIM_LABELS[verdict_of['S']]}", ["S"])
    if output.get("third_party_as_student"):
        add(UNSUPPORTED, "récit sur des tiers présenté comme position de l'enquêté·e", moves, output.get("notes") or "")
    if output.get("status_fit") == "too_strong":
        add(UNSUPPORTED, "statut trop fort pour le matériau", moves)
    for cid, value in verdict_of.items():
        central_handled = cid in central and (value == "contradicted" or cid == "P" and value in NOT_ESTABLISHED
                                              or cid == "S" and value in NOT_ESTABLISHED
                                              or cid in moves and not held and value in NOT_ESTABLISHED)
        if value not in HOLDS and not central_handled:
            add(DOUBTFUL, f"{cid} {CLAIM_LABELS.get(value, value)}", [cid],
                "affirmation sans verdict" if value == NOT_CHECKED else "")
    for mid in moves:
        check = types.get(mid)
        if check is None or not check.get("type_fits_definition"):
            move_type = next(c["move_type"] for c in claims if c["claim_id"] == mid)
            detail = " — ".join(x for x in ((check or {}).get("definition_requirement"),
                                           (check or {}).get("evidence_shows"), (check or {}).get("reason")) if x)
            add(DOUBTFUL, f"{mid} : type « {move_type} » hors définition", [mid], detail or "type non vérifié")
    if output.get("interviewer_framing_as_student"):
        add(DOUBTFUL, "formulation de l'enquêteur attribuée à l'enquêté·e", [], output.get("notes") or "")
    if output.get("status_fit") == "too_weak":
        add(DOUBTFUL, "statut trop faible pour le matériau", [])
    rank = {SUPPORTED: 0, DOUBTFUL: 1, UNSUPPORTED: 2, CONTRADICTED: 3}
    verdict = max((f["level"] for f in findings), key=rank.get, default=SUPPORTED)
    return {"verdict": verdict, "findings": sorted(findings, key=lambda f: -rank[f["level"]]),
            "claim_verdicts": verdict_of}


def rescore(document: dict, interview_dirs: list[Path]) -> list[dict]:
    """Verdicts recalculés, SANS appel, à partir des réponses d'un rapport existant et de l'agrégation actuelle."""
    by_interview = {r["interview_id"]: r for r in document.get("interviews", [])}
    results = []
    for directory in interview_dirs:
        loaded = read_episodes(directory)
        old = by_interview.get(loaded[0]["interview_id"]) if loaded else None
        if old is None:
            continue
        planned = {p["episode_id"]: p for p in plan(directory)}
        episodes = {e["episode_id"]: e for e in loaded[1]["episodes"]}
        rows = []
        for row in old["rows"]:
            item = planned.get(row["episode_id"])
            if item is None or row.get("checker_output") is None:
                rows.append({**row, "verdict": NOT_CHECKED, "findings": []})
                continue
            rows.append({**row, **episode_verdict(episodes[row["episode_id"]], item["request"], row["checker_output"]),
                         "quotes": [{k: q[k] for k in ("quote_id", "turn_id", "speaker", "kind")}
                                    for q in item["request"]["quotes"]]})
        results.append({**old, "rows": rows, "counts": {v: sum(r["verdict"] == v for r in rows)
                                                        for v in (*VERDICTS, NOT_CHECKED)}})
    return results


# --- Comparaison avec un audit manuel ---------------------------------------------------------------------

def compare(rows: list[dict], manual: dict[str, list[str]]) -> dict:
    """Accord entre les verdicts et un audit manuel {"solide"|"douteux"|"faux": [suffixes E001…]}."""
    expected = {eid: label for label, ids in manual.items() for eid in ids}
    lines, agree = [], 0
    for row in rows:
        short = row["episode_id"].rsplit("_", 1)[-1]
        label = expected.get(short)
        if label is None:
            continue
        ok = row["verdict"] in MANUAL_CLASSES[label]
        agree += ok
        lines.append({"episode_id": short, "manual": label, "automatic": row["verdict"], "agree": ok})
    matrix = {label: {v: sum(1 for l in lines if l["manual"] == label and l["automatic"] == v)
                      for v in (*VERDICTS, NOT_CHECKED)} for label in MANUAL_CLASSES}
    return {"compared": len(lines), "agreements": agree, "lines": lines, "matrix": matrix,
            "missing": sorted(set(expected) - {l["episode_id"] for l in lines})}


# --- Exécution (rapport seulement) ------------------------------------------------------------------------

def cache_key_fields(transcript: dict, settings: LLMSettings, message: str) -> dict:
    return {"source_sha256": transcript["source"]["sha256"], "transcript_sha256": sha256_text(message),
            **SPEC.identity(), "model": settings.model, "request_params": settings.request_params()}


async def _check(transcript: dict, episode: dict, request: dict, client, cache: AnalysisCache,
                 settings: LLMSettings) -> dict:
    key_fields = cache_key_fields(transcript, settings, request["message"])
    row = {"cache_key": compute_cache_key(key_fields), "cache_hit": False, "error": None, "output": None}
    entry = cache.load(key_fields, SPEC.output_model)
    if entry is not None:
        row.update(cache_hit=True, output=entry["output"])
        return row
    try:
        result = await client.complete_json(system_prompt=SPEC.system_prompt, user_content=request["message"],
                                            output_schema=SPEC.output_schema, response_model=SPEC.output_model,
                                            label=f"{transcript['interview_id']}/{SPEC.name}/{episode['episode_id']}")
    except LLMError as error:
        row["error"] = error.to_dict()
        return row
    row["output"] = result.data.model_dump(mode="json")
    cache.store(key_fields, row["output"], {"usage": result.usage, "attempts": result.attempts,
                                            "duration_seconds": result.duration_seconds,
                                            "request_id": result.request_id})
    row.update(usage=result.usage, duration_seconds=result.duration_seconds, request_id=result.request_id)
    return row


def read_episodes(interview_dir: Path) -> tuple[dict, dict] | None:
    analysis_dir = Path(interview_dir) / config.ANALYSIS_SUBDIR
    path = analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME
    if not path.is_file():
        return None
    transcript = json.loads((Path(interview_dir) / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
    return transcript, json.loads(path.read_text(encoding="utf-8"))


def plan(interview_dir: Path) -> list[dict]:
    """Requêtes prévues (une par épisode), sans appel : pour estimer le coût et lire les avertissements formels."""
    loaded = read_episodes(interview_dir)
    if loaded is None:
        return []
    transcript, document = loaded
    return [{"episode_id": e["episode_id"], "request": checker_request(transcript, e),
             "formal_warnings": formal_warnings(transcript, e)} for e in document["episodes"]]


async def check_interview(interview_dir: Path, client, settings: LLMSettings,
                          cache: AnalysisCache | None = None) -> dict:
    """Diagnostic d'un entretien : un appel par épisode (ou le cache), verdicts, avertissements formels. Ne modifie
    aucune sortie de l'étape 4."""
    cache = cache or AnalysisCache()
    loaded = read_episodes(interview_dir)
    if loaded is None:
        return {"interview_id": Path(interview_dir).name, "available": False, "rows": []}
    transcript, document = loaded
    planned = plan(interview_dir)
    episodes = {e["episode_id"]: e for e in document["episodes"]}
    results = list(await asyncio.gather(*(_check(transcript, episodes[p["episode_id"]], p["request"], client, cache,
                                                 settings) for p in planned)))
    rows = []
    for item, result in zip(planned, results):
        episode, request = episodes[item["episode_id"]], item["request"]
        row = {"episode_id": item["episode_id"], "episode_status": episode.get("episode_status"),
               "candidate_ids": episode.get("candidate_ids"), "formal_warnings": item["formal_warnings"],
               "cache_hit": result["cache_hit"], "error": result["error"],
               "quotes": [{k: q[k] for k in ("quote_id", "turn_id", "speaker", "kind")} for q in request["quotes"]],
               "contexts": [{k: c[k] for k in ("turn_id", "before")} for c in request["contexts"]]}
        if result["output"] is None:
            row.update(verdict=NOT_CHECKED, findings=[], claim_verdicts={}, checker_output=None)
        else:
            row.update(episode_verdict(episode, request, result["output"]), checker_output=result["output"])
        rows.append(row)
    return {"interview_id": transcript["interview_id"], "available": True,
            "stage4_validator_version": document.get("validator_version"), "rows": rows,
            "counts": {v: sum(r["verdict"] == v for r in rows) for v in (*VERDICTS, NOT_CHECKED)}}


def report(metadata: dict, results: list[dict], settings: LLMSettings, manual: dict | None = None) -> dict:
    """Rapport complet, écrit dans local_runs/stage4/grounding_report.json (fichier de diagnostic séparé)."""
    from core import local_pipeline as lp
    document = {"grounding_version": GROUNDING_VERSION, "mode": "report-only", **SPEC.identity(),
                "model": settings.model, "run_id": metadata.get("run_id"), "interviews": results}
    if manual:
        rows = [r for result in results for r in result["rows"]]
        document["manual_comparison"] = compare(rows, manual)
    write_json_atomic(lp.stage_dir(metadata, "4") / REPORT_FILENAME, document)
    return document
