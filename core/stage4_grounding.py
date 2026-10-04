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
      → verdict d'épisode déterministe (`episode_verdict`) :
           contradicted  une affirmation est contredite par une citation ;
           unsupported   aucune opération soutenue (épisode d'accountability), résumé non soutenu (autres statuts),
                         récit sur des tiers présenté comme position de l'enquêté·e, ou statut trop fort ;
           doubtful      une autre affirmation non soutenue, un type d'opération hors définition, une formulation de
                         l'enquêteur attribuée à l'enquêté·e, un statut trop faible ;
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
GROUNDING_VERSION = "1.0"
REPORT_FILENAME = "grounding_report.json"
SUPPORTED, DOUBTFUL, UNSUPPORTED, CONTRADICTED = "supported", "doubtful", "unsupported", "contradicted"
VERDICTS = (SUPPORTED, DOUBTFUL, UNSUPPORTED, CONTRADICTED)
NOT_CHECKED = "not_checked"
HOLDS = ("supported", "inferred")
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
        if turn and turn["text"].strip() != item["text"].strip():  # tour brut : une citation coupée de son sens ?
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
    shown = {"episode_status": episode.get("episode_status"), "confidence": episode.get("confidence"),
             "claims": [{k: v for k, v in c.items() if k in ("claim_id", "field", "move_type", "text", "turn_ids")}
                        for c in claims]}
    lines = [f'{q["quote_id"]} [{q["turn_id"]}, {q["speaker"]}{", " + q["kind"] if q["kind"] != "citation" else ""}] '
             f'« {q["text"]} »' + (f'\n    tour complet de {q["quote_id"]} : « {q["turn_text"]} »'
                                   if q.get("turn_text") else "")
             for q in quotes]
    lines += [f'contexte [{c["turn_id"]}, enqueteur, avant {c["before"]} — jamais une preuve] « {c["text"]} »'
              for c in contexts]
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


# --- Verdict d'épisode (déterministe) ----------------------------------------------------------------------

def episode_verdict(episode: dict, request: dict, output: dict) -> dict:
    """Verdict de l'épisode à partir de la réponse du vérificateur. Une affirmation sans verdict compte comme non
    vérifiée (doubtful), jamais comme soutenue."""
    by_id = {c["claim_id"]: c for c in output.get("claims", [])}
    types = {m["move_id"]: m for m in output.get("move_types", [])}
    claims = request["claims"]
    verdict_of = {c["claim_id"]: (by_id.get(c["claim_id"]) or {}).get("verdict", NOT_CHECKED) for c in claims}
    moves = [c["claim_id"] for c in claims if c["claim_id"].startswith("M")]
    is_episode = episode.get("episode_status") == "accountability_episode"
    findings = []

    def add(level: str, problem: str, claim_ids=(), reason: str = "") -> None:
        quote_ids = sorted({q for cid in claim_ids for q in (by_id.get(cid) or {}).get("quote_ids", [])})
        findings.append({"level": level, "problem": problem, "claim_ids": list(claim_ids), "quote_ids": quote_ids,
                         "reason": reason or "; ".join((by_id.get(cid) or {}).get("reason", "") for cid in claim_ids)})

    for cid, value in verdict_of.items():
        if value == "contradicted":
            add(CONTRADICTED, f"{cid} contredit par les citations", [cid])
    held = [m for m in moves if verdict_of[m] in HOLDS]
    if is_episode and moves and not held:
        add(UNSUPPORTED, "aucune opération soutenue par les citations", [m for m in moves if verdict_of[m] != "contradicted"])
    if not is_episode and verdict_of.get("S") == "not_supported":
        add(UNSUPPORTED, "résumé non soutenu par les citations", ["S"])
    if output.get("third_party_as_student"):
        add(UNSUPPORTED, "récit sur des tiers présenté comme position de l'enquêté·e", moves,
            output.get("notes") or "")
    if output.get("status_fit") == "too_strong":
        add(UNSUPPORTED, "statut trop fort pour le matériau", moves)
    for cid, value in verdict_of.items():
        if value == "not_supported" and not (cid.startswith("M") and is_episode and not held) \
                and not (cid == "S" and not is_episode):
            add(DOUBTFUL, f"{cid} non soutenu par les citations", [cid])
        elif value == NOT_CHECKED:
            add(DOUBTFUL, f"{cid} non vérifié (absent de la réponse)", [cid], "affirmation sans verdict")
    for mid in moves:
        check = types.get(mid)
        if check is None or not check.get("type_fits_definition"):
            move_type = next(c["move_type"] for c in claims if c["claim_id"] == mid)
            add(DOUBTFUL, f"{mid} : type « {move_type} » hors définition", [mid],
                (check or {}).get("reason", "type non vérifié"))
    if output.get("interviewer_framing_as_student"):
        add(DOUBTFUL, "formulation de l'enquêteur attribuée à l'enquêté·e", [], output.get("notes") or "")
    if output.get("status_fit") == "too_weak":
        add(DOUBTFUL, "statut trop faible pour le matériau", [])
    rank = {SUPPORTED: 0, DOUBTFUL: 1, UNSUPPORTED: 2, CONTRADICTED: 3}
    verdict = max((f["level"] for f in findings), key=rank.get, default=SUPPORTED)
    return {"verdict": verdict, "findings": sorted(findings, key=lambda f: -rank[f["level"]]),
            "claim_verdicts": verdict_of}


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
