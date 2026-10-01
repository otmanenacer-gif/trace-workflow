"""Étape 4 — validation DÉTERMINISTE des épisodes produits par l'Accountability Episode Builder.

Aucun LLM. Pour chaque épisode :
- identifiants : practice_ids, signal_ids, candidate_ids existent ; turn_start / turn_end existent et
  sont ordonnés ; chaque evidence_turn_id d'une opération existe ;
- citations : sous-chaîne EXACTE du tour cité (core.evidence_validator, même règle qu'à l'étape 3) ;
- voix de l'enquêté·e : un `accountability_episode` doit avoir au moins une citation valide d'un tour
  de l'enquêté·e (ou d'un tour signalé par l'audit des locuteurs, alors à revoir), et chacune de ses
  opérations doit s'appuyer sur un tel tour : une question de l'enquêteur ne fait pas une position ;
- matériau : un `accountability_episode` sans matériau substantiel (signal déclencheur, frontière
  explicite, contradiction, usage / non-usage) est rejeté ; sans opération, il est rejeté ;
- `ordinary_practice` avec plusieurs opérations ou des frontières : à revoir ;
- `uncertain` : `needs_review` toujours vrai ;
- avertissements de locuteur : propagés à l'épisode (`speaker_warnings`) par TRACE, sans confiance
  dans le modèle ; l'épisode est alors à revoir ;
- vocabulaire psychologisant ou d'intention (`FORBIDDEN_TERMS`) dans les champs rédigés, sauf si
  l'enquêté·e l'emploie lui-même ou elle-même dans une citation associée (épisode, pratiques, signaux ;
  tours de l'enquêté·e seulement) ; un terme présent seulement dans une question de l'enquêteur est en
  outre signalé `INTERVIEWER_TERM_ATTRIBUTED` ; « justification » seulement si une citation donne
  réellement une raison ;
- fusions : pratiques de tâches ou de domaines différents sans lien, pratiques situées entre les deux
  tours d'une contradiction, matériau absent des candidats cités ; étape 4.1 : les candidats réunis dans
  un épisode doivent former un graphe CONNEXE de relations explicites (core.accountability_candidates.
  candidate_links) — sinon erreur `DISCONNECTED_MERGE` : l'épisode est rejeté et
  `usable_for_next_stages: false` (il reste dans le fichier, à revoir) ;
- couverture : chaque candidat est traité par un épisode (sinon `CANDIDATE_NOT_ADDRESSED`).

Rien n'est réécrit : l'épisode garde le texte du modèle ; TRACE ajoute `validation`, `needs_review`,
`review_reasons` et `speaker_warnings`. Erreur → épisode `rejected` (conservé, exclu des comptes).
"""

from __future__ import annotations

import re

from core import evidence_validator, interpretation_guard
from core.accountability_candidates import TRIGGER_SIGNAL_TYPES, connected_groups, normalize_task
from core.schemas import SPEAKER_INTERVIEWER

VALIDATOR_VERSION = "1.1"  # 1.1 : fusions déconnectées, vocabulaire élargi et sensible au locuteur

ERROR, WARNING, INFO = "error", "warning", "info"

ISSUE_CODES = {
    # identifiants et citations
    "UNKNOWN_PRACTICE_ID": (ERROR, "practice_id inexistant dans les sorties de l'étape 3."),
    "UNKNOWN_SIGNAL_ID": (ERROR, "signal_id inexistant dans les sorties de l'étape 3."),
    "UNKNOWN_CANDIDATE_ID": (ERROR, "candidate_id inexistant."),
    "UNKNOWN_RANGE_TURN": (ERROR, "turn_start ou turn_end inexistant dans cet entretien."),
    "FOREIGN_INTERVIEW_TURN": (ERROR, "turn_id appartenant à un autre entretien."),
    "INVALID_TURN_RANGE": (ERROR, "Intervalle inversé : turn_start est postérieur à turn_end."),
    "UNKNOWN_TURN_ID": (ERROR, "turn_id inexistant dans cet entretien."),
    "UNKNOWN_MOVE_TURN": (ERROR, "Tour d'une opération (evidence_turn_ids) inexistant."),
    "EMPTY_QUOTE": (ERROR, "Citation vide."),
    "QUOTE_NOT_FOUND": (ERROR, "Citation absente du texte du tour cité."),
    "QUOTE_NOT_EXACT": (ERROR, "Citation proche mais pas littérale."),
    "NO_VALID_EVIDENCE": (ERROR, "Aucune citation valide pour cet épisode."),
    # voix de l'enquêté·e et matériau
    "NO_INTERVIEWEE_EVIDENCE": (ERROR, "Épisode d'accountability sans aucune citation valide d'un tour de l'enquêté·e."),
    "NO_INTERVIEWEE_EVIDENCE_NON_EPISODE": (WARNING, "Aucune citation valide d'un tour de l'enquêté·e."),
    "MOVE_WITHOUT_INTERVIEWEE_TURN": (ERROR, "Opération appuyée seulement sur des tours de l'enquêteur."),
    "NO_ACCOUNTING_MOVES": (ERROR, "Épisode d'accountability sans aucune opération (accounting_moves vide)."),
    "INSUFFICIENT_MATERIAL": (ERROR, "Épisode d'accountability sans matériau substantiel (ni signal déclencheur, "
                                     "ni frontière explicite, ni contradiction, ni usage / non-usage)."),
    "ORDINARY_PRACTICE_WITH_ACCOUNTING": (WARNING, "Pratique ordinaire dotée de plusieurs opérations ou de frontières."),
    "BOUNDARY_WITHOUT_ACCOUNTABILITY": (WARNING, "Frontières (boundary_objects) hors d'un épisode d'accountability."),
    "MOVE_TURN_OUTSIDE_MATERIAL": (WARNING, "Opération appuyée sur un tour absent du matériau de l'épisode."),
    "EVIDENCE_OUTSIDE_RANGE": (WARNING, "Citation située hors de l'intervalle turn_start–turn_end."),
    "PRACTICE_NOT_IN_CANDIDATES": (WARNING, "Pratique absente des candidats cités par l'épisode."),
    "SIGNAL_NOT_IN_CANDIDATES": (WARNING, "Signal absent des candidats cités par l'épisode."),
    # fusions
    "MERGED_DIFFERENT_TASKS": (WARNING, "Épisode réunissant des pratiques de tâches différentes sans candidat commun."),
    "MERGED_DIFFERENT_DOMAINS": (WARNING, "Épisode réunissant une pratique d'études et une pratique personnelle sans candidat commun."),
    "INTERVENING_PRACTICE_MERGED": (WARNING, "Pratique située entre les deux tours d'une contradiction, sans lien avec eux."),
    "DISCONNECTED_MERGE": (ERROR, "Épisode réunissant des candidats sans relation explicite (pratique ou signal partagé, "
                                  "mêmes tours et même tâche, contradiction) : composantes déconnectées."),
    # vocabulaire
    "INTERPRETIVE_VOCABULARY": (WARNING, "Vocabulaire psychologisant ou d'intention dans un champ rédigé."),
    "UNSUPPORTED_JUSTIFICATION": (WARNING, "« Justification » sans raison explicite dans les citations."),
    "INTERVIEWER_TERM_ATTRIBUTED": (WARNING, "Terme proposé par l'enquêteur, non repris par l'enquêté·e, employé dans "
                                             "la description de l'épisode."),
    # statut, locuteur, couverture
    "UNCERTAIN_NOT_FLAGGED": (INFO, "Épisode incertain non marqué à revoir : marqué par TRACE."),
    "UNCERTAIN_CONFIDENCE": (INFO, "Épisode incertain avec une confiance différente de « low »."),
    "MODEL_FLAGGED_REVIEW": (INFO, "Le modèle demande une vérification humaine."),
    "SPEAKER_WARNING_PROPAGATED": (INFO, "L'épisode repose sur un tour dont l'attribution du locuteur est douteuse."),
    "CANDIDATE_IN_SEVERAL_EPISODES": (WARNING, "Candidat traité par plusieurs épisodes."),
    "CANDIDATE_NOT_ADDRESSED": (WARNING, "Candidat traité par aucun épisode."),
    "PAYLOAD_OVER_THRESHOLD": (WARNING, "Représentation envoyée au-delà du seuil d'un appel unique."),
    "STAGE3_INCOMPLETE": (WARNING, "Sorties de l'étape 3 incomplètes : analyse de l'étape 4 incomplète."),
}

# Vocabulaire interdit dans les champs rédigés (texte replié). Signalé sauf emploi par l'enquêté·e.
FORBIDDEN_TERMS = {
    "stratégie (défensive, de légitimation…)": r"strateg\w*",
    "rationalisation": r"rationalis\w*",
    "manipulation": r"manipul\w*",
    "mauvaise foi": r"mauvaise foi",
    "honte": r"hont\w*",
    "peur / crainte": r"peurs?\b|craint\w*",
    "culpabilité / coupable": r"culpabil\w*|coupables?\b",
    "anxiété / angoisse / inquiétude": r"anxi\w*|angoiss\w*|inquiet\w*",
    "embarras": r"embarras\w*",
    "intention / motivation": r"intention\w*|motivation\w*",
    "identité menacée / protection identitaire": r"identit\w*",
    "image de soi": r"image de soi",
    "volonté de légitimer / légitimation / légitimité": r"legitim\w*",
    "tentative de se justifier": r"(?:tentative|tente|cherche|essaie|volonte) (?:de |a )?(?:se |s )?justifi\w*",
    "se justifier": r"se justifi\w*",
    "défensif / se défendre": r"defensi\w*|se defend\w*",
    "se protéger": r"se proteg\w*|proteger son\w*|protection\w*",
    "motivation cachée / implicite": r"motivations? (?:cachee|implicite|profonde)s?|inconsci\w*",
    "chercher à paraître": r"(?:cherche|veut|souhaite|tente) (?:a |de )?(?:paraitre|se montrer|se presenter|preserver)",
    "calcul": r"\bcalcul\w* (?:strategique|interesse)\w*|par calcul",
    "dissimulation": r"dissimul\w*",
    "malaise / gêne": r"malaise\w*|\bgene(?:e|s|es)?\b",
}
_FORBIDDEN = [(label, re.compile(r"\b" + pattern)) for label, pattern in FORBIDDEN_TERMS.items()]
_JUSTIFICATION = re.compile(r"\bjustifi\w*")
_REASON_MARKERS = re.compile(
    r"\b(?:parce que|parce qu|car|puisque|puisqu|vu que|comme ca|pour que|pour qu|sinon|a cause|grace a|"
    r"c est pour|du coup|donc|j aurais peur|j ai peur|pour ne pas|pour pas)\b")

AUTHORED_FIELDS = ("accountability_problem", "episode_summary", "student_role_reference", "external_reference",
                   "boundary_objects", "accounting_moves.description")
MAX_ORDINARY_MOVES = 1


def make_issue(code: str, **context) -> dict:
    severity, message = ISSUE_CODES[code]
    issue = {"code": code, "severity": severity, "message": message}
    issue.update({k: v for k, v in context.items() if v is not None})
    return issue


def _authored(episode: dict) -> dict[str, str]:
    texts = {}
    for name in ("accountability_problem", "episode_summary", "student_role_reference", "external_reference"):
        if isinstance(episode.get(name), str) and episode[name]:
            texts[name] = episode[name]
    if episode.get("boundary_objects"):
        texts["boundary_objects"] = " | ".join(b for b in episode["boundary_objects"] if isinstance(b, str))
    moves = [m.get("description") or "" for m in episode.get("accounting_moves", [])]
    if any(moves):
        texts["accounting_moves.description"] = " | ".join(moves)
    return texts


def scan_vocabulary(episode: dict, interviewee_quotes: list[str], interviewer_texts: list[str] = ()) -> list[dict]:
    """Termes interdits dans les champs rédigés ; un terme employé par l'enquêté·e (citation d'un tour de
    l'enquêté·e) n'est pas signalé. Une citation de l'enquêteur n'exempte jamais : sa catégorie ne devient
    pas celle de l'enquêté·e ; si le terme figure dans une question de l'enquêteur du matériau, le constat
    porte `from_interviewer: true`."""
    quotes = interpretation_guard.fold(" ".join(interviewee_quotes))
    interviewer = interpretation_guard.fold(" ".join(interviewer_texts))
    findings, seen = [], set()
    for name, text in _authored(episode).items():
        folded = interpretation_guard.fold(text)
        for label, regex in _FORBIDDEN:
            if label not in seen and regex.search(folded) and not regex.search(quotes):
                seen.add(label)
                findings.append({"term": label, "field": name, "from_interviewer": bool(regex.search(interviewer))})
    return findings


# « justification » niée (« aucune restriction, justification ou évaluation explicite ») : rien n'est affirmé.
_NEGATED_JUSTIFICATION = re.compile(r"\b(?:aucune?|sans|ni|pas de|pas d)\b[^.;]{0,80}?\bjustifi\w*")


def justification_unsupported(episode: dict, interviewee_quotes: list[str], turn_texts: list[str]) -> bool:
    authored = _NEGATED_JUSTIFICATION.sub(" ", interpretation_guard.fold(" ".join(_authored(episode).values())))
    if not _JUSTIFICATION.search(authored):
        return False
    material = interpretation_guard.fold(" ".join(interviewee_quotes + turn_texts)).replace("'", " ")
    return not _REASON_MARKERS.search(material)


def validate_episodes(episodes: list[dict], transcript: dict, practices: list[dict], signals: list[dict],
                      candidates: list[dict], speaker_warnings: dict | None = None) -> dict:
    """Valide les épisodes d'UN entretien. Renvoie {"episodes": annotés, "report": bilan}."""
    warnings = speaker_warnings or {}
    interview_id = transcript["interview_id"]
    turns = evidence_validator.index_turns(transcript)
    practice_by_id = {p["practice_id"]: p for p in practices}
    signal_by_id = {s["signal_id"]: s for s in signals}
    candidate_by_id = {c["candidate_id"]: c for c in candidates}

    def voiced(turn_id: str) -> bool:
        return turn_id in turns and (turns[turn_id]["speaker"] != SPEAKER_INTERVIEWER or turn_id in warnings)

    def pos(turn_id: str) -> int | None:
        return turns[turn_id]["position"] if turn_id in turns else None

    ordered = sorted(episodes, key=lambda e: (pos(e.get("turn_start")) if pos(e.get("turn_start")) is not None
                                              else len(turns), e.get("candidate_ids", [])))
    annotated, all_issues = [], []
    addressed: dict[str, list[str]] = {}
    for number, episode in enumerate(ordered, start=1):
        episode_id = f"{interview_id}_E{number:03d}"
        issues: list[dict] = []
        status = episode.get("episode_status")
        cands = episode.get("candidate_ids", [])
        for cid in cands:
            if cid in candidate_by_id:
                addressed.setdefault(cid, []).append(episode_id)
            else:
                issues.append(make_issue("UNKNOWN_CANDIDATE_ID", candidate_id=cid))
        cand_practices = {p for cid in cands if cid in candidate_by_id for p in candidate_by_id[cid]["practice_ids"]}
        cand_signals = {s for cid in cands if cid in candidate_by_id for s in candidate_by_id[cid]["signal_ids"]}
        cand_turns = {t for cid in cands if cid in candidate_by_id for t in candidate_by_id[cid]["turn_ids"]}

        for pid in episode.get("practice_ids", []):
            if pid not in practice_by_id:
                issues.append(make_issue("UNKNOWN_PRACTICE_ID", practice_id=pid))
            elif pid not in cand_practices:
                issues.append(make_issue("PRACTICE_NOT_IN_CANDIDATES", practice_id=pid))
        for sid in episode.get("signal_ids", []):
            if sid not in signal_by_id:
                issues.append(make_issue("UNKNOWN_SIGNAL_ID", signal_id=sid))
            elif sid not in cand_signals:
                issues.append(make_issue("SIGNAL_NOT_IN_CANDIDATES", signal_id=sid))

        # intervalle
        span = None
        start, end = episode.get("turn_start"), episode.get("turn_end")
        range_problems = []
        for name, turn_id in (("turn_start", start), ("turn_end", end)):
            problem = evidence_validator.turn_problem(turn_id, turns, interview_id)
            if problem:
                range_problems.append(make_issue("UNKNOWN_RANGE_TURN" if problem == "UNKNOWN_TURN_ID" else problem,
                                                 field=name, turn_id=turn_id))
        issues.extend(range_problems)
        if not range_problems:
            if pos(start) > pos(end):
                issues.append(make_issue("INVALID_TURN_RANGE", turn_start=start, turn_end=end))
            else:
                span = (pos(start), pos(end))

        # citations
        evidence = episode.get("evidence", [])
        results = evidence_validator.validate_evidence(evidence, turns, interview_id)
        issues.extend(make_issue(r["code"], evidence_index=i, turn_id=evidence[i].get("turn_id"))
                      for i, r in enumerate(results) if r["code"])
        valid = [evidence[i] for i, r in enumerate(results) if r["valid"]]
        if not valid:
            issues.append(make_issue("NO_VALID_EVIDENCE"))
        interviewee_quotes = [e["quote"] for e in valid if voiced(e["turn_id"])]
        if valid and not interviewee_quotes:
            issues.append(make_issue("NO_INTERVIEWEE_EVIDENCE" if status == "accountability_episode"
                                     else "NO_INTERVIEWEE_EVIDENCE_NON_EPISODE"))
        if span:
            for i, r in enumerate(results):
                if r["valid"] and not span[0] <= pos(evidence[i]["turn_id"]) <= span[1]:
                    issues.append(make_issue("EVIDENCE_OUTSIDE_RANGE", evidence_index=i, turn_id=evidence[i]["turn_id"]))

        # opérations
        moves = episode.get("accounting_moves", [])
        material_turns = (cand_turns | {e["turn_id"] for e in valid}
                          | {e.get("turn_id") for pid in episode.get("practice_ids", []) if pid in practice_by_id
                             for e in practice_by_id[pid].get("evidence", [])}
                          | {t for sid in episode.get("signal_ids", []) if sid in signal_by_id
                             for t in signal_by_id[sid].get("turn_ids", [])})
        for m_index, move in enumerate(moves):
            move_turns = move.get("evidence_turn_ids", [])
            unknown = [t for t in move_turns if t not in turns]
            for t in unknown:
                issues.append(make_issue("UNKNOWN_MOVE_TURN", move_index=m_index, turn_id=t))
            known = [t for t in move_turns if t in turns]
            if known and not any(voiced(t) for t in known):
                issues.append(make_issue("MOVE_WITHOUT_INTERVIEWEE_TURN", move_index=m_index, move_type=move.get("type")))
            for t in known:
                if t not in material_turns:
                    issues.append(make_issue("MOVE_TURN_OUTSIDE_MATERIAL", move_index=m_index, turn_id=t))

        if status == "accountability_episode":
            if not moves:
                issues.append(make_issue("NO_ACCOUNTING_MOVES"))
            trigger_signals = [s for s in episode.get("signal_ids", []) if s in signal_by_id
                               and signal_by_id[s].get("signal_type") in TRIGGER_SIGNAL_TYPES]
            structural = {t for cid in cands if cid in candidate_by_id for t in candidate_by_id[cid]["trigger_types"]
                          if t != "signal_proximity"}
            if not trigger_signals and not structural:
                issues.append(make_issue("INSUFFICIENT_MATERIAL"))
        elif status == "ordinary_practice":
            if len(moves) > MAX_ORDINARY_MOVES or episode.get("boundary_objects"):
                issues.append(make_issue("ORDINARY_PRACTICE_WITH_ACCOUNTING", move_count=len(moves)))
        if status != "accountability_episode" and episode.get("boundary_objects") and status != "ordinary_practice":
            issues.append(make_issue("BOUNDARY_WITHOUT_ACCOUNTABILITY"))

        # fusions
        pids = [p for p in episode.get("practice_ids", []) if p in practice_by_id]
        shared = lambda a, b: any(a in candidate_by_id[c]["practice_ids"] and b in candidate_by_id[c]["practice_ids"]  # noqa: E731
                                  for c in cands if c in candidate_by_id)
        for i, a in enumerate(pids):
            for b in pids[i + 1:]:
                if shared(a, b):
                    continue
                ta, tb = (normalize_task(practice_by_id[x].get("academic_task")) for x in (a, b))
                if ta and tb and ta != tb:
                    issues.append(make_issue("MERGED_DIFFERENT_TASKS", practice_ids=[a, b]))
                da, db = (practice_by_id[x].get("practice_domain") for x in (a, b))
                if {da, db} == {"academic", "personal"}:
                    issues.append(make_issue("MERGED_DIFFERENT_DOMAINS", practice_ids=[a, b]))
        for sid in episode.get("signal_ids", []):
            signal = signal_by_id.get(sid)
            if not signal or signal.get("signal_type") != "cross_turn_contradiction":
                continue
            cited = sorted(p for p in (pos(t) for t in signal.get("turn_ids", [])) if p is not None)
            if len(cited) < 2:
                continue
            for pid in pids:
                points = [pos(e.get("turn_id")) for e in practice_by_id[pid].get("evidence", []) if pos(e.get("turn_id")) is not None]
                if points and all(cited[0] < p < cited[-1] for p in points) and not any(
                        abs(p - c) <= 1 for p in points for c in cited):
                    issues.append(make_issue("INTERVENING_PRACTICE_MERGED", practice_id=pid, signal_id=sid))

        # fusions : les candidats réunis doivent être reliés (graphe connexe de relations explicites)
        known_cands = [candidate_by_id[c] for c in dict.fromkeys(cands) if c in candidate_by_id]
        components = connected_groups(known_cands) if len(known_cands) > 1 else [known_cands]
        if len(components) > 1:
            issues.append(make_issue("DISCONNECTED_MERGE", components=components))

        # vocabulaire (citations associées de l'enquêté·e ; questions de l'enquêteur du matériau)
        associated = interviewee_quotes + [
            e.get("quote", "") for pid in episode.get("practice_ids", []) if pid in practice_by_id
            for e in practice_by_id[pid].get("evidence", []) if voiced(e.get("turn_id"))
            and (e.get("validation") or {}).get("valid", True)] + [
            e.get("quote", "") for sid in episode.get("signal_ids", []) if sid in signal_by_id
            for e in signal_by_id[sid].get("evidence", []) if voiced(e.get("turn_id"))
            and (e.get("validation") or {}).get("valid", True)]
        interviewer_texts = [turns[t]["text"] for t in material_turns | {t for m in moves for t in m.get(
            "evidence_turn_ids", [])} if t in turns and not voiced(t)]
        for finding in scan_vocabulary(episode, associated, interviewer_texts):
            issues.append(make_issue("INTERPRETIVE_VOCABULARY", **finding))
            if finding["from_interviewer"]:
                issues.append(make_issue("INTERVIEWER_TERM_ATTRIBUTED", term=finding["term"], field=finding["field"]))
        material_texts = [turns[t]["text"] for t in material_turns if voiced(t)]
        if justification_unsupported(episode, interviewee_quotes, material_texts):
            issues.append(make_issue("UNSUPPORTED_JUSTIFICATION"))

        # avertissements de locuteur : propagés par TRACE
        involved = [t for t in dict.fromkeys([*(e.get("turn_id") for e in evidence),
                                              *(t for m in moves for t in m.get("evidence_turn_ids", [])),
                                              *(t for cid in cands if cid in candidate_by_id
                                                for t in candidate_by_id[cid]["speaker_warning_turn_ids"])])
                    if t in warnings]
        speaker_warnings_out = [{"turn_id": t, "suggested_speaker": warnings[t].get("suggested_speaker"),
                                 "confidence": warnings[t].get("confidence")} for t in involved]
        if involved:
            issues.append(make_issue("SPEAKER_WARNING_PROPAGATED", turn_ids=involved))

        if status == "uncertain":
            if not episode.get("needs_review"):
                issues.append(make_issue("UNCERTAIN_NOT_FLAGGED"))
            if episode.get("confidence") != "low":
                issues.append(make_issue("UNCERTAIN_CONFIDENCE", confidence=episode.get("confidence")))
        if episode.get("needs_review"):
            issues.append(make_issue("MODEL_FLAGGED_REVIEW"))

        severities = {i["severity"] for i in issues}
        validation_status = "rejected" if ERROR in severities else "needs_review" if WARNING in severities else "valid"
        needs_review = bool(episode.get("needs_review")) or bool(issues)
        annotated.append({
            "episode_id": episode_id,
            **{k: v for k, v in episode.items() if k not in ("evidence", "needs_review")},
            "evidence": [{**e, "validation": r} for e, r in zip(evidence, results)],
            "model_needs_review": bool(episode.get("needs_review")),
            "needs_review": needs_review,
            "speaker_warnings": speaker_warnings_out,
            "validation_status": validation_status,
            # « épisode propre » pour les étapes ultérieures : jamais un épisode rejeté (dont fusion déconnectée)
            "usable_for_next_stages": validation_status != "rejected",
            "review_reasons": sorted({i["code"] for i in issues}),
        })
        all_issues.extend({"object_id": episode_id, **issue} for issue in issues)

    for cid, eids in addressed.items():
        if len(eids) > 1:
            all_issues.append({"object_id": cid, **make_issue("CANDIDATE_IN_SEVERAL_EPISODES", episode_ids=eids)})
    unaddressed = [c["candidate_id"] for c in candidates if c["candidate_id"] not in addressed]
    for cid in unaddressed:
        all_issues.append({"object_id": cid, **make_issue("CANDIDATE_NOT_ADDRESSED")})

    severities = [i["severity"] for i in all_issues]
    evidence_count = sum(len(e["evidence"]) for e in annotated)
    valid_count = sum(e["validation"]["valid"] for ep in annotated for e in ep["evidence"])
    report = {
        "validator_version": VALIDATOR_VERSION,
        "episode_count": len(annotated),
        "rejected_episode_ids": [e["episode_id"] for e in annotated if e["validation_status"] == "rejected"],
        "disconnected_merge_episode_ids": [e["episode_id"] for e in annotated
                                           if "DISCONNECTED_MERGE" in e["review_reasons"]],
        "episodes_needing_review": [e["episode_id"] for e in annotated if e["needs_review"]],
        "unaddressed_candidate_ids": unaddressed,
        "evidence_count": evidence_count,
        "valid_evidence_count": valid_count,
        "invalid_evidence_count": evidence_count - valid_count,
        "error_count": severities.count(ERROR),
        "warning_count": severities.count(WARNING),
        "info_count": severities.count(INFO),
        "issues": all_issues,
    }
    return {"episodes": annotated, "report": report}


def has_problems(report: dict) -> bool:
    return bool(report.get("error_count") or report.get("warning_count"))
