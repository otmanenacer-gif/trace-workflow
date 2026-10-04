"""Contrôle méthodologique d'une réponse d'agent, AVANT de la rendre à l'orchestrateur de l'étape.

Applique à une réponse (déjà conforme au schéma Pydantic de l'agent) les validateurs EXISTANTS de TRACE, sans rien
enregistrer, pour que le LocalAgentRunner puisse demander au modèle local de corriger une anomalie bloquante :

- étape 3 : audit des locuteurs (core/speaker_attribution_auditor.py), puis pour le Practice Extractor,
  l'Interaction Signal Reader et la lecture à longue distance : post-traitements (sélectivité) et validation des
  preuves (core/evidence_validator.py) sur la transcription de l'entretien ;
- étape 4 : validateur des épisodes (core/accountability_episode_validator.py) sur les SEULS candidats du bloc de la
  requête (identifiants abrégés rétablis comme dans le pipeline) ;
- étape 5 : validateur de l'étape 5 (core/trajectory_validator.py) sur le matériau préparé de l'entretien ;
- étape 6 : validateur de l'étape 6 (core/cross_interview_validator.py) sur le matériau du corpus.

Gravité : « error » bloquante (citation absente ou non littérale, tour ou identifiant inconnu, épisode rejeté,
appui vide…) ; « warning » et « info » : signalées seulement. Les validateurs ne sont jamais modifiés ni
contournés : après les corrections, l'orchestrateur les applique de nouveau et décide (needs_review, rejet).
"""

from __future__ import annotations

import json
from pathlib import Path

from agents import episode_grounding_checker
from agents.base import AgentSpec, build_agent_input, serialize_agent_input, sha256_text
from core import accountability, analysis, config, cross_interview, evidence_validator, signal_selectivity
from core import accountability_candidates as candidates_mod
from core import accountability_episode_validator as episode_validator
from core import cross_interview_material as cm
from core import cross_interview_validator as cross_validator
from core import speaker_attribution_auditor as speaker_audit
from core import trajectory
from core import trajectory_candidates as tc
from core import trajectory_validator
from core import citation_resolver, stage3_repair

# Agents sémantiques des étapes 3 à 6
STAGE3_SPECS: tuple[AgentSpec, ...] = (analysis.AUDITOR, analysis.PRACTICE, analysis.INTERACTION,
                                       analysis.LONG_DISTANCE)
STAGE4_SPECS: tuple[AgentSpec, ...] = (accountability.SPEC,)
STAGE5_SPECS: tuple[AgentSpec, ...] = (trajectory.SPEC,)
STAGE6_SPECS: tuple[AgentSpec, ...] = (cross_interview.SPEC,)
# Agents de diagnostic (rapport seulement) : connus du runner (journal, réponse réservée), jamais contrôlés ici
DIAGNOSTIC_SPECS: tuple[AgentSpec, ...] = (episode_grounding_checker.SPEC,)
AGENT_SPECS = STAGE3_SPECS + STAGE4_SPECS + STAGE5_SPECS + STAGE6_SPECS + DIAGNOSTIC_SPECS
STAGE_OF_AGENT = {**{spec.name: "3" for spec in STAGE3_SPECS}, **{spec.name: "4" for spec in STAGE4_SPECS},
                  **{spec.name: "5" for spec in STAGE5_SPECS}, **{spec.name: "6" for spec in STAGE6_SPECS}}


def issue_line(issue: dict) -> str:
    where = issue.get("item_turn_id") or issue.get("object_id") or ""
    turn = issue.get("turn_id")
    context = " ".join(x for x in (where, f"(tour {turn})" if turn and turn != where else "") if x)
    return f"{issue['code']}{' — ' + context if context else ''} : {issue['message']}"


def _classify(report: dict, issues: list[dict]) -> dict:
    for issue in issues:
        target = {evidence_validator.ERROR: report["blocking"], evidence_validator.WARNING: report["warnings"]}.get(
            issue["severity"], report["info"])
        target.append(issue_line(issue))
    report["ok"] = not report["blocking"]
    return report


class MethodChecker:
    """Contrôle d'une réponse d'agent dans son contexte : dossiers des entretiens d'un run (étapes 3 à 5) ou
    corpus préparé de l'étape 6. Appelé par le LocalAgentRunner : renvoie les anomalies BLOQUANTES."""

    def __init__(self, interview_dirs: dict[str, str] | None = None, corpus=None):
        self.interview_dirs = {k: Path(v) for k, v in (interview_dirs or {}).items()}
        self.corpus = corpus  # cross_interview.Stage6Input
        self._contexts: dict[str, dict] = {}

    def __call__(self, spec: AgentSpec, label: str, user_content: str, output: dict) -> list[str]:
        if spec.name == accountability.SPEC.name:
            # étape 4 : une anomalie d'un épisode est réparée sur son seul candidat par l'orchestrateur
            # (core/stage4_repair.py), jamais en redemandant tout le bloc
            return []
        return self.report(spec, label, user_content, output)["blocking"]

    def report(self, spec: AgentSpec, label: str, user_content: str, output: dict) -> dict:
        report = {"agent": spec.name, "agent_label": spec.label, "ok": True, "blocking": [], "warnings": [],
                  "info": [], "counts": {}, "checked": False}
        if spec.name in {s.name for s in DIAGNOSTIC_SPECS}:
            return report  # diagnostic : sa réponse est un rapport, rien à valider
        if spec.name == cross_interview.SPEC.name:
            issues = self._comparison(user_content, output, report)
        else:
            interview_dir = self.interview_dirs.get(label.split("/", 1)[0])
            if interview_dir is None:
                return report  # contexte inconnu : les validateurs de l'orchestrateur s'appliquent seuls
            if spec.name == accountability.SPEC.name:
                issues = self._episodes(user_content, output, interview_dir, report)
            elif spec.name == trajectory.SPEC.name:
                issues = self._trajectory(user_content, output, interview_dir, report)
            else:
                issues = self._stage3(spec, label, output, interview_dir, report)
        if issues is None:
            return report
        report["checked"] = True
        return _classify(report, issues)

    # --- Étape 3 -------------------------------------------------------------------------------

    def _stage3(self, spec: AgentSpec, label: str, output: dict, interview_dir: Path, report: dict) -> list[dict]:
        transcript = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
        if spec.name == analysis.AUDITOR.name:
            document = speaker_audit.build_audit_document(transcript, speaker_audit.prepare_audit(transcript), output)
            report["counts"] = {"assessments": len(output["assessments"]), "evidence": document["evidence_count"],
                                "invalid_evidence": document["invalid_evidence_count"],
                                "review_count": document["review_count"]}
            return document["issues"]
        kind = analysis.PRACTICE if spec.name == analysis.PRACTICE.name else analysis.INTERACTION
        audit_path = interview_dir / config.ANALYSIS_SUBDIR / analysis.AUDITOR.output_filename
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
        warnings = speaker_audit.agent_warnings(audit, transcript) if audit else {}
        prepared = analysis.with_speaker_warnings(analysis.prepare_interview({"output_dir": str(interview_dir)}),
                                                  warnings)
        items, extra, _ = analysis.postprocess_for(kind, prepared)(output[spec.items_key])
        validated = evidence_validator.validate_agent_output(kind.name, items, transcript, kind.id_letter, warnings)
        report["counts"] = {spec.items_key: len(output[spec.items_key]),
                            "evidence": validated["report"]["evidence_count"],
                            "invalid_evidence": validated["report"]["invalid_evidence_count"],
                            "needs_review": len(validated["report"]["objects_needing_review"])}
        for signal in extra.get("set_aside_signals") or []:
            report["info"].append("Signal écarté par la sélectivité déterministe "
                                  f"({signal.get('set_aside_reason')}) : « {signal.get('surface_form')} ».")
        cues = extra.get("non_use_cues") or {}
        if cues.get("uncovered_turn_ids") and "/bloc" not in label:
            report["info"].append("Tours où l'enquêté·e semble évoquer un non-usage sans pratique correspondante : "
                                  + ", ".join(cues["uncovered_turn_ids"]) + " (à relire, sans rien inventer).")
        return validated["report"]["issues"]

    # --- Étape 3 : contrôle et réparation d'UN objet (core/stage3_repair.py) ------------------------

    def repairable(self, spec: AgentSpec | None, label: str) -> bool:
        """Réparation ciblée possible : Practice Extractor, Interaction Reader ou lecture à longue distance, sur un
        entretien dont le contexte est connu."""
        return stage3_repair.repairable(spec) and label.split("/", 1)[0] in self.interview_dirs

    def _context(self, label: str) -> dict:
        interview_id = label.split("/", 1)[0]
        if interview_id not in self._contexts:
            interview_dir = self.interview_dirs[interview_id]
            transcript = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
            audit_path = interview_dir / config.ANALYSIS_SUBDIR / analysis.AUDITOR.output_filename
            audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
            turns = evidence_validator.index_turns(transcript)
            self._contexts[interview_id] = {
                "transcript": transcript, "turns": turns,
                "warnings": speaker_audit.agent_warnings(audit, transcript) if audit else {},
                "position": {turn_id: info["position"] for turn_id, info in turns.items()}}
        return self._contexts[interview_id]

    def item_check(self, spec: AgentSpec, label: str, item: dict) -> dict:
        """Anomalies BLOQUANTES d'un objet, par le validateur habituel (core/evidence_validator.validate_item, comme
        pour la réponse entière : les anomalies « error » de l'étape 3 portent toutes sur un seul objet).
        → {"problems", "invalid_citations", "citations"}."""
        context = self._context(label)
        kind = analysis.PRACTICE if spec.name == analysis.PRACTICE.name else analysis.INTERACTION
        if kind is analysis.INTERACTION:
            item = signal_selectivity.complete_turn_ids(item, context["position"])
        results, issues = evidence_validator.validate_item(item, kind.name, context["turns"],
                                                           context["transcript"]["interview_id"],
                                                           context["warnings"])
        problems = []
        for issue in issues:
            if issue["severity"] != evidence_validator.ERROR:
                continue
            where = []
            if issue.get("evidence_index") is not None:
                where.append(f"citation {issue['evidence_index'] + 1}")
            if issue.get("field"):
                where.append(issue["field"])
            if issue.get("turn_id"):
                where.append(f"tour {issue['turn_id']}")
            problems.append(f"{issue['code']}{' — ' + ', '.join(where) if where else ''} : {issue['message']}")
        blocking = [i for i in issues if i["severity"] == evidence_validator.ERROR]
        return {"problems": problems, "invalid_citations": sum(not r["valid"] for r in results),
                "citations": len(results), "invalid_indices": [i for i, r in enumerate(results) if not r["valid"]],
                # anomalies portant uniquement sur des citations (pas sur un champ ni un intervalle)
                "citation_only": bool(blocking) and all(i["code"] in stage3_repair.CITATION_CODES
                                                        and not i.get("field") for i in blocking)}

    def _allowed(self, label: str, user_content: str) -> set[str]:
        """turn_id du matériau envoyé à l'agent (bloc ou sélection) : seuls tours citables par une correction."""
        turns = self._context(label)["transcript"]["turns"]
        allowed = {t["turn_id"] for t in turns if f'"{t["turn_id"]}"' in user_content}
        return allowed or {t["turn_id"] for t in turns}

    def resolve_citations(self, spec: AgentSpec, label: str, user_content: str, item: dict) -> tuple[dict, list]:
        """Correction DÉTERMINISTE des citations invalides d'un objet (core/citation_resolver.py), sans modèle.
        → (objet, journal des décisions)."""
        invalid = self.item_check(spec, label, item)["invalid_indices"]
        if not invalid:
            return item, []
        turns = self._context(label)["transcript"]["turns"]
        return citation_resolver.resolve_item(item, invalid, turns, self._allowed(label, user_content),
                                              signal=spec.name != analysis.PRACTICE.name)

    def repair_turns(self, spec: AgentSpec, label: str, user_content: str, item) -> str:
        """Les seuls tours utiles à la réparation d'un objet, pris dans le matériau envoyé à l'agent (bloc ou
        sélection) : tours cités, intervalle de la pratique, turn_ids du signal, tours où figure (à la forme près)
        une citation fautive, et leurs voisins immédiats ; texte exact, présentation habituelle des agents."""
        context = self._context(label)
        turns, transcript = context["transcript"]["turns"], context["transcript"]
        position = context["position"]
        allowed_ids = self._allowed(label, user_content)
        allowed = {i for i, t in enumerate(turns) if t["turn_id"] in allowed_ids}
        item = item if isinstance(item, dict) else {}
        evidence = [e for e in item.get("evidence") or [] if isinstance(e, dict)]
        cited = [position[e.get("turn_id")] for e in evidence if e.get("turn_id") in position]
        listed = [position[t] for t in item.get("turn_ids") or [] if t in position]
        span = []
        if item.get("turn_start") in position and item.get("turn_end") in position:
            first, last = sorted((position[item["turn_start"]], position[item["turn_end"]]))
            span = list(range(first, min(last, first + 6) + 1))
        located = []
        for e in evidence:
            quote = e.get("quote") or ""
            if quote.strip() and not (e.get("turn_id") in position
                                      and evidence_validator.match_quote(quote, turns[position[e["turn_id"]]]["text"])
                                      == "exact"):
                located += [i for i in sorted(allowed)
                            if evidence_validator.match_quote(quote, turns[i]["text"]) != "not_found"]
        core = list(dict.fromkeys(cited + located + listed + span))
        neighbours = [j for i in core for j in (i - 1, i + 1, i - 2, i + 2)]
        chosen: list[int] = []
        size = 0
        for i in dict.fromkeys(core + neighbours):
            if i not in allowed or len(chosen) >= stage3_repair.REPAIR_MAX_TURNS:
                continue
            length = len(turns[i]["text"])
            if chosen and size + length > stage3_repair.REPAIR_MAX_CHARS:
                continue
            chosen.append(i)
            size += length
        if not chosen:  # aucun repère exploitable : début du matériau envoyé
            chosen = sorted(allowed)[:6]
        sub = {"interview_id": transcript["interview_id"], "turns": [turns[i] for i in sorted(chosen)]}
        warnings = {t["turn_id"]: w for t in sub["turns"] if (w := context["warnings"].get(t["turn_id"]))}
        return serialize_agent_input(build_agent_input(sub, warnings))

    # --- Étape 4 -------------------------------------------------------------------------------

    def _episodes(self, user_content: str, output: dict, interview_dir: Path, report: dict) -> list[dict] | None:
        prepared = accountability.prepare_stage4({"output_dir": str(interview_dir)})
        digest = sha256_text(user_content)
        request = next((r for r in prepared.requests if sha256_text(r["user_message"]) == digest), None)
        if request is None:
            return None
        wanted = set(request["candidate_ids"])
        episodes = candidates_mod.expand_ids(output, prepared.interview_id)["episodes"]
        candidates = [c for c in prepared.built["candidates"] if c["candidate_id"] in wanted]
        validated = episode_validator.validate_episodes(episodes, prepared.transcript, prepared.practices,
                                                        prepared.signals, candidates, prepared.speaker_warnings)
        annotated = validated["episodes"]
        report["counts"] = {"candidates": len(wanted), "episodes": len(annotated),
                            "rejected": sum(e["validation_status"] == "rejected" for e in annotated),
                            "usable": sum(bool(e.get("usable_for_next_stages")) for e in annotated),
                            "needs_review": sum(e["needs_review"] for e in annotated)}
        return validated["report"]["issues"]  # un épisode rejeté porte une anomalie « error » : bloquante

    # --- Étape 5 -------------------------------------------------------------------------------

    def _trajectory(self, user_content: str, output: dict, interview_dir: Path, report: dict) -> list[dict] | None:
        prepared = trajectory.prepare_stage5({"output_dir": str(interview_dir)})
        if prepared.request is None or prepared.request["user_message"] != user_content:
            return None
        validated = trajectory_validator.validate_trajectory(tc.expand_ids(output, prepared.interview_id),
                                                             prepared.material, prepared.transcript,
                                                             prepared.speaker_warnings)
        document = validated["document"]
        claims, criteria = document["trajectory_claims"], document["student_role_criteria"]
        report["counts"] = {"claims": len(claims), "kept": sum(c["usable_for_next_stages"] for c in claims),
                            "rejected": sum(not c["usable_for_next_stages"] for c in claims),
                            "requalified": sum("model_claim_type" in c for c in claims),
                            "criteria": len(criteria),
                            "needs_review": sum(c["needs_review"] for c in claims + criteria),
                            "estimated_input_tokens": prepared.estimated_input_tokens}
        for claim in claims:
            if "model_claim_type" in claim:
                report["info"].append(f"{claim['claim_id']} : {claim['model_claim_type']} requalifiée en "
                                      f"{claim['claim_type']} par TRACE (règle de l'étape 5).")
        return validated["report"]["issues"]

    # --- Étape 6 -------------------------------------------------------------------------------

    def _comparison(self, user_content: str, output: dict, report: dict) -> list[dict] | None:
        prepared = self.corpus
        if prepared is None or prepared.request is None or prepared.request["user_message"] != user_content:
            return None
        validated = cross_validator.validate_comparison(cm.expand_ids(output, prepared.material), prepared.material,
                                                        prepared.excluded_ids)
        document = validated["document"]
        claims = document["cross_case_claims"]
        report["counts"] = {"interviews": prepared.checked["n_usable"], "claims": len(claims),
                            "kept": sum(c["usable_for_next_stages"] for c in claims),
                            "rejected": sum(not c["usable_for_next_stages"] for c in claims),
                            "requalified": sum("model_claim_type" in c for c in claims),
                            "negative_cases": len(document["negative_cases"]),
                            "needs_review": sum(c["needs_review"] for c in claims),
                            "estimated_input_tokens": prepared.estimated_input_tokens}
        return validated["report"]["issues"]
