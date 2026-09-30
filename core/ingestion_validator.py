"""Contrôles déterministes après structuration et rapport de qualité d'ingestion.

Contrôle de couverture : on vérifie qu'aucun texte n'a été perdu ni ajouté
entre raw_text.txt et les tours de parole. Méthode :
- on retire tous les blancs (espaces, tabulations, sauts de ligne) du texte
  brut d'une part, et de la concaténation « marqueur + texte » de chaque
  tour d'autre part ;
- les deux chaînes doivent être IDENTIQUES (exact_match = true) ;
- sinon, on compare les suites de mots (séparés par des blancs) pour mesurer
  la part du texte brut retrouvée dans l'ordre (token_coverage_ratio).

Il ne s'agit PAS d'une mesure de précision de l'attribution des locuteurs :
sans vérité terrain, TRACE ne prétend pas mesurer une précision.
"""

import difflib
import re

from core.schemas import (
    SCHEMA_VERSION,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    SPEAKER_INTERVIEWEE,
    SPEAKER_INTERVIEWER,
    SPEAKER_UNKNOWN,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PASS_WITH_WARNINGS,
    make_warning,
)

# En dessous de ce taux de mots retrouvés : perte substantielle (FAIL)
MIN_TOKEN_COVERAGE = 0.99
# Au-delà de cette part de mots ajoutés : contenu inventé (FAIL)
MAX_ADDED_TOKEN_RATIO = 0.01

_BLANKS = re.compile(r"\s+")


def _non_blank(text: str) -> str:
    return _BLANKS.sub("", text)


def check_coverage(raw_text: str, turns: list[dict]) -> tuple[dict, list[dict]]:
    """Compare le texte brut et les tours. Renvoie (métriques, avertissements)."""
    rebuilt = "\n".join((t["marker"] or "") + "\n" + t["text"] for t in turns)
    raw_compact, rebuilt_compact = _non_blank(raw_text), _non_blank(rebuilt)
    raw_tokens, rebuilt_tokens = raw_text.split(), rebuilt.split()

    if raw_compact == rebuilt_compact:
        matched = len(raw_tokens)
        missing_sample = []
    else:
        matcher = difflib.SequenceMatcher(None, raw_tokens, rebuilt_tokens, autojunk=False)
        blocks = matcher.get_matching_blocks()
        matched = sum(b.size for b in blocks)
        covered = set()
        for b in blocks:
            covered.update(range(b.a, b.a + b.size))
        missing_sample = [tok for i, tok in enumerate(raw_tokens) if i not in covered][:10]

    missing = len(raw_tokens) - matched
    added = len(rebuilt_tokens) - matched
    ratio = matched / len(raw_tokens) if raw_tokens else 1.0
    metrics = {
        "method": "comparaison hors blancs (marqueurs de locuteur inclus), puis alignement des mots",
        "raw_non_blank_chars": len(raw_compact),
        "structured_non_blank_chars": len(rebuilt_compact),
        "exact_match": raw_compact == rebuilt_compact,
        "raw_token_count": len(raw_tokens),
        "structured_token_count": len(rebuilt_tokens),
        "token_coverage_ratio": round(ratio, 4),
        "missing_token_count": missing,
        "added_token_count": added,
        "missing_token_sample": missing_sample,
    }

    warnings = []
    if ratio < MIN_TOKEN_COVERAGE:
        warnings.append(make_warning(
            "CONTENT_LOSS", f"Seulement {ratio:.1%} des mots extraits se retrouvent dans les tours ({missing} manquant(s))."
        ))
    if rebuilt_tokens and added / len(rebuilt_tokens) > MAX_ADDED_TOKEN_RATIO:
        warnings.append(make_warning("CONTENT_ADDED", f"{added} mot(s) absent(s) du texte extrait."))
    if not metrics["exact_match"] and not warnings:
        warnings.append(make_warning(
            "CONTENT_MISMATCH_MINOR",
            f"Texte non identique hors blancs ({missing} mot(s) manquant(s), {added} ajouté(s)).",
        ))
    return metrics, warnings


def check_turn_sequence(turns: list[dict]) -> list[dict]:
    """Identifiants uniques, index consécutifs, lignes source croissantes."""
    ids = [t["turn_id"] for t in turns]
    indexes_ok = [t["index"] for t in turns] == list(range(1, len(turns) + 1))
    lines = [t["source"]["line_start"] for t in turns]
    if len(set(ids)) != len(ids) or not indexes_ok or lines != sorted(lines):
        return [make_warning("INVALID_TURN_SEQUENCE")]
    return []


def compute_status(warnings: list[dict]) -> str:
    severities = {w["severity"] for w in warnings}
    if SEVERITY_ERROR in severities:
        return STATUS_FAIL
    if SEVERITY_WARNING in severities:
        return STATUS_PASS_WITH_WARNINGS
    return STATUS_PASS


def turn_statistics(turns: list[dict]) -> dict:
    """Comptes par locuteur et part approximative du texte attribué à un rôle."""
    counts = {s: 0 for s in (SPEAKER_INTERVIEWER, SPEAKER_INTERVIEWEE, SPEAKER_UNKNOWN)}
    chars = dict.fromkeys(counts, 0)
    for turn in turns:
        counts[turn["speaker"]] += 1
        chars[turn["speaker"]] += len(_non_blank(turn["text"]))
    total = sum(chars.values())
    attributed = chars[SPEAKER_INTERVIEWER] + chars[SPEAKER_INTERVIEWEE]
    return {
        "turn_count": len(turns),
        "interviewer_turn_count": counts[SPEAKER_INTERVIEWER],
        "interviewee_turn_count": counts[SPEAKER_INTERVIEWEE],
        "unknown_segment_count": counts[SPEAKER_UNKNOWN],
        "non_blank_chars_by_speaker": chars,
        "attributed_text_ratio": round(attributed / total, 4) if total else 0.0,
    }


def page_line_ranges(lines: list[tuple[int | None, str]]) -> list[dict]:
    """Pour un PDF : lignes de raw_text.txt correspondant à chaque page."""
    ranges = {}
    for number, (page, text) in enumerate(lines, start=1):
        if page is None:
            continue
        entry = ranges.setdefault(page, {"page": page, "line_start": number, "line_end": number, "char_count": 0})
        entry["line_end"] = number
        entry["char_count"] += len(text)
    return list(ranges.values())


def build_report(
    *,
    interview_id: str,
    source: dict,
    processed_at: str,
    extraction: dict | None,
    raw_text: str,
    lines: list,
    turns: list[dict],
    warnings: list[dict],
) -> dict:
    """Assemble le rapport de qualité d'ingestion et calcule le statut final.

    `warnings` contient déjà les avertissements d'extraction et de
    structuration ; ceux des contrôles sont ajoutés ici.
    """
    warnings = list(warnings)
    coverage = None
    if extraction is not None:
        if not raw_text.strip():
            warnings.append(make_warning("NO_TEXT_EXTRACTED"))
        coverage, coverage_warnings = check_coverage(raw_text, turns)
        warnings.extend(coverage_warnings)
        warnings.extend(check_turn_sequence(turns))

    status = compute_status(warnings)
    return {
        "schema_version": SCHEMA_VERSION,
        "interview_id": interview_id,
        "processed_at": processed_at,
        "status": status,
        "needs_manual_review": status != STATUS_PASS,
        "source": source,
        "extraction": None if extraction is None else {
            "extractor": extraction["extractor"],
            "encoding": extraction["encoding"],
            "page_count": extraction["page_count"],
            "empty_pages": extraction["empty_pages"],
            "pages": page_line_ranges(lines),
        },
        "char_count": len(raw_text),
        "non_blank_char_count": len(_non_blank(raw_text)),
        "line_count": len(lines),
        **turn_statistics(turns),
        "coverage": coverage,
        "warning_count": len(warnings),
        "warnings_by_severity": {
            s: sum(1 for w in warnings if w["severity"] == s) for s in ("error", "warning", "info")
        },
        "warnings": warnings,
        "note": (
            "Aucune mesure de précision : sans vérité terrain, l'exactitude de l'attribution "
            "des locuteurs n'est pas évaluée. La couverture mesure seulement la conservation du texte."
        ),
    }
