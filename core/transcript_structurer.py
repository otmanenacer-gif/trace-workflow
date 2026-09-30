"""Découpage déterministe d'une transcription en tours de parole.

Aucune inférence sémantique : seules les règles ci-dessous sont appliquées.

Règle 1 — marqueur reconnu.
    Une ligne qui COMMENCE par un libellé de KNOWN_LABELS (casse indifférente),
    éventuellement suivi d'un numéro (« Participant 2 ») ou d'une précision
    entre parenthèses (« Enquêteur (Marie) »), puis de « : », ouvre un
    nouveau tour. Le libellé détermine le locuteur (enqueteur / enquete).
    Un « : » ailleurs dans la ligne n'est jamais un changement de locuteur.

Règle 2 — marqueur récurrent non reconnu.
    Une ligne qui commence par 1 à 3 mots à majuscule initiale suivis de « : »
    (ex. « Eloïse : ») ouvre un tour « unknown » SI ce même libellé ouvre au
    moins MIN_UNKNOWN_LABEL_OCCURRENCES lignes du document. Le libellé est
    conservé dans speaker_raw, jamais converti en rôle. Avertissement émis.

Règle 3 — continuation.
    Toute autre ligne prolonge le tour en cours (prise de parole multilignes).

Règle 4 — texte sans marqueur.
    Le texte qui précède le premier marqueur (ou tout le texte s'il n'y en a
    aucun) est conservé en segments « unknown », découpés aux lignes vides.

Transformations techniques (et seulement celles-ci) :
- le marqueur (ex. « Enquêteur : ») est retiré du texte et conservé dans
  le champ « marker » du tour ;
- les espaces et lignes vides en début et fin de tour sont retirés ;
- le texte interne d'un tour est recopié à l'identique (sauts de ligne compris).
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

from core.schemas import (
    SPEAKER_INTERVIEWEE,
    SPEAKER_INTERVIEWER,
    SPEAKER_UNKNOWN,
    make_warning,
)

# Libellés reconnus (comparés en minuscules, après normalisation Unicode NFC).
# « Enquête » (le nom commun) est volontairement absent : c'est souvent un titre.
KNOWN_LABELS = {
    SPEAKER_INTERVIEWER: (
        "enquêteur", "enqueteur", "enquêteuse", "enqueteuse", "enquêtrice", "enquetrice",
        "intervieweur", "intervieweuse", "interviewer", "question", "q",
    ),
    SPEAKER_INTERVIEWEE: (
        "enquêté", "enquêtée", "enqueté", "enquetée", "enquete", "enquetee",
        "interviewé", "interviewée", "interviewe", "interviewee",
        "participant", "participante", "réponse", "reponse", "r",
    ),
}
LABEL_TO_SPEAKER = {label: speaker for speaker, labels in KNOWN_LABELS.items() for label in labels}

MIN_UNKNOWN_LABEL_OCCURRENCES = 2

_WORD = r"[^\W\d_][\w'’\-.]*"
_QUALIFIER = r"(?P<qual>[ \t]*(?:\d{1,3}|\([^()\n]{1,40}\)))?"
_COLON = r"[ \t]*[:：]"
# Règle 1 : un seul mot en tête de ligne, puis qualificatif facultatif, puis « : »
KNOWN_LABEL_RE = re.compile(rf"^[ \t]*(?P<label>{_WORD}){_QUALIFIER}{_COLON}")
# Règle 2 : 1 à 3 mots (la majuscule initiale est vérifiée en Python)
CANDIDATE_LABEL_RE = re.compile(rf"^[ \t]*(?P<label>{_WORD}(?:[ \t]+{_WORD}){{0,2}}){_QUALIFIER}{_COLON}")
# Marqueur reconnu en milieu de ligne, après une fin de phrase (signalé, jamais découpé)
_KNOWN_ALTERNATION = "|".join(sorted(map(re.escape, LABEL_TO_SPEAKER), key=len, reverse=True))
INLINE_LABEL_RE = re.compile(
    rf"(?<=[.!?…»\"”])[ \t]+(?P<label>{_KNOWN_ALTERNATION}){_QUALIFIER}{_COLON}", re.IGNORECASE
)


def make_interview_id(filename: str) -> str:
    """Identifiant d'entretien lisible et stable, dérivé du nom de fichier.

    « Éloïse_Franzmann.pdf » -> « ELOISE_FRANZMANN ». Accents retirés,
    caractères non alphanumériques remplacés par « _ », 40 caractères au plus.
    """
    stem = unicodedata.normalize("NFKD", Path(filename).stem)
    ascii_stem = "".join(c for c in stem if not unicodedata.combining(c))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_stem).strip("_").upper()[:40].strip("_")
    return slug or "ENTRETIEN"


def pages_to_lines(pages: list[dict]) -> list[tuple[int | None, str]]:
    """Aplatit les pages extraites en lignes (numéro de page, texte).

    raw_text.txt est exactement « \\n ».join(textes) : le numéro de ligne
    (1-indexé) d'un tour renvoie donc directement à raw_text.txt.
    """
    lines = []
    for page in pages:
        parts = page["text"].split("\n")
        if len(parts) > 1 and parts[-1] == "":
            parts.pop()  # saut de ligne final de la page
        lines.extend((page["page"], part) for part in parts)
    return lines


def _nfc_prefix_to_original(line: str, nfc_length: int) -> int:
    """Longueur du préfixe de `line` correspondant à `nfc_length` caractères en NFC."""
    for end in range(len(line) + 1):
        if len(unicodedata.normalize("NFC", line[:end])) >= nfc_length:
            return end
    return len(line)


def _split_marker(line: str, match: re.Match, probe: str) -> tuple[str, str]:
    """Sépare (marqueur, reste) dans la ligne d'origine, même si elle n'est pas en NFC."""
    end = match.end() if probe == line else _nfc_prefix_to_original(line, match.end())
    return line[:end].strip(), line[end:]


def _candidate_label(probe: str) -> str | None:
    """Libellé candidat de la règle 2 (mots à majuscule initiale), ou None."""
    match = CANDIDATE_LABEL_RE.match(probe)
    if not match:
        return None
    label = match.group("label")
    if not all(word[0].isupper() for word in label.split()):
        return None
    return label


def match_speaker_label(line: str, recurrent_unknown: set[str] = frozenset()) -> dict | None:
    """Applique les règles 1 et 2 à une ligne. Renvoie None si ce n'est pas un marqueur."""
    probe = unicodedata.normalize("NFC", line)
    match = KNOWN_LABEL_RE.match(probe)
    if match and match.group("label").casefold() in LABEL_TO_SPEAKER:
        speaker = LABEL_TO_SPEAKER[match.group("label").casefold()]
    else:
        label = _candidate_label(probe)
        if label is None or label.casefold() not in recurrent_unknown:
            return None
        match = CANDIDATE_LABEL_RE.match(probe)
        speaker = SPEAKER_UNKNOWN
    marker, rest = _split_marker(line, match, probe)
    speaker_raw = re.sub(r"[ \t]*[:：]$", "", marker)
    return {"speaker": speaker, "speaker_raw": speaker_raw, "marker": marker, "rest": rest}


def _recurrent_unknown_labels(lines: list[tuple[int | None, str]]) -> Counter:
    """Compte les libellés candidats (règle 2) qui ne sont pas des libellés connus."""
    counts = Counter()
    for _, text in lines:
        probe = unicodedata.normalize("NFC", text)
        known = KNOWN_LABEL_RE.match(probe)
        if known and known.group("label").casefold() in LABEL_TO_SPEAKER:
            continue
        label = _candidate_label(probe)
        if label:
            counts[label.casefold()] += 1
    return Counter({k: v for k, v in counts.items() if v >= MIN_UNKNOWN_LABEL_OCCURRENCES})


def structure_transcript(pages: list[dict], interview_id: str, source_file: str) -> dict:
    """Découpe le texte extrait en tours de parole.

    Renvoie {"turns": [...], "warnings": [...], "lines": [...]} où `lines`
    est la liste (page, texte) qui sert aussi à écrire raw_text.txt.
    """
    lines = pages_to_lines(pages)
    unknown_counts = _recurrent_unknown_labels(lines)
    recurrent = set(unknown_counts)

    segments = []  # tours bruts, avant numérotation
    current = None

    def close():
        if current is not None:
            segments.append(current)

    for number, (page, text) in enumerate(lines, start=1):
        label = match_speaker_label(text, recurrent)
        if label:
            close()
            current = {**label, "labeled": True, "lines": [(number, page, label["rest"])]}
        elif current is not None and current["labeled"]:
            current["lines"].append((number, page, text))  # règle 3
        elif not text.strip():
            close()  # règle 4 : une ligne vide sépare les segments sans marqueur
            current = None
        elif current is None:
            current = {"speaker": SPEAKER_UNKNOWN, "speaker_raw": None, "marker": None,
                       "labeled": False, "lines": [(number, page, text)]}
        else:
            current["lines"].append((number, page, text))
    close()

    turns, warnings = [], []
    width = max(4, len(str(len(segments))))
    for index, segment in enumerate(segments, start=1):
        content = [(n, p) for n, p, t in segment["lines"] if t.strip()]
        first_n, first_p = content[0] if content else segment["lines"][0][:2]
        last_n, last_p = content[-1] if content else (first_n, first_p)
        turn = {
            "turn_id": f"{interview_id}_T{index:0{width}d}",
            "index": index,
            "speaker": segment["speaker"],
            "speaker_raw": segment["speaker_raw"],
            "marker": segment["marker"],
            "text": "\n".join(t for _, _, t in segment["lines"]).strip(),
            "source": {
                "file": source_file,
                "page": first_p,
                "page_end": last_p,
                "line_start": first_n,
                "line_end": last_n,
            },
        }
        turns.append(turn)
        if segment["labeled"] and not turn["text"]:
            warnings.append(make_warning("EMPTY_TURN", turn_id=turn["turn_id"], line=first_n))
        for n, p, t in segment["lines"]:
            inline = INLINE_LABEL_RE.search(unicodedata.normalize("NFC", t))
            if inline:
                warnings.append(make_warning(
                    "POSSIBLE_INLINE_SPEAKER_LABEL",
                    f"Marqueur « {inline.group('label')} : » en milieu de ligne {n} : tour non découpé, à vérifier.",
                    turn_id=turn["turn_id"], line=n, page=p,
                ))

    labeled = [t for t in turns if t["marker"] is not None]
    unlabeled = [t for t in turns if t["marker"] is None]
    if turns and not labeled:
        warnings.insert(0, make_warning("NO_SPEAKER_LABEL_DETECTED"))
    elif unlabeled:
        warnings.insert(0, make_warning(
            "TEXT_BEFORE_FIRST_LABEL",
            f"{len(unlabeled)} segment(s) avant le premier marqueur de locuteur, attribué(s) à « unknown ».",
            turn_ids=[t["turn_id"] for t in unlabeled],
        ))
    for label, count in sorted(unknown_counts.items()):
        ids = [t["turn_id"] for t in labeled
               if t["speaker"] == SPEAKER_UNKNOWN
               and unicodedata.normalize("NFC", t["speaker_raw"]).casefold().startswith(label)]
        shown = next((t["speaker_raw"] for t in turns if t["turn_id"] in ids[:1]), label)
        warnings.append(make_warning(
            "UNRECOGNIZED_SPEAKER_LABEL",
            f"Libellé « {shown} » ({count} occurrences) non reconnu : tours attribués à « unknown ».",
            label=shown, count=count, turn_ids=ids[:20],
        ))
    roles = {t["speaker"] for t in labeled} - {SPEAKER_UNKNOWN}
    if len(roles) == 1:
        warnings.append(make_warning(
            "SINGLE_ROLE_ONLY", f"Seul le rôle « {roles.pop()} » a été détecté."
        ))
    return {"turns": turns, "warnings": warnings, "lines": lines}
