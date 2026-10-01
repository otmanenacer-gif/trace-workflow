"""Constantes et vocabulaire partagés par la couche d'ingestion.

Ce module ne contient aucune logique : il fixe les valeurs autorisées
(locuteurs, statuts, codes d'avertissement) afin que les sorties JSON
restent stables pour les futurs agents. Voir docs/data_model.md.
"""

SCHEMA_VERSION = "1.0"

# Catégories de locuteurs (valeur du champ "speaker" d'un tour de parole)
SPEAKER_INTERVIEWER = "enqueteur"
SPEAKER_INTERVIEWEE = "enquete"
SPEAKER_UNKNOWN = "unknown"
SPEAKERS = (SPEAKER_INTERVIEWER, SPEAKER_INTERVIEWEE, SPEAKER_UNKNOWN)

# Statut final d'ingestion d'un entretien
STATUS_PASS = "PASS"
STATUS_PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
STATUS_FAIL = "FAIL"

# Gravité des avertissements :
# - info    : information technique, n'affecte pas le statut ;
# - warning : à vérifier manuellement, statut PASS_WITH_WARNINGS ;
# - error   : entretien non exploitable en l'état, statut FAIL.
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

# Code -> (gravité, description). Tout avertissement émis doit figurer ici.
WARNING_CODES = {
    # Extraction
    "UNSUPPORTED_FORMAT": (SEVERITY_ERROR, "Extension de fichier non prise en charge."),
    "EXTRACTION_FAILED": (SEVERITY_ERROR, "Le fichier n'a pas pu être lu (corrompu, chiffré ou invalide)."),
    "NO_TEXT_EXTRACTED": (SEVERITY_ERROR, "Aucun texte exploitable extrait (fichier vide ou PDF scanné) : vérification manuelle nécessaire."),
    "EMPTY_PDF_PAGES": (SEVERITY_WARNING, "Certaines pages du PDF ne contiennent aucun texte extractible (images, pages blanches ?)."),
    "ENCODING_FALLBACK": (SEVERITY_WARNING, "Le fichier TXT n'est pas en UTF-8 : un encodage de repli a été utilisé."),
    "DOCX_TABLE_FLATTENED": (SEVERITY_INFO, "Le DOCX contient des tableaux : chaque ligne a été extraite, cellules séparées par une tabulation."),
    "DOCX_UNKNOWN_ELEMENT": (SEVERITY_WARNING, "Le DOCX contient un élément inhabituel dont seul le texte brut a été récupéré."),
    # Structuration
    "NO_SPEAKER_LABEL_DETECTED": (SEVERITY_WARNING, "Aucun marqueur de locuteur reconnu : tout le texte est attribué à « unknown »."),
    "TEXT_BEFORE_FIRST_LABEL": (SEVERITY_WARNING, "Du texte précède le premier marqueur de locuteur : attribué à « unknown »."),
    "UNRECOGNIZED_SPEAKER_LABEL": (SEVERITY_WARNING, "Marqueur de locuteur récurrent mais non reconnu : tours attribués à « unknown »."),
    "POSSIBLE_INLINE_SPEAKER_LABEL": (SEVERITY_WARNING, "Un marqueur de locuteur semble apparaître au milieu d'une ligne : non découpé, à vérifier."),
    "INLINE_SPEAKER_LABELS_SPLIT": (SEVERITY_INFO, "Marqueurs explicites de locuteur en milieu de ligne : un nouveau tour a été ouvert à chacun (texte inchangé)."),
    "SINGLE_ROLE_ONLY": (SEVERITY_WARNING, "Un seul des deux rôles (enquêteur / enquêté) a été détecté."),
    "EMPTY_TURN": (SEVERITY_INFO, "Un marqueur de locuteur n'est suivi d'aucun texte."),
    # Contrôles de validation
    "CONTENT_MISMATCH_MINOR": (SEVERITY_WARNING, "Légère différence entre le texte extrait et les tours de parole."),
    "CONTENT_LOSS": (SEVERITY_ERROR, "Perte substantielle de texte entre l'extraction et la structuration."),
    "CONTENT_ADDED": (SEVERITY_ERROR, "Les tours de parole contiennent du texte absent de l'extraction."),
    "SOURCE_MODIFIED": (SEVERITY_ERROR, "L'empreinte SHA-256 du fichier a changé pendant le traitement."),
    "INVALID_TURN_SEQUENCE": (SEVERITY_ERROR, "Identifiants de tours non uniques ou non ordonnés."),
    "OVERSIZED_TURN_WITH_INTERNAL_MARKERS": (SEVERITY_ERROR, "Tour anormalement long contenant plusieurs marqueurs de locuteur internes : segmentation à vérifier, analyse IA bloquée."),
}


def make_warning(code: str, message: str | None = None, **context) -> dict:
    """Construit un avertissement normalisé : code, gravité, message, contexte.

    `context` accueille des informations de localisation (turn_id, page, line...).
    """
    severity, description = WARNING_CODES[code]
    warning = {"code": code, "severity": severity, "message": message or description}
    warning.update({k: v for k, v in context.items() if v is not None})
    return warning
