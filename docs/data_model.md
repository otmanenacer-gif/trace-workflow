# Modèle de données de l'ingestion (schema_version 1.0)

Ce document décrit ce que la couche d'ingestion transmet aux futurs agents.
Tout est produit de façon **déterministe**, sans IA.

## Emplacement

```
data/outputs/<run_id>/
    metadata.json                       run : fichiers, état du pipeline, résumé d'ingestion
    interviews/<interview_id>/
        raw_text.txt                    texte brut extrait
        structured_transcript.json      tours de parole
        ingestion_report.json           rapport de qualité
data/inputs/<run_id>/                   copies des fichiers originaux (jamais modifiées)
```

`interview_id` est dérivé du nom de fichier : majuscules, sans accents,
caractères non alphanumériques remplacés par `_` (40 caractères au plus).
`Éloïse_Franzmann.pdf` → `ELOISE_FRANZMANN`. En cas de doublon dans un
même run : `ELOISE_2`, `ELOISE_3`…

## Trois niveaux conservés distinctement

| Niveau | Où | Garantie |
|---|---|---|
| A. Fichier source | `data/inputs/<run_id>/` | copie octet pour octet, SHA-256 revérifié après traitement |
| B. Texte brut | `raw_text.txt` | texte extrait tel quel (seules transformations : fins de ligne `\r\n`/`\r` → `\n`, BOM retiré) |
| C. Tours de parole | `structured_transcript.json` | texte recopié sans correction ; seul le marqueur de locuteur est retiré (et conservé dans `marker`) |

Les lignes de `raw_text.txt` sont numérotées à partir de 1 : `line_start` et
`line_end` d'un tour renvoient directement à ces lignes. Pour un PDF, les
pages sont concaténées dans l'ordre et `ingestion_report.json →
extraction.pages` donne les lignes correspondant à chaque page.

## `structured_transcript.json`

```json
{
  "schema_version": "1.0",
  "interview_id": "ELOISE",
  "source": {
    "filename": "Eloise.pdf",
    "stored_name": "Eloise.pdf",
    "sha256": "…",
    "format": "pdf",
    "page_count": 12
  },
  "raw_text_file": "raw_text.txt",
  "status": "PASS_WITH_WARNINGS",
  "turn_count": 214,
  "turns": [
    {
      "turn_id": "ELOISE_T0001",
      "index": 1,
      "speaker": "enqueteur",
      "speaker_raw": "Enquêteur",
      "marker": "Enquêteur :",
      "text": "Est-ce que tu peux te présenter ?",
      "source": {"file": "Eloise.pdf", "page": 1, "page_end": 1, "line_start": 3, "line_end": 3}
    }
  ],
  "warnings": []
}
```

Champs d'un tour :

| Champ | Sens |
|---|---|
| `turn_id` | `<interview_id>_T<n>` sur 4 chiffres au moins ; unique, ordonné, stable pour un même fichier |
| `index` | position (1, 2, 3…) |
| `speaker` | `enqueteur`, `enquete` ou `unknown` — jamais deviné |
| `speaker_raw` | libellé tel qu'écrit dans la transcription (`"Participant 2"`, `"Eloïse"`), `null` si aucun |
| `marker` | préfixe exact retiré de la ligne (`"Enquêteur :"`), `null` si aucun |
| `text` | texte du tour, **non corrigé** (hésitations, répétitions, oralité, grossièretés conservées) ; seuls les blancs en début/fin sont retirés |
| `source.file` | nom du fichier dans `data/inputs/<run_id>/` |
| `source.page`, `page_end` | pages PDF de début et de fin (`null` pour TXT/DOCX) |
| `source.line_start`, `line_end` | lignes de `raw_text.txt` (première et dernière ligne non vide) |

## Règles de détection des locuteurs

1. **Marqueur reconnu** : une ligne qui *commence* par l'un des libellés
   ci-dessous (casse indifférente), éventuellement suivi d'un numéro
   (`Participant 2`) ou d'une précision entre parenthèses
   (`Enquêteur (Marie)`), puis de `:`.
   - `enqueteur` : Enquêteur, Enqueteur, Enquêteuse, Enquêtrice, Intervieweur, Intervieweuse, Interviewer, Question, Q
   - `enquete` : Enquêté(e), Enquete(e), Interviewé(e), Interviewee, Participant(e), Réponse, Reponse, R
   - « Enquête » (nom commun) n'est **pas** un libellé : c'est souvent un titre.

   1 bis. **Marqueur explicite en milieu de ligne** (correctif « paragraphes ») : un
   paragraphe DOCX ou une ligne de PDF peut contenir toute une suite d'échanges
   (« … ? Enquêté : Oui. Enquêteur : Et pour… »). Un libellé **sans ambiguïté**
   (Enquêteur / Enquêté / Interviewer / Interviewé et leurs variantes de casse,
   d'accents et de genre ; jamais Q, R, Question, Réponse, Participant) suivi de `:`
   ouvre alors aussi un nouveau tour, s'il suit une fin de phrase (`. ! ? …`,
   guillemet, parenthèse) ou s'il commence par une majuscule après un blanc.
   La ligne est coupée juste avant le libellé : le texte est recopié à l'identique,
   chaque tour garde le numéro de ligne d'origine, le libellé écrit décide seul du
   locuteur (un libellé qui semble faux n'est pas corrigé : l'audit des locuteurs le
   signale). Avertissement `INLINE_SPEAKER_LABELS_SPLIT` (info). Sans LLM.
2. **Marqueur récurrent non reconnu** : 1 à 3 mots à majuscule initiale suivis
   de `:` en début de ligne, présents au moins 2 fois (ex. `Eloïse :`) →
   tour `unknown`, libellé conservé dans `speaker_raw`, avertissement.
3. **Continuation** : toute autre ligne prolonge le tour en cours. Un `:`
   au milieu d'une ligne (« il m'a dit : … », « 14:30 », « le rôle de l'enquêteur : … »)
   ne change jamais de locuteur ; seule la règle 1 bis découpe en milieu de ligne.
4. **Texte sans marqueur** (avant le premier marqueur, ou document sans
   marqueur) : segments `unknown` découpés aux lignes vides.

## `ingestion_report.json`

| Champ | Sens |
|---|---|
| `status` | `PASS`, `PASS_WITH_WARNINGS` ou `FAIL` (voir ci-dessous) |
| `needs_manual_review` | `true` dès que le statut n'est pas `PASS` |
| `source` | nom, format, type MIME, taille, SHA-256, `sha256_verified_after_processing` |
| `extraction` | bibliothèque utilisée, encodage (TXT), nombre de pages, pages vides, lignes par page |
| `char_count`, `non_blank_char_count`, `line_count` | volume de texte extrait |
| `turn_count`, `interviewer_turn_count`, `interviewee_turn_count`, `unknown_segment_count` | comptes de tours |
| `attributed_text_ratio` | part approximative du texte (hors blancs) attribuée à `enqueteur` ou `enquete` |
| `coverage` | contrôle de conservation du texte (ci-dessous) |
| `warnings` | liste `{code, severity, message, …contexte}` |

**Aucune « précision » n'est mesurée** : sans vérité terrain, l'exactitude de
l'attribution des locuteurs n'est pas évaluée.

### Contrôle de couverture

Tous les blancs sont retirés (1) du texte brut, (2) de la concaténation
`marker + text` de tous les tours. Les deux chaînes doivent être identiques
(`exact_match: true`). Sinon, les suites de mots sont alignées pour mesurer
`token_coverage_ratio` (part des mots bruts retrouvés dans l'ordre),
`missing_token_count` et `added_token_count`.

- couverture < 99 % → `CONTENT_LOSS` (FAIL) ;
- plus de 1 % de mots ajoutés → `CONTENT_ADDED` (FAIL) ;
- différence plus faible → `CONTENT_MISMATCH_MINOR` (PASS_WITH_WARNINGS).

### Statut

- `FAIL` : au moins un avertissement de gravité `error` ;
- `PASS_WITH_WARNINGS` : au moins un `warning` ;
- `PASS` : uniquement des `info` ou rien.

### Codes d'avertissement

| Code | Gravité | Signification |
|---|---|---|
| `UNSUPPORTED_FORMAT` | error | extension non prise en charge |
| `EXTRACTION_FAILED` | error | fichier corrompu, chiffré ou invalide |
| `NO_TEXT_EXTRACTED` | error | aucun texte (fichier vide, PDF scanné) — vérification manuelle |
| `CONTENT_LOSS` | error | perte substantielle de texte |
| `CONTENT_ADDED` | error | texte absent de l'extraction dans les tours |
| `SOURCE_MODIFIED` | error | SHA-256 du fichier modifié pendant le traitement |
| `INVALID_TURN_SEQUENCE` | error | identifiants non uniques ou désordonnés |
| `OVERSIZED_TURN_WITH_INTERNAL_MARKERS` | error | tour de plus de 4 000 caractères contenant au moins 2 marqueurs de locuteur internes : segmentation manquée, l'entretien n'est jamais envoyé aux agents IA |
| `EMPTY_PDF_PAGES` | warning | pages PDF sans texte extractible |
| `ENCODING_FALLBACK` | warning | TXT non UTF-8, décodé en cp1252 ou latin-1 |
| `DOCX_UNKNOWN_ELEMENT` | warning | élément DOCX inhabituel récupéré en texte brut |
| `NO_SPEAKER_LABEL_DETECTED` | warning | aucun marqueur : tout est `unknown` |
| `TEXT_BEFORE_FIRST_LABEL` | warning | texte (souvent un en-tête) avant le premier marqueur |
| `UNRECOGNIZED_SPEAKER_LABEL` | warning | libellé récurrent non reconnu (ex. un prénom) |
| `POSSIBLE_INLINE_SPEAKER_LABEL` | warning | marqueur ambigu (Q, R, Question, Réponse, Participant) en milieu de ligne, non découpé |
| `INLINE_SPEAKER_LABELS_SPLIT` | info | marqueurs explicites en milieu de ligne : un tour ouvert à chacun (règle 1 bis), texte inchangé |
| `SINGLE_ROLE_ONLY` | warning | un seul des deux rôles détecté |
| `CONTENT_MISMATCH_MINOR` | warning | légère différence de contenu |
| `DOCX_TABLE_FLATTENED` | info | tableaux DOCX extraits ligne par ligne |
| `EMPTY_TURN` | info | marqueur sans texte |

La liste de référence est `core/schemas.py → WARNING_CODES`.

## Ce qui est transmis aux agents de l'étape 3

Les agents ne reçoivent pas ces fichiers tels quels, mais une représentation
compacte d'UN entretien : `interview_id`, puis pour chaque tour `turn_id`,
`speaker`, `text` et, pour un PDF, `page`. Voir
[`docs/agents_stage3.md`](agents_stage3.md#5-données-envoyées).

Depuis l'étape 3.5, un tour dont l'attribution du locuteur paraît douteuse
peut aussi porter `speaker_warning` (`suggested_speaker`, `confidence`),
produit par l'auditeur des locuteurs. **`structured_transcript.json` n'est
jamais modifié** : `speaker`, `text`, `turn_id` et `source` restent ceux de
l'ingestion, et le locuteur officiel reste `speaker`. Voir
[`docs/speaker_attribution_audit.md`](speaker_attribution_audit.md).
