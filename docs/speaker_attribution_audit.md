# Étape 3.5 — Speaker Attribution Auditor

Une transcription peut attribuer un passage au mauvais locuteur : un tour
marqué `enqueteur` qui contient manifestement une réponse à la première
personne, une question de l'enquêteur attribuée à l'enquêté·e. L'auditeur
**signale** ces tours ; il ne corrige **jamais** rien.

```
SOURCE ORIGINALE          (data/inputs/<run_id>/, jamais modifiée)
      ↓
TRANSCRIPT STRUCTURÉ      (structured_transcript.json, jamais modifié)
      ↓
AUDIT                     A. présélection déterministe → B. un appel LLM au plus
      ↓
WARNINGS                  speaker_attribution_audit.json + speaker_warning transmis aux agents
```

L'auditeur n'a jamais le droit de réécrire `speaker`, `text`, `turn_id` ni
`source`. Ses suggestions ne sont **jamais appliquées automatiquement**, ni
par TRACE, ni par l'interface, ni par les agents. Le locuteur officiel reste
celui du transcript structuré.

Code : `core/speaker_attribution_auditor.py` (règles, extrait, schéma,
revalidation, avertissements) ; orchestration (appel, cache, manifest) :
`core/analysis.py → run_speaker_audit` ; consignes :
`prompts/speaker_attribution_auditor.md`.

## A. Présélection déterministe (sans IA, gratuite)

Seuls les tours `enqueteur` / `enquete` sont examinés (un tour `unknown` est
déjà signalé par l'ingestion). Le texte est comparé sans accents ni casse.
Chaque règle produit un **candidat**, jamais un changement de locuteur.

| Règle | Rôle évoqué | Condition (toutes requises) |
|---|---|---|
| `INTERVIEWER_FIRST_PERSON_ANSWER` | `enquete` | tour `enqueteur` ; pas une vraie question ; pas d'expression de cadrage d'entretien (« je vais te poser », « mon mémoire », « si je comprends bien », « merci »…) ; plus de marques de 1re personne (je, j', moi, me, m', mon, ma, mes) que de 2e (tu, te, toi, vous…) ; et soit ≥ 3 marques de 1re personne sur ≥ 12 mots, soit un marqueur de récit d'usage (« moi personnellement », « j'utilise », « je lui demande », « mes cours »…) avec ≥ 2 marques sur ≥ 6 mots |
| `CONSECUTIVE_INTERVIEWER_ANSWER` | `enquete` | deux tours `enqueteur` consécutifs ; le premier est une vraie question, le second ne l'est pas, contient au moins une marque de 1re personne (pas moins que de 2e) et ≥ 3 mots ; pas de cadrage |
| `INTERVIEWEE_INTERVIEW_QUESTION` | `enqueteur` | tour `enquete` ; vraie question ; aucune 1re personne ; au moins une 2e personne ; ≥ 4 mots ; et soit une ouverture typique (« est-ce que tu », « comment tu », « tu utilises »…), soit un tour précédent qui n'est pas une question, soit un tour suivant lui aussi `enquete` |
| `CONFLICTING_SPEAKER_MARKER` | — | le texte contient, en milieu de ligne après une fin de phrase, un marqueur reconnu de **l'autre** rôle (« … ? Enquêté : oui ») |

Une **vraie question** se termine par « ? » et n'est pas une simple question
de ponctuation du récit (« tu vois ? », « non ? », « tu vois ce que je veux
dire ? », « hein ? »…). Ainsi « Tu vois ce que je veux dire ? » n'est jamais
candidat, quel que soit le locuteur.

Chaque candidat porte : `rules`, `heuristic_suggestion` (rôle évoqué si les
règles concordent, sinon `null`) et `cues` : des passages **exacts** du tour
(première phrase, question finale ou marqueur), validés comme des citations.

## B. Audit LLM conditionnel : 0 ou 1 appel par entretien

- **Aucun candidat → aucun appel.** Le fichier d'audit est tout de même écrit
  (`audit_mode: "heuristics_only"`, `llm_called: false`, `items: []`).
- **Au moins un candidat → UN seul appel** pour tout l'entretien, quel que
  soit le nombre de candidats (au plus 40 transmis ; au-delà, les candidats
  restants sont signalés sans évaluation : `CANDIDATE_LIMIT_REACHED`).
- Le modèle ne reçoit qu'un **extrait** : les tours candidats et 2 tours
  voisins de part et d'autre (`turn_id`, `speaker`, `text`, `page`,
  `candidate`, `heuristics`), jamais l'entretien entier ni les autres entretiens.
- Même sécurité que les agents : extrait présenté comme une donnée, balise
  `</transcript>` neutralisée, sortie contrainte par un schéma JSON puis revalidée.

Le modèle répond, pour chaque candidat :

| Champ | Contenu |
|---|---|
| `turn_id` | tour candidat évalué |
| `suggested_speaker` | l'autre rôle si le texte l'indique clairement, sinon `null` (attribution plausible OU passage ambigu) |
| `confidence` | `high`, `medium`, `low` (en cas de doute : `low`) |
| `reason` | explication brève, fondée sur la forme du texte |
| `needs_review` | vérification humaine demandée |
| `evidence` | citations exactes `{turn_id, quote}` |

`current_speaker` n'est **pas** demandé au modèle : TRACE le lit dans le transcript.

## Revalidation déterministe (rien n'est corrigé)

| Code | Gravité | Sens |
|---|---|---|
| `UNKNOWN_TURN_ID` / `FOREIGN_INTERVIEW_TURN` | error | `turn_id` inexistant / d'un autre entretien (jamais transmis aux agents) |
| `QUOTE_NOT_FOUND`, `QUOTE_NOT_EXACT`, `EMPTY_QUOTE`, `NO_VALID_EVIDENCE` | error | mêmes contrôles de citation que pour les agents |
| `EVIDENCE_OUTSIDE_EXCERPT` | warning | citation d'un tour absent de l'extrait envoyé |
| `NOT_A_CANDIDATE` | warning | le modèle a évalué un tour non candidat |
| `CANDIDATE_NOT_ASSESSED` | warning | candidat absent de la réponse (ou non transmis, ou appel en échec) : conservé avec `suggested_speaker: null`, `confidence: "low"`, `source: "heuristics"` |
| `DUPLICATE_ASSESSMENT` | warning | plusieurs évaluations pour un même tour |
| `SUGGESTION_EQUALS_CURRENT` | warning | « suggestion » identique au locuteur actuel |
| `INTERPRETIVE_VOCABULARY` | warning | concept théorique ou jugement dans `reason` (garde-fou lexical) |
| `CANDIDATE_LIMIT_REACHED` | warning | plus de 40 candidats |
| `AUDIT_UNAVAILABLE` | warning | appel en échec : seuls les candidats déterministes sont signalés |
| `SUGGESTION_WITHOUT_REVIEW` | info | suggestion sans demande de vérification : TRACE la demande |

`needs_review` final = demande du modèle OU suggestion d'un autre locuteur OU
candidat non évalué OU anomalie `error`/`warning`.

## Sorties

```
analysis/
    speaker_attribution_audit.json    évaluations, citations annotées, anomalies
    speaker_audit_manifest.json       version, empreintes, modèle, cache, tokens, statut
```

```json
{
  "interview_id": "ENTRETIEN_ETAPE_3_5",
  "status": "SUCCESS",
  "notice": "Ces suggestions ne modifient pas la transcription originale : …",
  "transcript_modified": false,
  "structured_transcript_sha256": "…",
  "audit_mode": "heuristics_and_llm",
  "llm_called": true,
  "candidate_count": 1,
  "review_count": 1,
  "speaker_warning_count": 1,
  "items": [
    {
      "turn_id": "ENTRETIEN_ETAPE_3_5_T0008",
      "current_speaker": "enqueteur",
      "suggested_speaker": "enquete",
      "confidence": "high",
      "reason": "Réponse à la première personne qui suit directement la question du tour précédent.",
      "needs_review": true,
      "source": "llm",
      "heuristics": ["INTERVIEWER_FIRST_PERSON_ANSWER", "CONSECUTIVE_INTERVIEWER_ANSWER"],
      "evidence": [{"turn_id": "…_T0008", "quote": "…", "validation": {"valid": true, "match": "exact", "code": null}}],
      "review_reasons": []
    }
  ]
}
```

Statuts : `SUCCESS`, `SUCCESS_WITH_WARNINGS`, `CACHED`, `FAILED`. Un échec
de l'audit ne bloque jamais les agents. `structured_transcript_sha256` (aussi
dans le manifest) permet de vérifier après coup que la transcription n'a pas changé.

Le bilan de l'audit figure aussi dans `evidence_validation.json → speaker_audit`
(les totaux `total_*` de ce fichier restent ceux des deux agents).

## Transmission aux agents : `speaker_warning`

Les tours dont l'attribution reste douteuse (suggestion d'un autre locuteur,
demande de vérification, candidat non évalué) reçoivent, dans la
représentation envoyée aux **deux** agents, un champ secondaire :

```json
{"turn_id": "…_T0008", "speaker": "enqueteur", "text": "…",
 "speaker_warning": {"suggested_speaker": "enquete", "confidence": "high"}}
```

- `speaker` reste celui du transcript ; le message dit explicitement que le
  speaker officiel reste inchangé et que le warning signale seulement une
  attribution potentiellement douteuse, à ne jamais corriger.
- Seuls `suggested_speaker` et `confidence` passent : ni la raison, ni les
  règles, ni les citations de l'auditeur. Les deux agents reçoivent le même
  message et ne voient toujours pas la sortie l'un de l'autre.
- Une pratique ou un signal qui cite un tour signalé reçoit
  `EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN` (info) : il est marqué à revoir, sans
  que rien ne change dans sa citation ni dans le locuteur.

## Cache

Clé de l'audit : empreinte du fichier source, empreinte de l'**extrait**
envoyé, nom, version, prompt (consignes + gabarit) et schéma de l'auditeur,
modèle, paramètres. Un changement de l'auditeur relance l'audit seul ; les
agents ne sont relancés que si les avertissements qu'ils reçoivent changent
(leur entrée a alors réellement changé). Un appel en échec n'est pas mis en cache.

## Interface

Dans l'étape 3 : nombre de tours suspects et d'appels d'audit prévus avant le
lancement ; colonnes « Audit locuteurs », « Tours suspects », « Locuteurs à
vérifier » ; un panneau par entretien listant les avertissements, avec la
mention « Ces suggestions ne modifient pas la transcription originale. » et le
téléchargement de `speaker_attribution_audit.json`. Aucun contrôle de
l'interface ne permet de modifier la transcription.

## Limites

- Règles lexicales simples : des tours mal attribués sans marque de personne
  (« Oui. », « D'accord. ») ne sont pas détectés ; un enquêteur qui raconte sa
  propre expérience peut être signalé (l'appel au modèle sert à filtrer ces cas).
- Les tours `unknown` ne sont pas audités.
- Une transcription aux étiquettes inversées sur tout l'entretien produit de
  nombreux candidats : au-delà de 40, ils ne sont plus évalués par le modèle.
- Aucune « précision » n'est mesurée : sans vérité terrain, la qualité de
  l'audit n'est pas évaluée ; la vérification humaine reste nécessaire.
