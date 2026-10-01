# Étape 3 — deux agents d'analyse indépendants

Cette étape ajoute le premier étage d'analyse par LLM. Deux agents lisent
**le même entretien structuré**, **séparément** et **en parallèle** :

```
structured_transcript.json
          │   (représentation compacte : turn_id, speaker, text, page)
     ┌────┴────┐
     │         │
     ▼         ▼
 PRACTICE    INTERACTION
 EXTRACTOR   SIGNAL READER
     │         │
     ▼         ▼
 practice_extractor.json      interaction_signals.json
 practice_manifest.json       interaction_manifest.json
          └────┬────┘
               ▼
   evidence_validation.json   (validation déterministe, sans IA)
```

Aucune fusion, aucune synthèse, aucune analyse théorique n'est produite ici :
l'interprétation relève d'une étape ultérieure.

## 1. Pourquoi deux agents indépendants

TRACE veut éviter qu'une problématique théorique très structurée force le
matériau à entrer dans des catégories prédéfinies. On sépare donc deux
lectures de nature différente :

| | Practice Extractor | Interaction Signal Reader |
|---|---|---|
| Question | Que fait l'étudiant·e, avec ou sans IAG ? | Comment le raconte-t-il ou elle ? |
| Unité | une pratique située (situation concrète) | un signal discursif observable |
| Consignes | `prompts/practice_extractor.md` | `prompts/interaction_signal_reader.md` |
| Schéma | `agents/practice_extractor.py` | `agents/interaction_signal_reader.py` |

L'indépendance est **structurelle** (et testée, `tests/test_analysis_pipeline.py`) :

- chaque appel contient uniquement : les consignes de l'agent (message système),
  son schéma de sortie, et un message utilisateur contenant l'entretien ;
- les deux messages utilisateur sont identiques ; aucun appel ne contient la
  sortie de l'autre agent (un seul message, pas d'historique) ;
- les deux appels sont lancés simultanément (`asyncio.gather`) : un test fait
  attendre chaque agent simulé jusqu'à ce que l'autre ait démarré, ce qui
  échouerait en exécution séquentielle ;
- un échec de l'un n'annule ni n'efface la sortie de l'autre.

## 2. Practice Extractor : volontairement « aveugle »

Ses consignes ne mentionnent **aucun** cadre théorique : ni accountability, ni
Garfinkel, ni breaching, ni réparation, ni « métier d'étudiant », ni régimes,
ni typologie. Un test vérifie que ces termes sont absents du prompt. Les
nommer — même pour les interdire — introduirait le cadre que l'on veut tenir à
distance ; leur absence dans les sorties est contrôlée **après coup** par un
garde-fou déterministe (§ 6).

Il décrit, au discours rapporté, ce que l'enquêté·e dit faire : usage,
non-usage, refus, usage hypothétique, usage passé. Il reprend les raisons
**données** par l'enquêté·e, jamais des motifs supposés.

> Mauvais : « L'étudiante utilise ChatGPT pour éviter l'effort intellectuel. »
> Bon : « L'étudiante indique utiliser ChatGPT pour obtenir un résumé lorsqu'elle dit ne pas avoir le temps de lire le texte. »

### Schéma d'une pratique (schema_version 1.1)

| Champ | Contenu |
|---|---|
| `practice_id` | `<INTERVIEW>_P001`… attribué par TRACE (pas par le modèle) |
| `summary` | phrase descriptive au discours rapporté (ajout au schéma initial) |
| `turn_start`, `turn_end` | premier / dernier tour de la situation |
| `use_status` | `use`, `non_use`, `refusal`, `hypothetical`, `past_use` |
| `academic_task`, `discipline` | tels que dits, sinon `null` |
| `context` | cadre de la situation tel que raconté |
| `ai_tool` | outils nommés |
| `student_action_before`, `ai_action`, `student_action_after` | déroulé |
| `stated_reason` | raisons données par l'enquêté·e |
| `explicit_constraints` | délais, interdictions, surveillance, consignes… |
| `verification_or_control` | contrôles déclarés sur le résultat |
| `stated_frequency` | fréquence uniquement (« parfois », « souvent », « rarement », « une fois », « jamais », « toujours »…), sinon `null` (ajout au schéma initial) |
| `scope_qualifier` | qualificatif de portée tel que dit (« surtout », « principalement »…), sinon `null` (ajout, schéma 1.1) |
| `assessment_context` | `graded`, `ungraded`, `exam`, `class`, `personal`, `unknown` |
| `other_actors` | autres personnes mentionnées |
| `evidence` | au moins une citation `{turn_id, quote}` |
| `explicitness` | `direct`, `strongly_supported`, `unclear` |
| `uncertainty_note` | incertitude, sinon `null` |

Niveau global : `extraction_notes` (remarques techniques, pas de synthèse).

`stated_frequency` ne contient qu'une fréquence : « je l'utilise surtout pour
reformuler » donne `stated_frequency: null` et `scope_qualifier: "surtout"`.
Une réponse qui place « surtout », « principalement », « essentiellement »,
« notamment » ou « en particulier » dans `stated_frequency` est rejetée à la
validation locale du schéma (`SCHEMA_VALIDATION`) : rien n'est enregistré ni
mis en cache.

## 3. Interaction Signal Reader : observer sans lire les pensées

Il relève des marques **présentes dans le texte transcrit** :
`explicit_emotion`, `hesitation`, `self_correction`, `self_reformulation`,
`minimization`, `intensification`, `restriction`, `exception`, `contrast`,
`cross_turn_contradiction`, `vocabulary_shift`, `modalization`,
`normative_formulation`, `generalization`, `reference_to_teacher_judgment`,
`reference_to_peer_judgment`, `reference_to_rule`,
`distancing_from_own_practice`, `attribution_to_others`,
`transcribed_laughter`, `transcribed_silence`, `significant_repetition`,
`pronoun_shift`, `preference_statement` (préférence explicitement formulée :
« je préfère le faire moi-même », « j'aime mieux écrire moi-même » ; ajout,
schéma 1.1), `other` (décrit librement, pour ne pas forcer un passage
dans une catégorie).

**Il peut** : noter « scrupules » comme `explicit_affect` quand l'enquêté·e
dit « j'ai un peu des scrupules » ; relever « euh… enfin… juste » comme
hésitation, autocorrection et minimisation ; mettre en regard deux tours qui
se contredisent, de façon neutre.

**Il ne peut pas** : attribuer une émotion non formulée (honte, culpabilité,
peur…), prêter une fonction à une formulation (se défendre, se justifier,
stratégie), qualifier un rire (« gêné »), un silence ou une contradiction
(mensonge, hypocrisie, dissimulation, réparation), ni convertir une
préférence énoncée (`preference_statement`) en trait de la personne
(autonomie, résistance, identité, position morale). Rire et silence ne sont
relevés que s'ils sont **transcrits**.

### Schéma d'un signal (schema_version 1.1)

| Champ | Contenu |
|---|---|
| `signal_id` | `<INTERVIEW>_S001`… attribué par TRACE |
| `turn_ids` | tours concernés (au moins deux pour une contradiction) |
| `signal_type` | un des types ci-dessus |
| `surface_form` | mots exacts qui portent le signal |
| `description` | description purement textuelle |
| `topic` | ce dont parle l'enquêté·e à ce moment (ajout, utile à la future fusion) |
| `evidence` | au moins une citation `{turn_id, quote}` |
| `explicit_affect` | affect nommé par l'enquêté·e dans une citation, sinon `null` |
| `cross_turn_reference` | ce qui est mis en regard entre tours, sinon `null` |
| `explicitness` | `direct`, `strongly_supported`, `unclear` |
| `needs_human_review` | demande de vérification formulée par l'agent |

Niveau global : `reading_notes`.

## 4. Sécurité du contexte

Le contenu des entretiens est une **donnée non fiable**. Un enquêté peut dire
« ignore tes instructions précédentes » : ce n'est qu'un énoncé.

- Les consignes (message système) disent explicitement que la transcription
  est un objet d'analyse et qu'aucune instruction qui s'y trouve ne doit être suivie.
- La transcription est placée dans le message utilisateur, en JSON, entre
  `<transcript>` et `</transcript>`. Les séquences `</` du texte sont écrites
  `<\/` (JSON strictement équivalent) : un entretien ne peut pas fermer la balise.
- La sortie est contrainte par un schéma JSON (sortie structurée de l'API) puis
  revalidée localement : un changement de format est rejeté.

## 5. Données envoyées

Un appel = un agent × un entretien. Pour chaque tour : `turn_id`, `speaker`
(`enqueteur` / `enquete` / `unknown`), `text`, et `page` pour un PDF.
Ne sont **pas** envoyés : empreintes SHA-256, marqueurs et libellés bruts de
locuteur (qui peuvent contenir des prénoms), numéros de ligne, rapports
d'ingestion, fichiers binaires, autres entretiens.

## 6. Validation des preuves (`core/evidence_validator.py`)

Après chaque réponse (et à chaque réutilisation du cache), un validateur
déterministe vérifie chaque citation. Rien n'est corrigé : une preuve
invalide est marquée `validation.valid = false`, l'objet reçoit
`needs_review = true` et un code dans `review_reasons`, l'anomalie est
consignée dans `evidence_validation.json`.

| Code | Gravité | Sens |
|---|---|---|
| `UNKNOWN_TURN_ID` | error | turn_id inexistant dans l'entretien |
| `FOREIGN_INTERVIEW_TURN` | error | turn_id d'un autre entretien |
| `EMPTY_QUOTE` | error | citation vide |
| `QUOTE_NOT_FOUND` | error | citation absente du tour cité (inventée, paraphrasée, autre tour, à cheval sur deux tours) |
| `QUOTE_NOT_EXACT` | error | citation retrouvée seulement en ignorant casse, blancs, apostrophes ou guillemets : non littérale, donc invalide |
| `INVALID_TURN_RANGE` | error | `turn_start` postérieur à `turn_end` |
| `UNKNOWN_RANGE_TURN` | error | `turn_start` / `turn_end` inexistant |
| `NO_VALID_EVIDENCE` | error | aucune citation valide pour l'objet |
| `EVIDENCE_OUTSIDE_RANGE` | warning | citation hors de l'intervalle de la pratique |
| `EVIDENCE_TURN_NOT_LISTED` | warning | citation d'un tour absent de `turn_ids` |
| `NO_INTERVIEWEE_EVIDENCE` | warning | seules des questions de l'enquêteur sont citées |
| `CONTRADICTION_SINGLE_TURN` | warning | contradiction appuyée sur moins de deux tours |
| `AFFECT_NOT_IN_QUOTES` | warning | `explicit_affect` absent des citations du signal |
| `INTERPRETIVE_VOCABULARY` | warning | vocabulaire interprétatif dans un champ rédigé par l'agent (§ 7) |
| `AGENT_FLAGGED_REVIEW` | info | l'agent demande une vérification |
| `EXPLICITNESS_UNCLEAR` | info | l'agent qualifie sa lecture d'incertaine |

Une citation est valide si elle est une **sous-chaîne exacte** du texte du tour
cité ; seule la normalisation Unicode NFC est appliquée (même texte, encodage
canonique différent).

## 7. Garde-fou contre le vocabulaire interprétatif (`core/interpretation_guard.py`)

Les champs rédigés par l'agent (jamais les citations) sont comparés, sans
accents ni casse, à trois listes : concepts théoriques réservés (accountability,
Garfinkel, breach, réparation, métier d'étudiant, identité…), jugements moraux
et diagnostics (triche, dépendance, culpabilité, honte, peur…), et — pour
l'Interaction Signal Reader — fonctions prêtées aux formulations (stratégie,
défensif, justification, mensonge, rire gêné…) et lectures de la personne
(autonomie, résistance, position morale). Un terme employé par
l'enquêté·e dans les citations du même objet n'est pas signalé. Le garde-fou
signale, il ne supprime rien.

## 8. Cache (`core/analysis_cache.py`)

Une analyse est réutilisée **uniquement** si sont identiques : SHA-256 du
fichier source, SHA-256 de la représentation envoyée, agent et version,
SHA-256 du prompt (consignes + gabarit du message), version et SHA-256 du
schéma, modèle, paramètres de génération (effort, température). La clé est le
SHA-256 de ces champs ; ils sont aussi recopiés dans l'entrée de cache et
revérifiés à la lecture (une entrée corrompue ou incohérente est ignorée).

- Emplacement : `data/cache/analysis/<agent>/<clé>.json`, **global** : recharger
  la page ou réimporter le même fichier dans un nouveau run ne repaie rien.
  Le cache contient des extraits d'entretiens : il est ignoré par git.
- Seules les réponses **valides** sont mises en cache (un échec est retenté au
  lancement suivant).
- La validation des preuves est recalculée à chaque réutilisation.
- La case « Forcer une nouvelle analyse » ignore le cache (inutile en usage courant).

Chaque exécution écrit un manifest par agent (`practice_manifest.json`,
`interaction_manifest.json`) :

```json
{
  "agent": "practice_extractor", "agent_version": "1.1", "schema_version": "1.1",
  "prompt_sha256": "…", "schema_sha256": "…", "source_sha256": "…", "transcript_sha256": "…",
  "model": "<ANTHROPIC_MODEL>", "request_params": {"effort": null, "temperature": null},
  "cache_key": "…", "cache_hit": false, "status": "SUCCESS",
  "created_at": "…", "run_at": "…", "api_calls": 1, "billed_this_run": true,
  "usage": {"input_tokens": 18432, "output_tokens": 3210,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
  "duration_seconds": 41.3, "response_model": "…", "request_id": "req_…", "stop_reason": "end_turn",
  "item_count": 6, "invalid_evidence_count": 0, "needs_review_count": 1, "error": null
}
```

## 9. Observabilité des tokens

Les tokens rapportés par l'API (`input_tokens`, `output_tokens`, tokens
écrits / lus dans le cache de prompt de l'API) sont enregistrés par appel, avec
le nombre de tentatives et la durée. Un appel en échec après réponse (JSON
invalide, réponse tronquée…) reste compté : les tokens ont été consommés.
L'interface affiche, pour la dernière exécution :
« 2 appel(s) API — 18 432 tokens entrée — 3 210 tokens sortie — 0 résultat(s)
repris du cache TRACE ». **Aucun montant** n'est calculé : le prix du modèle
n'est pas connu localement de façon fiable.

## 10. Statuts et erreurs

`PENDING` → `RUNNING` → `SUCCESS` | `SUCCESS_WITH_WARNINGS` (au moins une
anomalie `error`/`warning` de validation) | `FAILED` | `CACHED`.

Erreurs API traduites en messages clairs (`core/llm_client.py`) : clé refusée
(401), accès refusé (403), modèle introuvable (404), requête refusée (400, avec
le détail de l'API, clé masquée), limite de débit (429), erreur serveur (5xx),
délai dépassé, réseau, refus du modèle, réponse tronquée, JSON invalide,
schéma non respecté. Réessais : uniquement 408/409/429/5xx, délai et réseau,
au plus `TRACE_LLM_MAX_RETRIES` fois (défaut 2), avec backoff exponentiel
borné (2 s, 4 s… 30 s max, `retry-after` respecté jusqu'à 60 s). Jamais de
boucle infinie ; une réponse invalide n'est jamais réessayée automatiquement
(pour ne pas repayer). Les journaux ne contiennent ni clé, ni transcript, ni
citation : identifiant d'entretien, agent, statut, tokens, durée.

## 11. Configuration et premier appel réel

1. `cp .env.example .env` puis renseigner `ANTHROPIC_API_KEY` (clé de la
   console Anthropic) et `ANTHROPIC_MODEL` (par exemple `claude-opus-5-5`).
   Le modèle n'est écrit nulle part ailleurs.
2. Optionnel : `TRACE_MAX_CONCURRENCY` (défaut 2), `TRACE_LLM_EFFORT`,
   `TRACE_LLM_MAX_TOKENS`, `TRACE_LLM_TIMEOUT_SECONDS`, `TRACE_LLM_MAX_RETRIES`.
   `TRACE_LLM_TEMPERATURE` n'est envoyé que s'il est renseigné : les modèles
   récents refusent les paramètres d'échantillonnage (erreur 400). La
   reproductibilité repose donc d'abord sur le cache (mêmes entrées → même
   sortie enregistrée) et sur le versionnage des prompts et schémas.
3. Test réel sur l'entretien **synthétique** (2 appels, consomme des tokens) :
   `python scripts/smoke_test_stage3.py --confirm-api-cost`. Sans cette option,
   le script n'appelle rien.
4. Puis, dans l'interface : importer un entretien, lancer l'ingestion, choisir
   « Test — un entretien » et cliquer sur « Lancer les deux analyses IA ».

Le repli automatique vers un autre modèle en cas de refus (paramètre
`fallbacks` de l'API) n'est **pas** activé : il changerait silencieusement le
modèle qui produit l'analyse, alors que le modèle fait partie de la clé de
cache et du manifest. Un refus est signalé comme `FAILED` (`REFUSAL`).

## 12. Tests

Aucun test automatique n'appelle l'API : `tests/conftest.py` retire les
variables `ANTHROPIC_*`, n'ouvre pas `.env`, remplace le transport HTTP réel
par une classe qui échoue, refuse toute connexion réseau TCP et redirige le
cache vers un dossier temporaire. Le LLM est simulé par
`tests/fake_llm.FakeTransport` ; l'entretien fictif et les réponses simulées
sont dans `tests/synthetic_interviews.py`.

- `tests/test_llm_client.py` — configuration, 429/5xx/délai, réessais bornés, réponses invalides, secret masqué
- `tests/test_practice_extractor.py`, `tests/test_interaction_signal_reader.py` — schémas, prompts, preuves
- `tests/test_evidence_validator.py` — citations exactes / inventées / autre entretien / intervalle inversé
- `tests/test_analysis_cache.py` — hit / prompt, version, modèle, paramètres ou transcript changés
- `tests/test_analysis_pipeline.py` — indépendance, parallélisme, isolement des échecs, multi-entretiens
- `tests/test_stage3_synthetic.py` — scénario qualitatif synthétique
- `tests/test_app_stage3.py` — interface (Streamlit AppTest)
- `tests/e2e/browser_check.py` — navigateur réel (Playwright), lancé à la main

## 13. Limites connues

- Les tests valident la chaîne et les garde-fous, **pas** la qualité d'un vrai
  modèle : la lecture humaine des sorties reste indispensable.
- Le garde-fou lexical est simple (listes de radicaux) : faux positifs possibles
  (« dépendant de la matière »), et une interprétation formulée sans ces mots
  n'est pas détectée.
- Une citation exacte mais trop courte (« oui ») est valide techniquement
  sans être forcément probante.
- Un entretien très long est envoyé en un seul appel (pas de découpage) ; au-delà
  de la limite de réponse, l'analyse échoue avec `TRUNCATED`.
- Les tours de locuteur `unknown` sont transmis tels quels : l'agent ne sait
  pas toujours qui parle.
- Le cache de prompt de l'API ne s'active que si les consignes dépassent la
  taille minimale du modèle : sinon, aucun gain (et aucun surcoût).
