# Étape 3 — deux agents d'analyse indépendants

> **Exécution** : les agents sont joués par Claude Code (workflow multi-agents, [`TRACE_WORKFLOW.md`](../TRACE_WORKFLOW.md)),
> sans aucun appel à une API de modèle. Dans ce document, un « appel » désigne une **tâche d'agent** du workflow
> (un paquet `task.json` → une réponse `response.json`).

Cette étape ajoute le premier étage d'analyse par LLM. Deux agents lisent
**le même entretien structuré**, **séparément** et **en parallèle**, après un
audit de l'attribution des locuteurs (étape 3.5, voir
[`speaker_attribution_audit.md`](speaker_attribution_audit.md)) :

```
structured_transcript.json                (jamais modifié)
          │
          ▼
 SPEAKER ATTRIBUTION AUDIT   règles déterministes → 0 appel LLM si aucun tour suspect,
          │                  sinon 1 appel sur un extrait (candidats + voisins)
          │   speaker_attribution_audit.json, speaker_audit_manifest.json
          │   (représentation compacte : turn_id, speaker, text, page, speaker_warning éventuel)
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
  son schéma de sortie, et un message utilisateur contenant l'entretien (avec,
  le cas échéant, les `speaker_warning` de l'auditeur des locuteurs — jamais la
  sortie de l'autre agent) ;
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
**données** par l'enquêté·e, jamais des motifs supposés. Il décrit aussi les
usages personnels, professionnels et quotidiens (champ `practice_domain`) : ils
ne sont pas filtrés.

### Usages ET non-usages / refus (version 1.2)

Le premier entretien réel a montré que les affirmations de non-usage étaient
traitées comme de simples nuances d'un usage. Les consignes demandent
désormais de chercher **systématiquement** les deux :

- `non_use` : l'enquêté·e décrit une situation où il ou elle n'utilise pas
  l'IAG (« pour mes plans je le fais moi-même », « je n'utilise jamais ChatGPT
  en examen », « pour cette matière je ne l'utilise pas ») ;
- `refusal` : l'enquêté·e formule explicitement une limite, une règle ou un
  refus de déléguer (« je veux pas qu'il fasse mes travaux », « je lui demande
  pas de réfléchir à ma place ») ;
- un même thème peut donner **plusieurs pratiques contradictoires** : « Il
  m'arrive de lui demander de rédiger » puis « Mais normalement mes devoirs je
  les écris moi-même » donnent deux pratiques (`use` et `non_use`), jamais
  fusionnées ; la contradiction n'est pas résolue (elle sera interprétée plus tard) ;
- un usage passé puis arrêté donne `past_use` + `non_use`.

`non_use_reason` (renseigné seulement pour `non_use` / `refusal`) distingue ce
sur quoi l'enquêté·e fait reposer le non-usage, tel qu'il ou elle le
formule : `not_stated` (non-usage simplement décrit), `preference`,
`personal_rule` (règle revendiquée), `external_rule` (interdiction, consigne),
`technical_limitation` (impossibilité technique), `other`. Ce champ était
nécessaire : `use_status` n'a que deux valeurs pour ces quatre formes que la
consigne demande de ne pas confondre, et `stated_reason` est un texte libre.
Une incohérence (raison renseignée pour un usage, absente pour un non-usage)
est signalée par `NON_USE_REASON_MISMATCH`, jamais corrigée.

> Mauvais : « L'étudiante utilise ChatGPT pour éviter l'effort intellectuel. »
> Bon : « L'étudiante indique utiliser ChatGPT pour obtenir un résumé lorsqu'elle dit ne pas avoir le temps de lire le texte. »

### Schéma d'une pratique (schema_version 1.2)

| Champ | Contenu |
|---|---|
| `practice_id` | `<INTERVIEW>_P001`… attribué par TRACE (pas par le modèle) |
| `summary` | phrase descriptive au discours rapporté (ajout au schéma initial) |
| `turn_start`, `turn_end` | premier / dernier tour de la situation |
| `use_status` | `use`, `non_use`, `refusal`, `hypothetical`, `past_use` |
| `non_use_reason` | pour `non_use` / `refusal` : `not_stated`, `preference`, `personal_rule`, `external_rule`, `technical_limitation`, `other` ; sinon `null` (ajout, schéma 1.2) |
| `practice_domain` | `academic`, `personal`, `professional`, `mixed`, `unknown` (ajout, schéma 1.2) |
| `academic_task`, `discipline` | tels que dits, sinon `null` |
| `context` | cadre de la situation tel que raconté |
| `ai_tool` | outils nommés |
| `student_action_before`, `ai_action`, `student_action_after` | déroulé |
| `stated_reason` | raisons données par l'enquêté·e |
| `explicit_constraints` | délais, interdictions, surveillance, consignes… |
| `verification_or_control` | contrôles déclarés sur le résultat |
| `stated_frequency` | fréquence uniquement (« parfois », « souvent », « rarement », « une fois », « jamais », « toujours »…), sinon `null` (ajout au schéma initial) |
| `scope_qualifier` | qualificatif de portée tel que dit (« surtout », « principalement »…), sinon `null` (ajout, schéma 1.1) |
| `assessment_context` | `graded`, `ungraded`, `exam`, `class`, `personal` (hors du cadre des études), `unknown` |
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
schéma 1.1), `metadiscursive_self_evaluation` (ajout, schéma 1.2, voir
ci-dessous), `other` (décrit librement, pour ne pas forcer un passage
dans une catégorie).

**`metadiscursive_self_evaluation`** : l'enquêté·e évalue explicitement sa
propre formulation, son propre récit ou la manière dont il ou elle présente sa
conduite (« c'est assez ridicule ce que je dis », « là j'abuse », « c'est un
peu facile de dire ça », « je sais que dit comme ça c'est bizarre »). Ces
formulations étaient auparavant rangées dans `other`. Ne relèvent pas de ce
type : « ChatGPT est ridicule » (l'outil est évalué, pas la parole),
« ça m'énerve » / « je suis triste » (`explicit_emotion`), « je préfère le
faire moi-même » (`preference_statement`). Le signal reste descriptif : ni
justification, ni fonction, ni sentiment non formulé ne lui sont prêtés. Un
signal de ce type dont les citations ne contiennent aucune référence de
l'enquêté·e à sa propre parole (première personne, « dire ça », « dit comme
ça ») est signalé `METADISCURSIVE_NO_SELF_REFERENCE`.

**Pertinence (version 1.3, étape 3.7)** : un élément de langage ne devient un
signal que s'il intervient dans la manière dont l'enquêté·e rend compte de sa
conduite (la décrit, la limite, la reformule, l'évalue, en distingue des usages,
exprime un trouble, une réserve, une préférence…). Ni quota ni plafond, mais
pas d'inventaire linguistique : « euh », « ben », « voilà », « juste », « un
peu », « on va dire »… ne sont jamais relevés seuls ; on préfère un signal
substantiel à plusieurs micro-signaux redondants. Le premier vrai entretien
long avait donné 498 signaux avec les consignes 1.2. Détails, exemples et
couche déterministe (`set_aside_signals`) : [`stage3_7.md`](stage3_7.md).

**Il peut** : noter « scrupules » comme `explicit_affect` quand l'enquêté·e
dit « j'ai un peu des scrupules » ; relever « Je l'utilise jamais pour écrire…
enfin, sauf une fois » comme autocorrection et exception ; relever « juste
reformuler, jamais écrire » comme restriction ; mettre en regard deux tours qui
se contredisent, de façon neutre.

**Il ne peut pas** : attribuer une émotion non formulée (honte, culpabilité,
peur…), prêter une fonction à une formulation (se défendre, se justifier,
stratégie), qualifier un rire (« gêné »), un silence ou une contradiction
(mensonge, hypocrisie, dissimulation, réparation), ni convertir une
préférence énoncée (`preference_statement`) en trait de la personne
(autonomie, résistance, identité, position morale). Rire et silence ne sont
relevés que s'ils sont **transcrits**.

### Entretiens longs (étapes 3.6 et 3.7)

Au-delà d'environ 7 000 tokens estimés, l'Interaction Signal Reader lit
l'entretien en blocs de tours qui se chevauchent (même agent, mêmes consignes,
même schéma), puis une lecture légère rapproche les passages éloignés ; les
signaux sont fusionnés de façon déterministe en un seul `interaction_signals.json`.
Détails : [`interaction_chunking.md`](interaction_chunking.md). Depuis l'étape 3.7,
le Practice Extractor lit lui aussi un entretien long par blocs (au-delà d'environ
5 600 tokens estimés) ; ses pratiques sont fusionnées sans jamais réunir deux
conduites différentes ([`stage3_7.md`](stage3_7.md)).

### Schéma d'un signal (schema_version 1.2)

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
(`enqueteur` / `enquete` / `unknown`), `text`, `page` pour un PDF et, pour les
seuls tours dont l'attribution du locuteur est douteuse selon l'audit,
`speaker_warning` (`suggested_speaker`, `confidence`). Le `speaker` officiel
n'est jamais remplacé ; les deux agents reçoivent le même message.
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
| `NON_USE_REASON_MISMATCH` | warning | `non_use_reason` renseigné pour un usage, ou absent pour `non_use` / `refusal` |
| `METADISCURSIVE_NO_SELF_REFERENCE` | warning | signal métadiscursif sans référence de l'enquêté·e à sa propre parole |
| `AGENT_FLAGGED_REVIEW` | info | l'agent demande une vérification |
| `EXPLICITNESS_UNCLEAR` | info | l'agent qualifie sa lecture d'incertaine |
| `EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN` | info | citation d'un tour dont l'attribution du locuteur est douteuse (audit) |

Les codes propres à l'auditeur des locuteurs sont décrits dans
[`speaker_attribution_audit.md`](speaker_attribution_audit.md#revalidation-déterministe-rien-nest-corrigé).

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
signale, il ne supprime rien. Le champ `reason` de l'auditeur des locuteurs
est contrôlé de la même façon (concepts et jugements).

**« Réparation » en contexte (garde-fou 1.2).** Dans le premier entretien
réel, l'étudiant parlait littéralement de réparer un lave-vaisselle : ces
occurrences étaient signalées à tort. Le terme n'est pas retiré de la liste ;
chaque occurrence (réparation, réparer, réparateur…) est classée d'après les
mots qui l'entourent dans la même proposition, sans IA :

1. marqueur interprétatif fort (identité, soi, conduite, image, discursive,
   symbolique, interactionnelle, accountability, situation…) → **signalé** ;
2. objet ou cadre matériel (lave-vaisselle, appareil, téléphone, ordinateur,
   voiture, domestique, coût de, tutoriel de…) → non signalé ;
3. nom d'analyse devant (travail de, opération de, stratégie de, forme de…) → **signalé** ;
4. objet matériel ailleurs dans la même pratique / le même signal → non signalé ;
5. sinon le nom seul (« une réparation ») reste signalé, comme avant ; le verbe non.

Exemples : « réparation du lave-vaisselle », « réparation d'un téléphone »,
« coût de réparation », « réparer un ordinateur » → aucun avertissement ;
« travail de réparation de sa conduite », « réparation discursive »,
« opération de réparation », « réparation de son identité étudiante » →
`INTERPRETIVE_VOCABULARY`. L'exemption par les citations est elle aussi
contextuelle : un enquêté qui parle de réparer son lave-vaisselle n'autorise
pas l'agent à écrire « réparation de son identité ».

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
- La validation des preuves et le garde-fou sont recalculés à chaque
  réutilisation : changer leur version ne repaie aucun appel.
- Invalidation ciblée : modifier les consignes, le schéma ou la version d'UN
  agent ne relance que cet agent ; modifier l'auditeur des locuteurs ne relance
  que l'audit — les agents ne sont relancés que si les `speaker_warning`
  qu'ils reçoivent changent (leur entrée a alors réellement changé). Le gabarit
  du message utilisateur est commun aux deux agents : le modifier relance les deux.
- Étape 3.5 : les versions 1.2 des deux agents (consignes et schémas modifiés)
  ne réutilisent aucune entrée 1.1 ; l'audit a sa propre entrée
  (`data/cache/analysis/speaker_attribution_auditor/`).
- Workflow Claude Code : ce cache n'est pas utilisé (`NoCache`) ; `response.json` fait foi et la garde
  contre les réexécutions (étape complète et à jour non rejouée) joue son rôle ; `--force` (sur demande
  explicite) rejoue une étape.

Chaque exécution écrit un manifest par agent (`practice_manifest.json`,
`interaction_manifest.json`) :

```json
{
  "agent": "practice_extractor", "agent_version": "1.2", "schema_version": "1.2",
  "prompt_sha256": "…", "schema_sha256": "…", "source_sha256": "…", "transcript_sha256": "…",
  "model": "workflow_claude_code", "request_params": {"effort": null, "temperature": null},
  "cache_key": "…", "cache_hit": false, "status": "SUCCESS",
  "created_at": "…", "run_at": "…", "api_calls": 0, "billed_this_run": false,
  "usage": {"input_tokens": null, "output_tokens": null,
            "cache_creation_input_tokens": null, "cache_read_input_tokens": null},
  "duration_seconds": 0.0, "response_model": "claude_code", "request_id": "<identifiant de la tâche>",
  "stop_reason": null, "speaker_warning_count": 0,
  "item_count": 6, "invalid_evidence_count": 0, "needs_review_count": 1, "error": null
}
```

`api_calls`, `billed_this_run`, `usage`, `request_params`, `stop_reason` sont des champs **legacy** (anciennes
exécutions par API) : conservés pour ne pas changer le format des sorties ni la relecture d'anciens fichiers,
ils valent toujours 0 / `false` / `null` ; `request_id` porte l'identifiant de la tâche du workflow.

## 9. Coût

Aucun : 0 appel API, 0 token facturé. L'interface affiche « 0 appel(s) API — 0 tokens entrée — 0 tokens
sortie » pour la dernière exécution. Le nombre de tâches d'agent se lit dans l'état du workflow (« n/m
tâche(s) avec une réponse conforme ») et dans `workflow/stage3/status.json`.

## 10. Statuts et erreurs

`PENDING` → `RUNNING` → `SUCCESS` | `SUCCESS_WITH_WARNINGS` (au moins une
anomalie `error`/`warning` de validation) | `FAILED` | `CACHED` | `PARTIAL`
(entretien long : au moins un bloc en échec ; les blocs réussis sont conservés,
`analysis_complete: false`, voir [`interaction_chunking.md`](interaction_chunking.md)).

Erreurs d'agent (`core/llm_client.py`) : `AWAITING_AGENT` (réponse pas encore écrite : la tâche reste en
attente), `EMPTY_RESPONSE`, `INVALID_JSON`, `SCHEMA_VALIDATION` (emplacements et types d'erreur seulement,
jamais de contenu d'entretien). Une réponse invalide n'est jamais corrigée en silence : la tâche est listée
« à corriger » et l'agent la reprend (`python scripts/trace_workflow.py check`, au plus 3 corrections). Aucun
repli vers une API. Les journaux ne contiennent ni transcript ni citation : identifiant d'entretien, agent,
statut.

## 11. Configuration et première exécution

1. Aucune clé ni variable obligatoire. Facultatif (`.env` ou environnement) :
   `TRACE_INTERACTION_CHUNK_TOKENS` (taille d'un bloc de l'Interaction Reader, défaut 5 000) et
   `TRACE_PRACTICE_CHUNK_TOKENS` (Practice Extractor, défaut 4 000).
2. Dans Claude Code, ouvert dans le dépôt : « Exécute TRACE sur `entretien.docx` jusqu'à l'étape 3 »
   (procédure : [`TRACE_WORKFLOW.md`](../TRACE_WORKFLOW.md)).
3. Ou dans l'interface : importer un entretien, lancer l'ingestion, choisir « Test — un entretien » et
   cliquer sur « Préparer / reprendre l'étape 3 (workflow Claude Code, 0 appel API) », puis demander à Claude
   Code d'exécuter les tâches en attente et cliquer de nouveau.

La reproductibilité repose sur le versionnage des prompts et des schémas, l'empreinte de chaque paquet de
tâche et la conservation de chaque réponse (`response.json`).

## 12. Tests

Aucun test automatique n'appelle une API ni le réseau : `tests/conftest.py` retire les variables `TRACE_*`,
n'ouvre pas `.env`, fait échouer toute connexion réseau TCP et redirige le cache vers un dossier temporaire.
Les agents sont simulés par `tests/fake_llm.FakeAgents` (même interface et même validation que le client du
workflow) ou par des réponses écrites dans les paquets de tâche ; l'entretien fictif et les réponses simulées
sont dans `tests/synthetic_interviews.py`.

- `tests/test_llm_client.py` — réponses invalides (JSON, schéma, vide), messages sans contenu d'entretien, paramètres, schémas stricts identiques à la référence, aucun SDK
- `tests/test_practice_extractor.py`, `tests/test_interaction_signal_reader.py` — schémas, prompts, preuves
- `tests/test_evidence_validator.py` — citations exactes / inventées / autre entretien / intervalle inversé
- `tests/test_analysis_cache.py` — hit / prompt, version, modèle ou transcript changés
- `tests/test_analysis_pipeline.py` — indépendance, parallélisme, isolement des échecs, multi-entretiens
- `tests/test_stage3_synthetic.py` — scénario qualitatif synthétique
- `tests/test_speaker_attribution_auditor.py` — présélection, tâche unique ou nulle, cache, transcript inchangé, avertissements
- `tests/test_interpretation_guard.py` — « réparation » ordinaire / interprétative
- `tests/test_stage3_5_synthetic.py` — scénario global de l'étape 3.5
- `tests/test_interaction_chunking.py` — étape 3.6 : entretien long (349 tours synthétiques), blocs, chevauchement, fusion, longue distance, bloc en échec, cache par bloc, avertissements
- `tests/test_claude_code_workflow.py` — étape 3 en workflow : paquets, réponses, validation, garde, restauration
- `tests/test_app_stage3.py` — interface (Streamlit AppTest)
- `tests/e2e/browser_check.py` — navigateur réel (Playwright), lancé à la main

## 13. Limites connues

- Les tests valident la chaîne et les garde-fous, **pas** la qualité d'un vrai
  modèle : la lecture humaine des sorties reste indispensable.
- Le garde-fou lexical est simple (listes de radicaux) : faux positifs possibles
  (« dépendant de la matière »), et une interprétation formulée sans ces mots
  n'est pas détectée. Pour « réparation », la détection en contexte repose sur
  de courtes listes de mots : un objet matériel absent de la liste (« réparation
  du store ») reste signalé si rien d'autre n'indique le sens matériel.
- Les tests des non-usages et du type `metadiscursive_self_evaluation` utilisent
  des réponses **simulées** : ils vérifient le contrat (schéma, consignes,
  validation, non-fusion), pas la capacité d'un vrai modèle à appliquer les
  consignes : seule la lecture des réponses produites par Claude Code sur des entretiens réels le permet.
- Une citation exacte mais trop courte (« oui ») est valide techniquement
  sans être forcément probante.
- Un entretien long est lu par l'Interaction Signal Reader en blocs qui se
  chevauchent (étape 3.6, [`interaction_chunking.md`](interaction_chunking.md)) ;
  le Practice Extractor aussi depuis l'étape 3.7 ([`stage3_7.md`](stage3_7.md)).
- Les tours de locuteur `unknown` sont transmis tels quels : l'agent ne sait
  pas toujours qui parle.
- Aucune parallélisation n'est imposée par TRACE : l'orchestrateur Claude Code peut lancer les tâches d'un
  même passage en parallèle (sous-agents), elles sont indépendantes.
