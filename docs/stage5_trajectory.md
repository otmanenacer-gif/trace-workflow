# Étape 5 — Configuration et trajectoire intra-entretien

## Pourquoi cette étape

L'étape 4 produit des épisodes **locaux** : épisodes d'accountability, pratiques ordinaires, cas incertains,
et pratiques racontées sans aucun marqueur. L'étape 5 travaille **à l'intérieur d'un seul entretien** et
reconstruit la **configuration** des manières dont l'étudiant·e rend ses usages et non-usages intelligibles,
et, seulement lorsque le matériau le permet réellement, leur **trajectoire** :

> Quelles frontières, règles, distinctions et manières de rendre compte de ses pratiques se répètent, restent
> stables, varient selon les tâches ou contextes, comportent des exceptions, entrent en tension, ou changent
> explicitement dans le temps ?

Elle ne compare jamais deux entretiens, ne construit aucune typologie, ne compte aucune fréquence de corpus et
ne répond pas à la problématique générale (étapes 6 et suivantes).

## Ordre de l'entretien ≠ ordre biographique

`T0030` avant `T0200` signifie seulement que le passage a été **raconté** avant. Une évolution temporelle ne
peut être affirmée que si le matériau contient des **ancrages explicites** prononcés par l'enquêté·e (« au
lycée », « en première », « l'année dernière », « avant », « à l'époque », « maintenant », « aujourd'hui »,
« depuis », « cette année », « je ne … plus », « j'ai arrêté »…). Sans ordre temporel explicite, TRACE décrit
une **variation contextuelle**, jamais une évolution.

## Architecture

```
accountability_episodes.json (+ validation)     ← étape 4 (calculée ou restaurée depuis fichiers)
practice_extractor.json, interaction_signals.json, speaker_attribution_audit.json  ← étape 3 (minimum)
        │
        ▼
core/trajectory_candidates.py   préparation DÉTERMINISTE (aucun LLM)
        │   items : épisodes utilisables + pratiques sans marqueur, ancrages temporels, régularités
        ▼
agents/trajectory_mapper.py     Trajectory Mapper (LLM, prompts/trajectory_mapper.md v1.0) — 1 appel au plus
        ▼
core/trajectory_validator.py    validation DÉTERMINISTE (requalifications, rejets, propagation)
        ▼
analysis/student_trajectory.json · student_trajectory_validation.json · student_trajectory_manifest.json
```

Orchestration : `core/trajectory.py` (état de l'étape 4, plan des appels, cache, sorties, metadata).
Restauration de l'étape 4 : `core/stage4_restore.py`.

## Entrées et état de l'étape 4

- seuls les épisodes `usable_for_next_stages: true` servent aux conclusions ; les épisodes rejetés sont listés
  (`material.excluded_episodes`) et jamais envoyés ;
- `needs_review`, `review_reasons` et les avertissements de locuteur sont propagés ;
- l'étape 4 absente, en échec, bloquée ou **périmée** (ses `source_hashes` ne sont plus ceux des sorties de
  l'étape 3 installées : étape 3 relancée ou restaurée depuis) → étape 5 `BLOCKED`, aucun appel ;
- étape 4 `PARTIAL` → étape 5 `PARTIAL`, `analysis_complete: false`.

L'entretien complet n'est **jamais** envoyé : seulement les citations validées, les résumés de l'étape 4, la
tâche / le domaine / le statut d'usage des pratiques et la phrase exacte qui porte chaque ancrage temporel.

## Préparation déterministe (`core/trajectory_candidates.py`)

Pour chaque épisode : identifiant, statut, pratiques, opérations (type + tours), frontières, référence au
rôle d'étudiant·e et à autrui, intervalle, résumé, question rendue accountable, confiance, `needs_review` /
`review_reasons`, citations valides, contradictions signalées par l'étape 3, avertissements de locuteur,
contextes (tâche, domaine, discipline, fréquence). Les pratiques sans marqueur entrent avec leurs citations
valides de l'enquêté·e.

**Ancrages temporels** (`TEMPORAL_PATTERNS`, texte replié) : recherchés dans le texte des tours de l'enquêté·e
cités par un élément, à au plus 200 caractères d'une de ses citations (jamais dans une question de
l'enquêteur). Types : `period` (période nommée avec un rang biographique : collège < lycée < L1/prépa < L2 <
L3 < master), `past`, `present`, `transition` (le changement est dit). Faux amis écartés : « avant de rendre »,
« avant les partiels », « pas plus que ça », « plus rapide », « une fois » (fréquence).

**Régularités** (identifiants groupés, sans conclusion) : opérations répétées, frontières répétées, mêmes
tâches, matériau « règle + cas » (dans un même épisode, ou refus d'une tâche + usage de la même tâche
ailleurs), contrastes usage / non-usage, contradictions déjà signalées, pratiques ordinaires.

**Représentation envoyée** (format 1) : normalisée comme à l'étape 4 — chaque citation, pratique et ancrage une
seule fois (`quotes_by_id`, `practices_by_id`, `anchors_by_id`), identifiants abrégés (`E003`, `P012`,
`T0040`), valeurs par défaut omises, JSON compact.

## Rôle du LLM (Trajectory Mapper 1.0)

Il reçoit uniquement la représentation compacte et produit :

```json
{
  "configuration_type": "temporal_trajectory | contextual_configuration | mixed | no_clear_pattern",
  "claims": [{"claim_type": "...", "description": "...", "episode_ids": [], "practice_ids": [], "contexts": [],
              "accounting_move_types": [], "temporal_anchors": [{"text": "...", "turn_id": "..."}],
              "evidence_turn_ids": [], "confidence": "high | medium | low", "needs_review": false}],
  "student_role_criteria": [{"criterion": "...", "description": "Dans cet entretien, l'étudiant·e associe…",
                             "episode_ids": [], "evidence_turn_ids": [], "confidence": "...", "needs_review": false}],
  "trajectory_summary": "...", "confidence": "...", "needs_review": false, "mapper_notes": null
}
```

Types d'affirmations : `stable_boundary`, `recurring_accounting_move`, `contextual_variation`,
`explicit_temporal_change`, `exception`, `unresolved_tension`, `ordinary_zone`. Le prompt interdit la
psychologie et le récit de conversion (maturation, prise de conscience, culpabilité cachée, identité menacée,
stratégie de légitimation, rationalisation, « devient plus responsable », « apprend à mieux utiliser l'IA »,
normalisation progressive, dépendance, émancipation) sauf mot prononcé par l'enquêté·e, et toute résolution
d'une tension à la place de l'enquêté·e.

## Validateur (`core/trajectory_validator.py`)

| Contrôle | Effet |
| --- | --- |
| episode_id inexistant ou rejeté par l'étape 4, practice_id inconnu, aucun appui | erreur → affirmation rejetée |
| tour cité inexistant / d'un autre entretien ; aucun tour de l'enquêté·e | erreur |
| ancrage temporel absent du texte du tour (citation inventée) | erreur |
| ancrage prononcé par l'enquêteur, non temporel, hors du matériau | avertissement, ancrage non compté |
| `explicit_temporal_change` sans deux états, sans différence, ou sans ancrages qui ORDONNENT | **requalifié** en `contextual_variation` |
| `exception` sans règle explicite ou sans cas (« une fois », « sauf », opération `exception`, conduite contraire) | **requalifiée** en `contextual_variation` |
| frontière stable / opération récurrente / zone ordinaire sur un seul élément ; variation sur un seul contexte ; tension sur une seule formulation | avertissement |
| zone ordinaire contenant un épisode d'accountability ou incertain | avertissement |
| affirmation appuyée **uniquement** sur des épisodes à revoir | `needs_review` forcé (et avertissement si confiance `high`) |
| vocabulaire psychologisant / de conversion, résolution d'une tension, vocabulaire du changement sans changement validé, ordre de l'entretien présenté comme un temps, comparaison avec d'autres entretiens | avertissement |
| terme proposé par l'enquêteur et non repris, attribué à l'enquêté·e | **erreur** |
| critère : affect non prononcé par l'enquêté·e | **erreur** ; généralisé / non situé / sans épisode qui le formule : avertissement |
| `temporal_trajectory` / `mixed` sans changement temporel validé | configuration **requalifiée** |

Rien n'est réécrit : TRACE ajoute `validation_status`, `needs_review`, `review_reasons`, `support_ids`,
`review_episode_ids`, `speaker_warnings`, `validated_temporal_anchors`, `temporal_ordering_basis` et, en cas de
requalification, `model_claim_type` / `model_configuration_type`.

## Sorties

`student_trajectory.json` : identité de l'agent, statut, étape 4 (statut, restaurée ou non), empreintes
(`source_hashes`, dont celles des fichiers de l'étape 4), `configuration_type`, `trajectory_claims` (objets
annotés), listes **d'identifiants** `stable_boundaries`, `contextual_variations`, `explicit_temporal_changes`,
`exceptions`, `unresolved_tensions`, `ordinary_zones`, `recurring_accounting_moves` (aucun doublon d'objet),
`student_role_criteria`, `trajectory_summary`, `confidence`, `needs_review`, comptes, et `material` (bilan de
la préparation : ancrages, régularités, épisodes exclus).

`student_trajectory_validation.json` : rapport du validateur (issues, comptes, affirmations rejetées et
requalifiées). `student_trajectory_manifest.json` : coût, cache, tokens estimés, taille du payload, seuil.

## Restauration de l'étape 4

Le stockage Streamlit est éphémère. Dans la section « Étape 5 », « Restaurer des résultats Stage 4 existants »
accepte `accountability_episodes.json` et `accountability_episode_validation.json` (reconnus par leur
contenu, préfixés ou non). Contrôles bloquants : même `interview_id` (entretien choisi), étape 3 installée et
complète, versions compatibles (schéma, constructeur de candidats, format de requête, validateur ; une autre
version du prompt est seulement signalée), statut exploitable et `analysis_complete: true`, étape 3 COMPLETE,
`validation_error_count = 0`, `source_hashes` égaux aux empreintes de l'étape 3 installée, et **fichiers non
altérés** : TRACE recalcule les candidats depuis l'étape 3, re-valide les épisodes (citations revérifiées sur la
transcription actuelle) et exige l'égalité exacte des candidats, épisodes annotés, pratiques sans marqueur,
comptes et anomalies. Les fichiers sont recopiés octet pour octet, les sorties périmées de l'étape 5 retirées,
et l'interface affiche « Stage 4 restauré depuis fichiers — 0 appel API ». L'étape 4 n'est jamais relancée.

## Cache

Clé : SHA-256 du fichier source, SHA-256 du message exact envoyé, agent, version, SHA-256 du prompt et du
gabarit, schéma, modèle, paramètres. Mêmes entrées → 0 appel (statut `CACHED`). Modifier l'étape 5 n'invalide
que l'étape 5 (data/cache/analysis/trajectory_mapper/) ; une étape 4 relancée avec les mêmes épisodes, ou
restaurée depuis fichiers, donne la même requête et réutilise le cache.

## Coût

0 appel si moins de deux éléments utilisables (configuration `no_clear_pattern` déterministe), sinon **1 appel
par entretien**. Pas de découpage : sur l'entretien synthétique long « OTMANE-like » (190 tours, 50 pratiques,
20 épisodes, 23 pratiques sans marqueur), la représentation fait ≈ 16 700 caractères, ≈ 5 300 tokens estimés
(≈ 21 % des sorties brutes équivalentes des étapes 3–4), très loin du seuil de 24 000 tokens. Au-delà du seuil,
l'appel unique a lieu quand même et l'entretien est signalé (`PAYLOAD_OVER_THRESHOLD`) ; `max_tokens` n'est
jamais augmenté.

Estimation pour un vrai entretien de la taille du premier run réel (≈ 50 pratiques, ≈ 25 épisodes, citations
plus longues que dans les fixtures) : **1 appel**, ≈ 10 000 à 14 000 tokens d'entrée (dont ≈ 3 900 de consignes
et de schéma fixes, réutilisables par le cache de prompt de l'API) et ≈ 2 000 à 4 000 tokens de sortie ;
0 appel en relance (cache TRACE).

## Tests (aucun appel réel)

- `tests/test_trajectory_candidates.py` : ancrages temporels (formes, faux amis, rangs), base d'ordonnancement,
  éléments retenus, représentation normalisée, identifiants ;
- `tests/test_trajectory_validator.py` : requalifications, ancrages inventés / de l'enquêteur, appuis,
  tours, exception, régularités, vocabulaire, critères, propagation, configuration ;
- `tests/test_stage5_sociological.py` : cas A à H de la consigne (pipeline réel, lecteurs simulés, versions
  « adverses ») ;
- `tests/test_stage5_pipeline.py` : blocages, étape 4 périmée / partielle, cache, invalidation ciblée, coût,
  échecs, isolement des entretiens, seuil ;
- `tests/test_stage4_restore.py` : restauration de l'étape 4 puis étape 5 sans aucun appel d'étape 3 ou 4 ;
  refus (fichiers, versions, statuts, erreurs, empreintes, altérations) ;
- `tests/test_stage5_long.py` : entretien long « OTMANE-like » ;
- `tests/test_app_stage5.py` : interface (AppTest) ;
- `tests/e2e/browser_check.py` : scénarios J à M (Chromium).

## Limites

- Les ancrages temporels sont lexicaux : une temporalité exprimée seulement par le temps des verbes
  (imparfait / présent) n'est pas reconnue ; un ancrage posé par l'enquêteur (« et au lycée ? ») n'est jamais
  compté, même si l'enquêté·e répond à l'imparfait — l'affirmation est alors requalifiée et marquée à revoir.
- La « différence réellement documentée » est vérifiée sur la tâche et le statut d'usage des pratiques, pas sur
  le détail de ce que l'outil fait.
- La règle et le cas d'une exception sont reconnus par les opérations de l'étape 4, quelques marqueurs lexicaux
  et la polarité usage / non-usage d'une même tâche (tâche normalisée de l'étape 4).
- Un entretien au-delà du seuil d'un appel n'est pas découpé (non nécessaire sur les entretiens mesurés).
- La restauration de l'étape 4 exige les versions actuelles du constructeur de candidats et du validateur de
  l'étape 4 (le recalcul d'intégrité en dépend).
