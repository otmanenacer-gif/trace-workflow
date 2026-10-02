# Étape 6 — Comparaison inter-entretiens (Cross-Interview Comparator)

> **Exécution** : les agents sont exécutés localement par un modèle servi par Ollama
> ([`local_runtime.md`](local_runtime.md)), sans aucune API ni clé. Dans ce document, un « appel » désigne un
> appel au modèle local ; les mentions de tokens facturés, de cache de l'API ou de `TRACE_LLM_MAX_TOKENS`
> décrivent la conception d'origine (exécution par API, retirée) : `api_calls` et `billed_this_run` valent
> toujours 0 / `false`.

## Pourquoi cette étape

L'étape 5 produit, pour chaque entretien SÉPARÉMENT, une configuration intra-entretien validée. L'étape 6
compare **plusieurs entretiens** :

> Quelles régularités, différences, oppositions et cas négatifs apparaissent entre les entretiens dans les
> manières de rendre les usages et non-usages des IAG intelligibles et acceptables ?

Le niveau d'analyse est celui des **frontières, critères du métier d'étudiant, manières de rendre compte
(accounting moves), zones ordinaires, exceptions, tensions, variations contextuelles et changements temporels
validés** — jamais celui de la personnalité des individus. L'étape 6 ne rédige pas l'analyse finale, ne répond pas
définitivement à la problématique, n'ajoute aucune référence théorique, ne classe personne et ne construit aucun
« type d'étudiant » (étape 7).

## Unité de comparaison

TRACE écrit « cette frontière apparaît dans les entretiens A, B et C », « deux configurations différentes sont
observées concernant la délégation », « l'entretien D constitue un cas négatif pour cette régularité » — jamais
« les étudiants sont de type X ». Chaque affirmation garde ses `interview_ids`.

## Architecture

```
student_trajectory.json · student_trajectory_validation.json · student_trajectory_manifest.json  (× N entretiens)
        │   importés depuis des fichiers, ou repris du run courant (lecture seule)
        ▼
core/cross_interview_corpus.py     constitution du corpus : reconnaissance, regroupement, contrôles (aucun LLM)
        ▼
core/cross_interview_material.py   préparation DÉTERMINISTE : matériau, index, représentation normalisée (aucun LLM)
        ▼
agents/cross_interview_comparator.py   Cross-Interview Comparator (LLM, prompts/cross_interview_comparator.md v1.0)
        │                                1 appel pour tout le corpus
        ▼
core/cross_interview_validator.py  validation DÉTERMINISTE (appuis, comptes, positions, revues, cas négatifs, texte)
        ▼
data/outputs/cross_interview/<corpus_id>/analysis/
    cross_interview_comparison.json · cross_interview_validation.json · cross_interview_manifest.json
```

Orchestration : `core/cross_interview.py`. L'étape 6 ne reçoit **ni** les transcriptions, **ni** les sorties des
étapes 3 et 4 ; elle n'appelle et ne modifie jamais les étapes 3, 4 et 5 et n'écrit rien dans les dossiers des
entretiens. Les seules citations disponibles sont celles que l'étape 5 a conservées (expressions entre « » de ses
descriptions, ancrages temporels validés) ; les tours sont repris par identifiant.

## Constituer le corpus (`core/cross_interview_corpus.py`)

Dans l'interface, section « Étape 6 — Comparaison inter-entretiens » › « Constituer le corpus Stage 6 » (disponible
même sans run) : importer les 3 JSON de l'étape 5 de chaque entretien, et/ou cocher « Inclure les sorties de
l'étape 5 du run en cours ». Les fichiers sont reconnus par leur **contenu** (préfixés ou non) et regroupés par
`interview_id`. Contrôles par entretien — un manquement **exclut cet entretien avec sa raison**, sans bloquer les
autres :

| Contrôle | Effet |
| --- | --- |
| les 3 fichiers présents, même `interview_id`, identifiants des objets préfixés par cet entretien | sinon exclu |
| même fichier importé deux fois à l'identique | dédoublonné (note) |
| deux versions différentes d'un même fichier pour un entretien | **doublon ambigu** : exclu |
| schéma et validateur de l'étape 5 = ceux de TRACE | sinon exclu ; autre prompt / version d'agent : avertissement |
| statut SUCCESS, SUCCESS_WITH_WARNINGS ou CACHED, `analysis_complete: true` (document, manifest, validation) | sinon exclu |
| `validation_error_count = 0` (document, manifest, validation) | sinon exclu |
| `source_hashes` identiques document / manifest ; manifest décrivant ce document (statut, configuration, fichier, comptes, matériau) | sinon exclu |
| **aucune altération** : comptes et listes par type recalculés depuis les affirmations ; statut, `needs_review` et `review_reasons` de chaque objet recalculés depuis les anomalies du fichier de validation | sinon exclu |

Tableau affiché : `interview_id | statut | claims | needs_review | configuration | importé`, avec **N importés**
et **N exploitables**. Fichiers illisibles ou non reconnus : listés, jamais comptés comme entretiens.

**Seuil** : moins de 2 entretiens exploitables → étape 6 **bloquée** (aucun appel) ; 2 → comparaison
**exploratoire** ; 3 et plus → comparaison normale.

Les triplets exploitables sont recopiés octet pour octet dans `<corpus_id>/stage5/<interview_id>/` et le bilan de
l'import dans `<corpus_id>/cross_interview_corpus.json`. `corpus_id` = empreinte des fichiers importés.

## Préparation déterministe (`core/cross_interview_material.py`)

Pour chaque entretien : `configuration_type`, affirmations **utilisables** (`usable_for_next_stages`) avec type,
description, contextes, confiance, `needs_review`, `review_reasons`, opérations, ancrages validés, tours cités,
requalification éventuelle (`requalified_from`) ; critères du métier d'étudiant utilisables avec les mêmes champs.
Les objets rejetés par l'étape 5 sont comptés, jamais transmis.

Index (identifiants groupés, aucune conclusion) : A. affirmations par famille, avec pour chaque famille les
entretiens où elle est **présente** et ceux où elle n'est **pas observée** ; B. critères par entretien ;
C. opérations par entretien et entretiens par opération ; D. contextes / tâches ; E–H. frontières, exceptions,
tensions, zones ordinaires (et variations, changements temporels, opérations récurrentes) par entretien.

Regroupements lexicaux (contextes, libellés de critères) : **à formulation identique seulement** (après
repliement de la casse, des accents et de la ponctuation). « dissertation » et « dissertations » ne sont pas
rapprochés ; « résultat rendu » n'est jamais rapproché d'« effort ». Un rapprochement est proposé par le
Comparator et contrôlé sur les identifiants qu'il cite.

**Représentation envoyée** : normalisée — entretiens abrégés `I01…` (table de correspondance incluse),
identifiants locaux de l'étape 5 (`TC003`, `RC001`, `T0040`), un entretien par ligne, valeurs par défaut omises,
JSON compact. `expand_ids` rétablit les identifiants entiers ; une valeur inconnue est laissée telle quelle pour
que le validateur la signale.

## Rôle du LLM (Cross-Interview Comparator 1.0)

Mission : A. régularités transversales ; B. variantes ; C. contrastes ; D. cas négatifs ; E. configurations
minoritaires ; F. comparaison des critères mobilisés pour rester reconnaissable comme étudiant·e.

Types d'affirmations (aucun n'est obligatoire) : `recurring_boundary`, `divergent_boundary`,
`recurring_accounting_move`, `divergent_accounting_move`, `recurring_student_role_criterion`,
`divergent_student_role_criterion`, `ordinary_zone_pattern`, `exception_pattern`, `contextual_association`,
`temporal_pattern`, `unresolved_cross_case_contrast`, `minority_configuration`, `negative_case`.

Schéma d'une affirmation inter-entretiens (après validation) :

```json
{
  "cross_claim_id": "CC001",
  "claim_type": "recurring_boundary",
  "description": "…", "description_display": "… (alias I01 remplacés par les interview_id)",
  "criterion_label": null,
  "interview_ids": ["ENT_A", "ENT_B", "ENT_C"],
  "support": [{"interview_id": "ENT_A", "claim_ids": ["ENT_A_TC001"], "criterion_ids": [],
               "evidence_turn_ids": ["ENT_A_T0002", "ENT_A_T0004"], "needs_review": false,
               "stage5_review_reasons": []}],
  "counterexamples": [{"interview_id": "ENT_D", "relation": "contrary_case", "description": "…",
                       "claim_ids": ["ENT_D_TC001"], "criterion_ids": [], "evidence_turn_ids": ["…"],
                       "validation_status": "supported", "needs_review": false}],
  "contexts": ["rédaction"],
  "corpus_n_total": 4, "corpus_n_usable": 4,
  "n_supporting_interviews": 3, "model_n_supporting_interviews": 3,
  "n_review_only_interviews": 0, "review_only_interview_ids": [],
  "n_counterexample_interviews": 1, "n_not_observed_interviews": 0,
  "interview_positions": {"ENT_A": "explicit_presence", "ENT_D": "contrary_case", "…": "not_observed"},
  "not_observed_interview_ids": [],
  "related_claim_numbers": [],
  "stage5_review_reasons": [],
  "confidence": "high", "model_needs_review": false, "needs_review": false, "review_reasons": [],
  "validation_status": "valid", "usable_for_next_stages": true
}
```

## Règles méthodologiques et leur contrôle (`core/cross_interview_validator.py`)

**Comptes en entretiens.** `n_supporting_interviews` est RECALCULÉ sur les `interview_id` distincts : plusieurs
affirmations, critères ou entrées d'un même entretien ne comptent qu'une fois (`DUPLICATE_SUPPORT_INTERVIEW`,
`COUNT_MISMATCH` corrigé). Chaque affirmation porte `corpus_n_total`, `corpus_n_usable`, `n_supporting_interviews`
et `interview_ids`. Un « 24 fois » est signalé (`OCCURRENCE_COUNT`) ; « 5 des 8 entretiens » est vérifié contre les
comptes recalculés (`COUNT_IN_TEXT_MISMATCH`).

**Petit N.** Une régularité (`recurring_*`, `*_pattern`, `contextual_association`) sur un seul entretien est
**requalifiée** en `minority_configuration`. Corpus de 2 : `exploratory: true`, confiance limitée à « medium ». Un
seul entretien d'appui : confiance « low ». « En général », « la plupart des étudiants », « les étudiants font… »
sont signalés (`GENERALIZATION_BEYOND_N`).

**Présence / refus explicite / non-observation.** Pour chaque affirmation, chaque entretien exploitable a une
position : `explicit_presence` (appui documenté), `explicit_refusal` (« l'effort n'est pas important pour moi »),
`contrary_case` (« je l'utilise même quand j'ai le temps »), `divergent_variant`, ou `not_observed`. Un
contre-exemple sans élément de l'étape 5 ne fait **pas** un refus : l'entretien reste `not_observed`
(`COUNTEREXAMPLE_WITHOUT_SUPPORT`, conservé à revoir). « ne possède pas ce critère », « n'accorde aucune
importance »… appliqués à une non-mention sont signalés (`NOT_OBSERVED_AS_ABSENCE`).

**Éléments à revoir (secondaires).** Chaque entrée d'appui porte `needs_review` (tous ses éléments sont à revoir)
et `stage5_review_reasons` ; l'affirmation porte l'union de ces raisons. Appui **uniquement** sur des éléments à
revoir → `needs_review` forcé et confiance « low » (`SUPPORTED_ONLY_BY_REVIEW_ITEMS`, `STRONG_PATTERN_ON_REVIEW_ITEMS`
si le modèle disait « high ») ; régularité dont moins de deux entretiens ont un appui propre →
`RECURRENCE_RESTS_ON_REVIEW_ITEMS`, confiance « low ». Un `CRITERION_NOT_GROUNDED`, un
`TEMPORAL_CHANGE_REQUALIFIED` ou un `RECURRENCE_NOT_DOCUMENTED` n'est jamais transformé silencieusement en résultat
propre. Un `temporal_pattern` exige des `explicit_temporal_change` **validés** par l'étape 5, sinon il est requalifié
en `contextual_association` (`TEMPORAL_PATTERN_REQUALIFIED`).

**Cas négatifs.** Le prompt demande de les chercher activement. Le document les rassemble dans `negative_cases`
(dédoublonnés par entretien et régularité compliquée), à partir des contre-exemples documentés ET des affirmations
`negative_case` — **même quand l'affirmation qui les porte est rejetée** pour une autre raison. Seuls les
identifiants inventés n'y entrent pas.

**Aucun appui inventé** (erreurs → affirmation rejetée, conservée dans le fichier, exclue des listes) : entretien
inconnu ou exclu, `claim_id` / `criterion_id` inexistant, d'un autre entretien ou rejeté par l'étape 5, entrée
d'appui sans élément, tour inexistant, contre-exemple inventé, citation « » absente du matériau de l'étape 5 des
entretiens cités.

**Pas de causalité, pas de typologie.** Erreurs : typologie de personnes (« type d'étudiant », « profil »,
« étudiants consciencieux »…) et explication par la personnalité, la morale, l'intelligence, la paresse, le milieu
social, le genre ou l'origine (absents des données). Avertissements : vocabulaire psychologisant (liste de l'étape 5),
formulation causale (« s'explique par », « est dû à », « la discipline conduit à »…), entretien nommé sans appui.
Une variation par discipline ou tâche est une `contextual_association` **observée**, avec ses entretiens et ses cas
divergents.

## Sorties

`cross_interview_comparison.json` : identité de l'agent, versions, `corpus_id`, `corpus_n_total`,
`corpus_n_usable`, `mode`, `exploratory`, `interview_ids`, `excluded_interviews` (avec raisons), `input_hashes`,
comptes, `configuration_distribution` (calculée par TRACE, comptée en entretiens), `cross_case_claims` (objets
annotés), listes **d'identifiants** (`recurring_boundaries`, `divergent_boundaries`, `recurring_accounting_moves`,
`divergent_accounting_moves`, `recurring_student_role_criteria`, `divergent_student_role_criteria`,
`ordinary_zone_patterns`, `exception_patterns`, `contextual_associations`, `temporal_patterns`,
`unresolved_cross_case_contrasts`, `minority_configurations`, `negative_case_claims`),
`student_role_criterion_patterns` (critère, `interview_ids`, `criterion_ids` et preuves par entretien, positions,
contrastes), `negative_cases`, `cross_case_summary`, `confidence`, `needs_review`, `material` (comptes et index).

`cross_interview_validation.json` : rapport du validateur. `cross_interview_manifest.json` : coût, cache, tokens
estimés, taille de la représentation et des JSON bruts, seuil, `map_reduce_used`, `upstream_stage_calls` (toujours 0).

## Coût, payload et MAP → REDUCE

1 appel pour tout le corpus (0 s'il est bloqué ou en cache). Mesures (`python scripts/stage6_payload_report.py`,
estimation 3,5 caractères par token) :

| Corpus | Affirmations / critères | Représentation | Message (tokens estimés) | Part des JSON bruts | Appels |
| --- | --- | --- | --- | --- | --- |
| 2 entretiens | 12 / 7 | 5 316 car. | ≈ 1 960 | 17,5 % | 1 |
| 8 entretiens | 41 / 23 | 17 516 car. | ≈ 5 440 | 15,9 % | 1 |
| 17 entretiens | 75 / 40 | 31 258 car. | ≈ 9 370 | 14,4 % | 1 |
| 17 × « OTMANE-like » (projection) | ≈ 170 / 85 | ≈ 76 400 car. | ≈ 21 800 (représentation) | — | 1 |

Consignes et schéma fixes : ≈ 2 800 + 1 400 tokens (réutilisables par le cache de prompt de l'API).

**MAP → REDUCE : non implémenté**, parce que la mesure ne le justifie pas : 17 entretiens synthétiques ≈ 9 400
tokens, et même 17 entretiens de la taille du plus long entretien synthétique restent sous le seuil d'un appel
unique de TRACE (24 000 tokens, le même qu'à l'étape 5). Un appel unique est en outre préférable pour la
comparaison elle-même : le Comparator voit simultanément tous les entretiens, donc toutes les régularités et tous
les cas négatifs. Au-delà du seuil, l'appel unique a lieu quand même et le corpus est signalé
(`PAYLOAD_OVER_THRESHOLD`, `needs_review`) ; `max_tokens` n'est jamais augmenté. Si un vrai corpus dépassait
nettement le seuil, la voie prévue est un découpage par blocs d'entretiens avec cache par bloc, réduction finale
sur toutes les affirmations et tous les cas négatifs, et statut PARTIAL si un bloc manque.

## Cache

Clé : empreintes des sources de l'étape 5 (`source_hashes`), SHA-256 du message exact envoyé, agent, version,
SHA-256 du prompt et du gabarit, schéma, modèle, paramètres (`data/cache/analysis/cross_interview_comparator/`).
Même corpus + même version → 0 appel (statut `CACHED`), quel que soit l'ordre d'import. Ajouter, retirer ou modifier
un entretien ne change que la requête de l'étape 6 : les étapes 3 à 5 restent intactes. Un échec n'est jamais mis
en cache, et une comparaison précédente est retirée (jamais de synthèse COMPLETE si l'appel manque).

## Tests (aucun appel réel)

- `tests/synthetic_stage6.py` : 17 entretiens synthétiques (+ 1 invalide) composés de modules ; leurs triplets
  d'étape 5 sont produits par le **vrai** orchestrateur de l'étape 5 (Trajectory Mapper simulé) ; Comparator simulé
  qui résout un plan « par module » dans la représentation réellement envoyée ;
- `tests/test_cross_interview_corpus.py` : reconnaissance, doublons, triplets mélangés, versions, statuts,
  empreintes, altérations, seuils, triplet du pipeline complet (étapes 3 à 5) repris d'un run ;
- `tests/test_cross_interview_material.py` : matériau, index, présence / non-observation, regroupement à
  formulation identique, représentation, déterminisme, identifiants, tailles 2 / 8 / 17 ;
- `tests/test_cross_interview_validator.py` : appuis inventés, comptes, positions, revues, requalifications,
  cas négatifs, vocabulaire, synthèse ;
- `tests/test_stage6_sociological.py` : cas A à J de la consigne (et versions adverses) ;
- `tests/test_stage6_pipeline.py` : cache, invalidation ciblée, échecs, isolement des étapes 3 à 5, seuil,
  déterminisme, fixture de 17 entretiens ;
- `tests/test_app_stage6.py` : interface (AppTest) ;
- `tests/e2e/browser_check.py` : scénarios N à P (Chromium ; `--only-stage6`).

## Limites

- L'intégrité d'un triplet est vérifiée par cohérence interne (comptes, listes, statuts, revues, empreintes du
  manifest) : l'étape 5 n'enregistre pas d'empreinte de son propre document, donc une modification d'un **texte**
  (description) qui garde la structure cohérente n'est pas détectable à l'import.
- Une étape 5 avec au moins une erreur de validation (une affirmation rejetée suffit) est exclue, comme exigé.
- Les contrôles de vocabulaire (typologie, causalité, généralisation, attribution sociale) sont lexicaux : ils
  signalent des formulations, ils ne garantissent pas l'absence de toute inférence implicite.
- La vérification d'une citation « » se fait contre le texte de l'étape 5 (descriptions, critères, ancrages),
  pas contre la transcription, que l'étape 6 ne reçoit pas.
- Les regroupements déterministes ne rapprochent que des formulations identiques ; l'équivalence de formulations
  proches reste une proposition du Comparator, contrôlée seulement sur les identifiants.
