# Étape 4 — Épisodes d'accountability

## Pourquoi cette étape

La problématique de TRACE porte sur la manière dont les étudiant·es rendent leurs usages et
non-usages des IAG **intelligibles et acceptables**, et continuent d'être reconnaissables comme
étudiant·es. L'étape 3 décrit séparément *ce que l'étudiant·e dit faire ou ne pas faire*
(Practice Extractor) et *les phénomènes interactionnels de son récit* (Interaction Signal Reader).

L'étape 4 relie ces deux descriptions pour reconstruire des **épisodes locaux d'accountability** :
des passages liés où une pratique (ou un non-usage) devient quelque chose que l'enquêté·e précise,
limite, distingue, reformule, défend explicitement, contextualise, évalue, oppose à autre chose,
présente comme une exception, ou relie à une norme, un effort, un apprentissage, un contrôle ou au
regard d'autrui.

« Accountability » est pris au **sens ethnométhodologique** : rendre une conduite descriptible,
intelligible et reconnaissable. Ce n'est pas une lecture des intentions : aucune justification n'est
transformée en stratégie, rationalisation, défense psychologique ou calcul ; aucune motivation cachée
n'est attribuée.

Hypothèse de travail essentielle : **toutes les pratiques ne produisent pas d'épisode**. Beaucoup sont
racontées comme allant de soi — c'est un résultat sociologique à part entière (`ordinary_practice`).

## Architecture

```
sorties de l'étape 3 (lues sur le disque, jamais modifiées)
    practice_extractor.json, interaction_signals.json, speaker_attribution_audit.json, manifests
        ↓
Candidate Episode Builder DÉTERMINISTE      core/accountability_candidates.py   (aucun LLM)
        ↓   candidats + citations + tours de contexte + avertissements de locuteur
Accountability Episode Builder (LLM)        agents/accountability_episode_builder.py
                                            prompts/accountability_episode_builder.md (v1.0)
        ↓   au plus 1 appel par entretien, 0 s'il n'y a pas de candidat
Episode Validator DÉTERMINISTE              core/accountability_episode_validator.py
        ↓
analysis/accountability_episodes.json
analysis/accountability_episode_manifest.json
analysis/accountability_episode_validation.json
```

Orchestration, cache, états de l'étape 3 : `core/accountability.py`. L'étape 4 ne relance jamais
l'étape 3 et n'en modifie aucun fichier ; elle est déclenchée par un bouton distinct dans l'interface.

## Entrées

- `structured_transcript.json` (texte exact des tours, locuteurs) ;
- `practice_extractor.json` : pratiques validées (`practice_id`, citations, statut d'usage, tâche, domaine…) ;
- `interaction_signals.json` : signaux **retenus** après la sélectivité de l'étape 3.7 (les signaux
  écartés, `set_aside_signals`, ne sont pas utilisés) ;
- `speaker_attribution_audit.json` : les tours au locuteur douteux deviennent des `speaker_warning` ;
- `practice_manifest.json`, `interaction_manifest.json` : statut de chaque agent.

## Candidate Episode Builder (déterministe)

**Appuis.** Seules les citations **valides** comptent (vérification exacte refaite avec
`core.evidence_validator`). Une pratique ou un signal dont toutes les citations valides viennent de
tours de l'enquêteur (sans `speaker_warning`) n'est jamais un appui : une question ne fait pas une
position de l'enquêté·e.

**Signaux.**

| Catégorie | Types | Rôle |
|---|---|---|
| déclencheurs | `explicit_emotion`, `reference_to_teacher_judgment`, `reference_to_peer_judgment`, `reference_to_rule`, `preference_statement`, `metadiscursive_self_evaluation`, `normative_formulation`, `self_correction`, `self_reformulation`, `restriction`, `exception`, `contrast`, `cross_turn_contradiction`, `distancing_from_own_practice` | peuvent créer un candidat |
| d'appui | `minimization`, `modalization`, `generalization`, `attribution_to_others`, `significant_repetition`, `vocabulary_shift`, `other` | rattachés à un candidat existant, jamais seuls |
| ignorés | `hesitation`, `intensification`, `transcribed_laughter`, `transcribed_silence`, `pronoun_shift`, et tout micro-marqueur nu (« euh », « ben », « juste »…) | jamais de candidat |

**Règles de création** (aucune autre) :

- **A. proximité** — un signal déclencheur est rattaché à la ou aux pratiques **les plus proches** si
  leurs tours se chevauchent, sont distants d'au plus `PROXIMITY_MAX_GAP = 2` positions (une question
  entre deux réponses) ou appartiennent au même échange question/réponse ;
- **B. contradiction entre tours** — `cross_turn_contradiction` est rattachée aux pratiques ancrées sur
  les tours **qu'elle cite**, même très éloignés ; jamais aux pratiques situées entre eux ;
- **C. polarité** — un non-usage / refus et l'usage le plus proche portant sur la **même tâche**
  (`academic_task` normalisée : sans accents, articles ni pluriel final) et le même domaine ;
- **D. même passage** — un usage et un non-usage / refus ancrés sur un même tour (« pour reformuler oui,
  mais pas écrire ») forment un groupe ;
- **E. frontière explicite** — une citation de la pratique formule une limite : « mais pas / mais
  jamais », « pas pour », « à ma place », « moi-même », « je (ne) veux pas que », « normalement »,
  « sauf », « seulement / uniquement / que pour / juste pour ».

Jamais de relation entre deux passages au seul motif qu'ils parlent du même outil.

**Fusion déterministe.** Deux candidats aux mêmes pratiques sont fusionnés ; un candidat dont les
pratiques sont incluses dans celles d'**un seul** autre candidat y est rattaché (ex. la frontière de
T0006 et l'exception de T0012 rejoignent la contradiction T0006/T0012). Rien d'autre.

**Pratiques sans marqueur.** Une pratique qui n'entre dans aucun candidat est listée dans
`unmarked_practices` avec `episode_status: "ordinary_practice"` et
`classification: "deterministic_no_marker"` : elle est racontée sans aucun des indices ci-dessus et
n'est **pas envoyée au modèle** (économie, et aucun épisode forcé).

**Représentation envoyée.** Candidats (identifiants, déclencheurs), pratiques et signaux concernés
(citations valides seulement), et les seuls tours cités plus la question qui précède chacun (au plus
8 par candidat), chacun une seule fois. Un tour de plus de 1 200 caractères est abrégé autour des
citations (« […] »). `</` est échappé : le texte d'un entretien ne peut pas fermer la balise
`<candidates>`. Jamais l'entretien entier.

## Rôle du LLM

L'Accountability Episode Builder (v1.0) décide, pour chaque candidat :

1. s'il forme réellement un épisode d'accountability ;
2. quels candidats forment un même épisode (`candidate_ids`) ;
3. quelle question pratique est rendue accountable (`accountability_problem`) ;
4. quelles opérations descriptibles le récit accomplit (`accounting_moves`).

Il ne voit que les candidats. Il ne corrige jamais un locuteur et ne transforme jamais une formulation
de l'enquêteur en position de l'enquêté·e.

### Les trois statuts

| Statut | Quand | Contraintes |
|---|---|---|
| `accountability_episode` | au moins une opération explicite sur la pratique (restriction, exception, distinction, refus, évaluation, reformulation, jugement d'autrui…) | ≥ 1 opération, appuyée sur un tour de l'enquêté·e ; matériau substantiel |
| `ordinary_practice` | pratique racontée sans réserve, justification, frontière, trouble, reformulation, évaluation, contradiction ni restriction | en général 0 opération ; pas de frontière ; `accountability_problem: null` |
| `uncertain` | matériau ambigu (citation peu claire, locuteur douteux, lien incertain) | `confidence: "low"`, `needs_review: true` |

### Opérations (`accounting_moves`)

Descriptions de ce que le texte **fait**, jamais des motivations : `restriction`, `exception`,
`general_rule` (ajouté : « normalement je fais mes plans moi-même »), `distinction`, `comparison`,
`refusal`, `preference`, `appeal_to_control`, `appeal_to_verification`, `appeal_to_effort`,
`appeal_to_learning`, `appeal_to_authorship`, `appeal_to_external_judgment`, `normalization`,
`self_evaluation`, `reformulation`, `other_explicit_move`. Chaque opération cite ses tours
(`evidence_turn_ids`).

### Frontières (`boundary_objects`)

Facultatif, « A / B », seulement si le texte oppose explicitement deux termes : « faire / faire
faire », « assistance / délégation », « comprendre / produire », « écrire / reformuler », « apprendre /
obtenir une réponse », « contrôle / abandon du contrôle », « usage / non-usage », « travail personnel /
production IA ».

### Schéma d'un épisode (après validation)

```json
{
  "episode_id": "ENTRETIEN_E001",
  "candidate_ids": ["ENTRETIEN_C001"],
  "turn_start": "ENTRETIEN_T0003",
  "turn_end": "ENTRETIEN_T0004",
  "practice_ids": ["ENTRETIEN_P002", "ENTRETIEN_P003"],
  "signal_ids": ["ENTRETIEN_S001"],
  "episode_status": "accountability_episode",
  "accountability_problem": "Jusqu'où l'outil peut-il intervenir dans l'écriture d'un devoir ?",
  "accounting_moves": [
    {"type": "restriction", "description": "L'étudiant limite l'usage à la reformulation.",
     "evidence_turn_ids": ["ENTRETIEN_T0004"]}
  ],
  "boundary_objects": ["écrire / reformuler"],
  "student_role_reference": null,
  "external_reference": null,
  "episode_summary": "L'étudiant accepte que l'outil reformule et exclut qu'il écrive à sa place.",
  "confidence": "high",
  "evidence": [{"turn_id": "ENTRETIEN_T0004", "quote": "…", "validation": {"valid": true, "match": "exact", "code": null}}],
  "model_needs_review": false,
  "needs_review": false,
  "speaker_warnings": [],
  "validation_status": "valid",
  "review_reasons": []
}
```

Les champs en amont de `evidence` sont produits par le modèle (schéma `EpisodeDraft`) ; `episode_id`,
`evidence[].validation`, `model_needs_review`, `needs_review`, `speaker_warnings`, `validation_status`
et `review_reasons` sont ajoutés par TRACE.

## Validateur (déterministe)

Rien n'est réécrit ; chaque anomalie a un code et une gravité. **Erreur → épisode `rejected`**
(conservé dans le fichier, exclu des comptes) ; **avertissement → `needs_review`** ; information → trace.

| Contrôle | Code(s) |
|---|---|
| identifiants existants | `UNKNOWN_PRACTICE_ID`, `UNKNOWN_SIGNAL_ID`, `UNKNOWN_CANDIDATE_ID`, `UNKNOWN_RANGE_TURN`, `FOREIGN_INTERVIEW_TURN`, `UNKNOWN_MOVE_TURN` |
| `turn_start <= turn_end` | `INVALID_TURN_RANGE` |
| citation mot pour mot | `QUOTE_NOT_FOUND`, `QUOTE_NOT_EXACT`, `EMPTY_QUOTE`, `NO_VALID_EVIDENCE` |
| voix de l'enquêté·e | `NO_INTERVIEWEE_EVIDENCE` (erreur pour un épisode), `MOVE_WITHOUT_INTERVIEWEE_TURN` |
| épisode sans opération / sans matériau substantiel | `NO_ACCOUNTING_MOVES`, `INSUFFICIENT_MATERIAL` |
| pratique ordinaire « étoffée » (≥ 2 opérations ou des frontières) | `ORDINARY_PRACTICE_WITH_ACCOUNTING` |
| incertain non signalé | `UNCERTAIN_NOT_FLAGGED`, `UNCERTAIN_CONFIDENCE` (TRACE force `needs_review`) |
| avertissements de locuteur | `SPEAKER_WARNING_PROPAGATED` (propagés par TRACE, même si le modèle les ignore) |
| vocabulaire psychologisant ou d'intention | `INTERPRETIVE_VOCABULARY` |
| « justification » sans raison dans le texte | `UNSUPPORTED_JUSTIFICATION` |
| fusions abusives | `MERGED_DIFFERENT_TASKS`, `MERGED_DIFFERENT_DOMAINS`, `INTERVENING_PRACTICE_MERGED`, `PRACTICE_NOT_IN_CANDIDATES`, `SIGNAL_NOT_IN_CANDIDATES` |
| couverture | `CANDIDATE_NOT_ADDRESSED`, `CANDIDATE_IN_SEVERAL_EPISODES` |
| contexte | `PAYLOAD_OVER_THRESHOLD`, `STAGE3_INCOMPLETE` |

**Vocabulaire interdit** (champs rédigés : problème, résumé, descriptions d'opérations, frontières,
références) : stratégie (défensive, de légitimation…), rationalisation, manipulation, mauvaise foi,
honte, peur implicite, culpabilité, identité (menacée, protection identitaire), image de soi,
légitimer / légitimation, tentative de se justifier, se justifier, défensif, se protéger, motivation
cachée, inconscient, chercher à paraître, calcul stratégique, dissimulation, malaise / gêne. Un terme
n'est pas signalé s'il figure dans une citation **d'un tour de l'enquêté·e** ; une citation de
l'enquêteur n'exempte jamais (sa catégorie ne devient pas celle de l'enquêté·e).

## Sorties

`accountability_episodes.json` : `status`, `analysis_complete`, `stage3_status`, `episode_count`,
`accountability_episode_count`, `ordinary_practice_count` (examinées par le modèle),
`unmarked_practice_count` (sans marqueur, déterministes), `uncertain_count`, `rejected_episode_count`,
`needs_review_count`, `validation_error_count`, `validation_warning_count`, `model`, `agent_version`,
`prompt_sha256`, `candidate_builder_version`, `validator_version`, `cache_hit`, `llm_called`,
`source_hashes`, le bilan des candidats (`candidates`), `unmarked_practices` et `episodes`.

`accountability_episode_manifest.json` : identité versionnée de l'agent, empreintes des sources
(fichier, transcript, sorties de l'étape 3, audit), clé et statut du cache, appels, tokens, durée,
estimation de la taille de la requête et seuil, comptes, erreur.

`accountability_episode_validation.json` : état de l'étape 3, bilan des candidats, anomalies.

## Étape 3 incomplète

| État de l'étape 3 | Étape 4 |
|---|---|
| les deux agents réussis (SUCCESS, SUCCESS_WITH_WARNINGS, CACHED) | exécutée normalement |
| un agent `PARTIAL` (bloc en échec, `analysis_complete: false`) | exécutée, statut **`PARTIAL`**, `analysis_complete: false`, anomalie `STAGE3_INCOMPLETE` — jamais présentée comme complète, même depuis le cache |
| un agent `FAILED` ou sortie absente | **`BLOCKED`** : aucun appel, aucun `accountability_episodes.json` (un fichier périmé est supprimé), le manifest et la validation disent pourquoi |
| étape 3 jamais lancée | `BLOCKED` (`stage3_status: NOT_RUN`) |

## Cache

Cache propre à l'étape 4 (`data/cache/analysis/accountability_episode_builder/`, jamais versionné).
Clé : empreinte du fichier source + empreinte du **message exact** envoyé (candidats, citations, tours,
avertissements) + nom, version, empreinte du prompt et du schéma de l'agent + modèle + paramètres.

- entretien et sorties de l'étape 3 inchangés → **0 appel** ;
- version, prompt ou schéma de l'étape 4 modifiés → nouvel appel de l'étape 4 seulement ; le cache de
  l'étape 3 n'est pas touché ;
- étape 3 relancée depuis son propre cache (mêmes objets) → la requête de l'étape 4 est identique : 0 appel ;
- objets de l'étape 3 modifiés → nouvelle requête, donc nouvel appel ;
- le validateur est recalculé à chaque fois (gratuit) : le faire évoluer ne coûte rien.

## Coût futur

- **0 appel** si un entretien n'a aucun candidat ; **1 appel** sinon (tous les candidats dans une seule
  requête) ; 0 depuis le cache.
- Entretien de référence (25 tours, 6 candidats) : ≈ 3 500 tokens d'entrée estimés pour la requête
  (consignes système en plus, mises en cache par l'API) ; la sortie réelle est à mesurer (≈ 250 tokens par épisode).
- Entretien long synthétique (320 tours, 31 pratiques, 72 signaux, 8 candidats) : ≈ 4 700 tokens
  d'entrée estimés, un seul appel.
- **Seuil documenté d'un appel unique** : `SINGLE_CALL_MAX_INPUT_TOKENS = 24 000` tokens estimés
  (longueur / 3,5) et `SINGLE_CALL_MAX_CANDIDATES = 60`. Au-delà, la requête part quand même en un
  appel, mais le manifest (`over_single_call_threshold: true`) et la validation
  (`PAYLOAD_OVER_THRESHOLD`) le signalent. Un découpage par groupes de candidats n'est **pas** implémenté :
  les tests montrent qu'un entretien long synthétique tient largement dans un appel (≈ 5 fois sous le
  seuil). À prévoir seulement si un vrai entretien dépasse le seuil.
- Sortie : ≈ 250 tokens par épisode ; 60 candidats ≈ 15 000 tokens, sous `TRACE_LLM_MAX_TOKENS` (32 000).

## Tests (aucun appel réel)

- `tests/test_accountability_candidates.py` : règles A–E, micro-marqueurs, question de l'enquêteur,
  proximité, contradiction à distance, fusion, avertissements, représentation compacte ;
- `tests/test_accountability_validator.py` : chaque contrôle du validateur ;
- `tests/test_stage4_pipeline.py` : entretien de référence (A–I), cache, invalidation, FAILED / PARTIAL /
  non lancée, échec du modèle, sortie inventée, entretien sans accountability, hygiène du dépôt ;
- `tests/test_stage4_long.py` : 320 tours, 31 pratiques → 8 candidats → 5 épisodes, 3 pratiques
  ordinaires examinées, 20 sans marqueur, 1 appel ;
- `tests/test_app_stage4.py` : interface (AppTest) ;
- `tests/e2e/browser_check.py` scénarios F et G : Chromium + Streamlit + LLM simulé.

Le LLM est simulé par `tests/fake_llm.FakeTransport` (agent `ACCOUNTABILITY`) ; le lecteur simulé
`tests/synthetic_stage4.scripted_builder` lit les candidats réellement envoyés et applique une table de
décisions. **Ces tests vérifient la chaîne et les garde-fous déterministes ; ils ne mesurent pas la
qualité d'un vrai modèle.**

## Limites

- Les candidats dépendent de la qualité de l'étape 3 : un signal manqué par l'Interaction Reader peut
  laisser une pratique « sans marqueur » alors qu'elle fait l'objet d'un travail d'accountability ; les
  pratiques sans marqueur ne sont pas examinées par le modèle.
- La proximité est positionnelle (≤ 2 positions ou même échange) : un commentaire qui revient sur une
  pratique plusieurs échanges plus tard n'est relié que par une contradiction explicite ou une même tâche.
- La règle C repose sur l'égalité des `academic_task` normalisées : deux formulations différentes d'une
  même tâche ne sont pas rapprochées ; deux tâches homonymes peuvent l'être (le modèle tranche, le
  validateur signale les fusions douteuses).
- Les motifs de frontière (règle E) et le vocabulaire interdit sont des listes lexicales courtes : ils
  ne couvrent pas toutes les formulations.
- Le validateur vérifie la forme et l'ancrage, pas la justesse de l'interprétation : un épisode
  `valid` reste une proposition à relire.
- Après une nouvelle exécution de l'étape 3, le résumé de l'étape 4 affiché reste celui de la dernière
  exécution de l'étape 4 jusqu'à ce qu'elle soit relancée (l'estimation des appels, elle, est à jour).
- Pas de découpage par candidats au-delà du seuil (voir « Coût futur »).
- Hors périmètre (étapes 5+) : trajectoire de l'étudiant·e, typologie, comparaison entre entretiens,
  régimes d'accountability, rapport.
