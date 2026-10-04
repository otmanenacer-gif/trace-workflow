# Étape 4 — Épisodes d'accountability

> **Exécution** : les agents sont exécutés localement par un modèle servi par Ollama
> ([`local_runtime.md`](local_runtime.md)), sans aucune API ni clé. Dans ce document, un « appel » désigne un
> appel au modèle local ; les mentions de tokens facturés, de cache de l'API ou de `TRACE_LLM_MAX_TOKENS`
> décrivent la conception d'origine (exécution par API, retirée) : `api_calls` et `billed_this_run` valent
> toujours 0 / `false`.

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

Si les sorties de l'étape 3 ont disparu (stockage local effacé par un redéploiement) mais ont été
téléchargées, elles peuvent être **restaurées** depuis l'interface sans aucun appel (section « Restaurer
des résultats Stage 3 existants », `core/stage3_restore.py`, voir le README) : l'étape 4 les lit alors
comme des sorties normales.

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

- **0 appel** si un entretien n'a aucun candidat ; sinon **un appel par bloc d'au plus 4 candidats**
  (`BLOCK_MAX_CANDIDATES`, étape 4.2, ci-dessous) ; 0 depuis le cache.
- Estimations d'entrée (longueur du message / 3,5 ; consignes système en plus, mises en cache par l'API) :

| Entretien synthétique | Candidats | Format 1 (4.0) | Format 2 normalisé (4.1) | Réduction | Appels |
|---|---|---|---|---|---|
| référence (25 tours) | 6 | ≈ 3 470 | ≈ 2 450 | −29 % | 1 |
| long (320 tours, 31 pratiques, 72 signaux) | 8 | ≈ 4 670 | ≈ 3 000 | −36 % | 1 |
| régression « OTMANE » (388 tours, 50 pratiques, 76 signaux) | 26 | ≈ 13 500 | ≈ 8 900 | −34 % | 1 |

  Ces réductions comparent les deux formats sur les MÊMES candidats. Sur l'entretien « OTMANE »
  synthétique, l'ancien pipeline complet (constructeur 4.0, qui rattachait un signal à toutes les
  pratiques d'un long tour : 38 candidats) produisait ≈ 19 100 tokens, soit −53 % au total.
- **Seuils documentés d'un appel unique** : `SINGLE_CALL_MAX_INPUT_TOKENS = 24 000` tokens estimés pour le
  message complet et `SINGLE_CALL_MAX_CANDIDATES = 60`. Au-delà (entretiens réellement très gros
  seulement), la requête est découpée en blocs de **composantes entières** (jamais au milieu d'une
  composante), un appel et une entrée de cache par bloc ; un bloc en échec rend le résultat `PARTIAL`.
  Une composante qui dépasse seule le seuil part en un bloc et est signalée (`PAYLOAD_OVER_THRESHOLD`).
  `max_tokens` n'est jamais augmenté pour compenser.

### 4.2 — Blocs bornés par la réponse

Mesure réelle (OTMANE_NACER, qwen2.5:7b) : la réponse de l'Accountability Episode Builder est dominée par les
citations recopiées mot pour mot (`evidence` : 77 % du JSON ; 4 à 9 citations par épisode, souvent des tours
entiers) : ≈ 545 tokens par épisode en moyenne, ≈ 1 080 au plus. 12 à 15 candidats en un appel demandent
≈ 7 000 tokens pour une réponse réservée de 6 144 (`OUTPUT_RESERVE`, core/local_agent_runner.py) : réponse tronquée,
redemandée en entier, puis correction méthodologique renvoyant toute la réponse (journal du 2026-10-03 : 4
tentatives, 2 troncatures, 693 s).

- Découpage : **au plus `BLOCK_MAX_CANDIDATES = 4` candidats par appel** (≈ 2 200 tokens de sortie en moyenne,
  ≈ 4 400 au pire, sous la réserve), en **composantes entières** dans l'ordre de l'entretien ; une composante
  plus grande forme seule son bloc. Chaque candidat figure dans exactement un bloc ; chaque bloc reçoit la
  représentation normalisée de ses seuls candidats et la consigne « Bloc i/n » du gabarit existant.
- Persistance : la réponse validée d'un bloc est mise en cache dès sa validation (clé : le message exact du bloc).
- Reprise : un bloc tronqué, en échec ou interrompu rend l'étape `PARTIAL` (épisodes des autres blocs conservés)
  ; la relance ne refait que les blocs manquants.
- Fusion : épisodes de tous les blocs réunis, puis validation habituelle sur l'ensemble (identifiants
  `E001…` attribués dans l'ordre de l'entretien) : même document que l'appel unique.
- Coût mesuré sur OTMANE_NACER (30 pratiques / 22 signaux, 12 candidats) : 3 appels au lieu de 1 ; entrée
  ≈ 28 000 tokens au lieu de ≈ 20 000 (consignes répétées : quelques secondes de lecture) ; sortie inchangée
  (≈ 6 500 tokens) ; durée estimée ≈ 207 s au lieu de ≈ 203 s sans troncature — contre 693 s mesurées avec.
  Estimation et mesure : `python scripts/stage4_blocks_report.py plan RUN` (sans appel) et
  `python scripts/stage4_blocks_report.py measure RUN` (journaux de la dernière étape 4).

### 4.3 — Réparations ciblées (core/stage4_repair.py)

Invariant : chaque candidat reçoit exactement une disposition (`accountability_episode`, `ordinary_practice`,
`uncertain`) ; un épisode ne porte que les pratiques et signaux de ses candidats. Un bloc validé n'est jamais
régénéré pour une anomalie locale (le contrôle méthodologique du runner ne redemande plus un bloc de l'étape 4) :

- identifiant d'un autre candidat sans appui dans l'épisode (aucune citation ni opération sur un tour qui
  n'appartient qu'à lui) : retiré sans modèle et noté (`removed_foreign_ids`, information `FOREIGN_ID_REMOVED`) ;
  s'il appuie l'épisode, l'épisode est réparé ;
- épisode avec une anomalie bloquante (ex. `NO_ACCOUNTING_MOVES`, citation absente, identifiant étranger utile) :
  UNE réparation sur ses seuls candidats — l'épisode, les erreurs exactes, et la consigne de ne rien inventer
  (une opération réellement appuyée, sinon `ordinary_practice` ou `uncertain`) ;
- candidat sans disposition (`CANDIDATE_NOT_ADDRESSED`, désormais bloquant) : UNE demande sur ce seul candidat.

Le message de réparation est le gabarit habituel de l'agent (prompt système et schéma inchangés) sur la
représentation normalisée des seuls candidats concernés, suivi des consignes de réparation ; sa réponse validée est
mise en cache comme un bloc. Une réparation n'est retenue que si elle traite chaque candidat demandé exactement une
fois sans anomalie bloquante ; l'épisode réparé porte `trace_repair` (type, anomalies). Bilan dans le manifest et la
validation (`repairs`). Les avertissements et informations (`SPEAKER_WARNING_PROPAGATED`…) n'appellent jamais le
modèle. Une étape 4 validée par une autre version du validateur est périmée pour la garde : la relancer reprend les
blocs du cache et ne demande que les réparations. Prévision sans appel :
`python scripts/stage4_blocks_report.py repairs RUN`.

## Étape 4.1 — stabilisation après le premier run réel

Le premier run réel (OTMANE_NACER, 388 tours) a révélé trois problèmes ; correctifs ciblés, sans refonte :

1. **Proximité trop large dans un même tour.** Un signal était rattaché à toutes les pratiques d'un même tour,
   même quand leurs citations étaient éloignées dans un long tour (T0028). Désormais, si le tour du signal porte
   des pratiques, la proximité est mesurée sur les citations exactes, localisées dans le texte du tour :
   au plus `INTRA_TURN_MAX_GAP_CHARS = 200` caractères (≈ 2 phrases, 35 mots) entre les deux passages ;
   sinon aucun rattachement (`FAR_WITHIN_TURN`). Les déclencheurs explicites (frontière, usage / non-usage d'un
   même passage, contradiction, polarité) ne dépendent pas de cette règle.
2. **Fusions trop permissives.** Deux candidats ne sont reliés que par une relation explicite
   (`candidate_links`) : A. pratique partagée ; B. signal déclencheur partagé ; C. mêmes tours cités ET même
   tâche normalisée ; D. contradiction entre tours qui cite les tours de l'autre. Chaque candidat porte un
   `component_id` (composante connexe ; une chaîne reliée est acceptée). Le modèle reçoit ces composantes et
   ne peut pas les réunir ; le validateur le vérifie : un épisode dont les candidats ne forment pas un graphe
   connexe reçoit l'erreur `DISCONNECTED_MERGE`, est `rejected`, `needs_review`, et
   `usable_for_next_stages: false` (`usable_episode_count` ne le compte pas). Proximité chronologique, thème
   commun (« un rendu ») ou même outil (« ChatGPT ») ne relient jamais deux candidats.
3. **Requête trop grosse** (≈ 35 800 tokens estimés pour 28 candidats). Format 2 normalisé : `practices_by_id`,
   `signals_by_id`, `evidence_by_id` (chaque citation une fois, référencée par `Q001`…), `turns_by_id`,
   candidats réduits à des identifiants ; préfixe commun des identifiants déclaré une fois (`id_prefix`) et
   omis partout, puis rétabli de façon déterministe dans la réponse avant validation (`expand_ids`) ; champs
   vides omis ; descriptions rédigées par l'Interaction Reader non envoyées (type, forme et citations le
   sont) ; JSON compact. Découpage par composantes au-delà des seuils (ci-dessus).

**Vocabulaire.** Affects et états intérieurs (culpabilité / coupable, honte, peur / crainte, anxiété, gêne,
embarras, intention, motivation, légitimité / légitimation…) ne sont jamais une catégorie d'analyse, sauf
s'ils figurent dans une citation **de l'enquêté·e** associée à l'épisode (épisode, pratiques, signaux). Un
terme qui ne vient que d'une question de l'enquêteur est en outre signalé `INTERVIEWER_TERM_ATTRIBUTED`.
Pratique ordinaire : écrire « Aucune restriction, justification ou évaluation explicite n'est relevée dans ce
passage. » (cette tournure négative n'est pas une « justification »), jamais « sans réserve portant sur sa
légitimité ». Prompt 1.1, agent 1.1, constructeur de candidats 1.1, validateur 1.1 : le cache de l'étape 4
est invalidé, celui de l'étape 3 ne l'est pas.

## Tests (aucun appel réel)

- `tests/test_accountability_candidates.py` : règles A–E, micro-marqueurs, question de l'enquêteur,
  proximité, contradiction à distance, fusion, avertissements, représentation compacte ;
- `tests/test_accountability_validator.py` : chaque contrôle du validateur ;
- `tests/test_stage4_pipeline.py` : entretien de référence (A–I), cache, invalidation, FAILED / PARTIAL /
  non lancée, échec du modèle, sortie inventée, entretien sans accountability, hygiène du dépôt ;
- `tests/test_stage4_long.py` : 320 tours, 31 pratiques → 8 candidats → 5 épisodes, 3 pratiques
  ordinaires examinées, 20 sans marqueur, 1 appel ;
- `tests/test_app_stage4.py` : interface (AppTest) ;
- `tests/test_stage4_1.py` : proximité intra-tour, relations et composantes, fusions déconnectées,
  vocabulaire sensible au locuteur, identifiants abrégés, coût (régression « OTMANE » synthétique,
  `tests/synthetic_stage4_otmane.py`), découpage par composantes, bloc en échec ;
- `tests/e2e/browser_check.py` scénarios F, G, H et I : Chromium + Streamlit + agents simulés.

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
- Une composante de plus de 4 candidats n'est jamais coupée : son bloc peut dépasser la réponse estimée.
- Une réparation ciblée ratée laisse l'épisode d'origine (rejeté) ; un candidat toujours sans disposition laisse
  l'étape 4 `PARTIAL` (une seule réparation par objet : la relance ne la redemande pas).
- Hors périmètre (étapes 5+) : trajectoire de l'étudiant·e, typologie, comparaison entre entretiens,
  régimes d'accountability, rapport.
