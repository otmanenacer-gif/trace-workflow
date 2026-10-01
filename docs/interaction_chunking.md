# Étape 3.6 — Interaction Signal Reader sur les entretiens longs

> Étape 3.7 : le Practice Extractor est découpé à son tour (même découpage, taille
> 4 000, chevauchement de 8 tours) ; l'Interaction Reader applique une règle de
> pertinence (consignes 1.3) et une couche déterministe de sélectivité avant la
> validation ; la lecture à longue distance exige deux tours CITÉS dans des blocs
> différents. Voir [`stage3_7.md`](stage3_7.md).

## Problème

Sur le premier vrai entretien long (349 tours), l'Interaction Signal Reader v1.2
a échoué : `Réponse tronquée (limite max_tokens atteinte)`. Un seul appel devait
écrire, en une réponse, **tous** les signaux de l'entretien (l'ancienne version
en produisait déjà ≈ 66) ; les consignes v1.2 demandent de n'en sélectionner
aucun (« il n'y a pas de nombre maximal de signaux ») : la sortie croît avec la
longueur de l'entretien et a dépassé `TRACE_LLM_MAX_TOKENS` (32 000 par défaut).
Le Practice Extractor et le Speaker Attribution Auditor, dont la sortie est plus
compacte, ont réussi.

Augmenter fortement `TRACE_LLM_MAX_TOKENS` ne ferait que déplacer la limite
(entretien plus long, plafond du modèle, délai). La correction rend la sortie de
chaque appel **bornée** en lisant l'entretien par blocs.

```
structured_transcript.json (jamais modifié)
        │  découpage déterministe sur les tours (core/interaction_chunking.py)
        ▼
 bloc 1   bloc 2   …   bloc n        MÊME agent, mêmes consignes, même schéma
 (T1–T78) (T73–T155)                 un appel par bloc, en parallèle, cache par bloc
        │        │          │
        └────────┴────┬─────┘
                      ▼
      lecture à longue distance      1 appel léger sur une SÉLECTION compacte de tours
                      │              (contradictions / répétitions / changements de vocabulaire
                      ▼               entre blocs différents uniquement)
      fusion déterministe (sans LLM) → validation des preuves (inchangée)
                      ▼
      interaction_signals.json (schéma inchangé) + interaction_manifest.json
```

Un entretien **court** garde exactement le comportement précédent : un seul
appel, même message, même clé de cache (les résultats déjà en cache restent valides).

## 1. Découpage (`plan_chunks`)

- Unité : le **tour structuré**, jamais une coupure au milieu d'un tour.
- Coût d'un tour : longueur de sa représentation JSON envoyée au modèle ÷ 3,5
  (≈ caractères par token en français), arrondi au supérieur. Déterministe.
- Taille cible d'un bloc : `TRACE_INTERACTION_CHUNK_TOKENS` (défaut **5 000**
  tokens estimés, de 500 à 30 000).
- Entretien d'au plus 1,4 × la cible (≈ 7 000 tokens, soit environ 100 tours
  d'un entretien ordinaire) : **un seul appel**.
- Sinon : blocs remplis jusqu'à la cible (au moins un tour, au plus 150 tours) ;
  un dernier bloc de moins de 40 % de la cible est rattaché au précédent.
- **Chevauchement** : chaque bloc après le premier reprend les **6 tours** qui le
  précèdent (≈ 3 échanges question/réponse), pour les hésitations et réparations
  à la jonction, les relations entre tours voisins et les petites contradictions
  locales. Le message du bloc le dit au modèle.
- Le découpage ignore les `speaker_warning` : il est connu avant l'audit des
  locuteurs, et l'interface peut l'annoncer avant tout appel.

**Pourquoi 5 000 tokens ?** La taille réelle de l'entretien en échec n'est pas
connue de TRACE (il n'est pas versionné) ; l'entretien synthétique de même longueur
(349 tours) fait ≈ 22 400 tokens estimés. Si l'entretien réel est comparable,
l'appel unique a produit plus de 32 000 tokens de sortie pour ≈ 22 000 tokens
d'entrée, soit un rapport sortie/entrée de l'ordre de 1,5. Un bloc de 5 000
tokens (≈ 80 tours) produirait alors ≈ 7 000 à 8 000 tokens de sortie : environ
quatre fois sous la limite, sans la modifier. Des blocs plus petits
multiplieraient les appels et réduiraient le contexte local ; plus grands, ils
réduiraient la marge.
La limite de sortie de chaque appel reste `TRACE_LLM_MAX_TOKENS` (inchangée).

Pour un entretien de 349 tours comme le premier entretien réel : **5 blocs**
(≈ 80 tours chacun) + 1 lecture à longue distance = **6 appels au plus** pour
l'Interaction Reader (l'entretien synthétique de test de même longueur, ≈ 22 400
tokens estimés, donne exactement ce découpage).

## 2. Contradictions et répétitions à longue distance

Une lecture par blocs ne peut pas rapprocher deux passages éloignés (T0010 et
T0330). Plutôt que de renvoyer l'entretien entier à un gros appel de synthèse,
une **lecture légère** reçoit une sélection compacte (`select_long_distance_turns`) :

1. les tours de l'enquêté·e cités par un signal local « de contenu » (tous les
   types sauf `hesitation`, `transcribed_laughter`, `transcribed_silence`,
   `pronoun_shift`), classés par nombre de signaux ;
2. les tours de l'enquêté·e contenant un marqueur lexical fort (« jamais »,
   « toujours », « tout le temps », « que pour », « pas du tout »,
   « uniquement », « seulement »…) ;
3. pour le contexte, la question de l'enquêteur qui précède chaque tour retenu ;

dans un budget de **6 000 tokens estimés** (≈ un bloc ; au-delà, les candidats les
moins appuyés sont écartés et comptés dans `dropped_candidate_count`). Chaque
tour porte le numéro du ou des blocs où il figure (`blocks`).

Cette lecture a ses propres consignes (`prompts/interaction_long_distance_reader.md`)
et un schéma **restreint** à `cross_turn_contradiction`, `significant_repetition`
et `vocabulary_shift` ; elle ne fait ni interprétation, ni typologie, ni synthèse.
TRACE ne garde que les signaux dont les tours ne tiennent dans **aucun** bloc
(`is_long_distance`) : les autres ont déjà été lus localement
(`signals_discarded_not_long_distance`).

Elle n'est appelée que si tous les blocs ont réussi (sa sélection dépend de leurs
signaux) et s'il existe au moins deux candidats qu'aucun bloc ne contient
ensemble ; sinon, aucun appel (`NOT_NEEDED` ou `SKIPPED_INCOMPLETE`).

## 3. Fusion déterministe (`merge_signals`)

Aucun LLM. Deux signaux de deux blocs **différents** sont le même signal si :

- même `signal_type` ;
- mêmes `turn_ids` ;
- tous ces tours appartiennent aux **deux** blocs (zone de chevauchement) ;
- et : même `surface_form` normalisée (casse, apostrophes, espaces), **ou** mêmes
  citations normalisées, **ou** `surface_form` incluses l'une dans l'autre ET une
  citation du même tour incluse dans l'autre.

On garde le premier relevé, enrichi de **toutes** les citations (sans doublon
exact), de tous les `turn_ids`, de `needs_human_review` (OU logique) et de la
provenance. Deux signaux d'un même bloc ne sont jamais fusionnés (l'agent les a
distingués). Dans le doute, les deux sont conservés : mieux vaut un doublon
visible qu'un signal perdu.

Ordre final : ordre de l'entretien (premier tour cité), puis bloc, puis rang dans
le bloc. Les identifiants `<ENTRETIEN>_S001…` sont ensuite attribués par la
validation des preuves, comme avant : mêmes entrées → mêmes identifiants.

Chaque signal d'un entretien long porte une provenance interne :
`"provenance": {"pass": "local" | "long_distance", "chunks": [2, 3], "merged_duplicates": 1}`.

## 4. Validation des preuves

Inchangée : `evidence_validation.json` est calculé sur le résultat **fusionné**,
exactement comme pour un appel unique (citation exacte, `turn_id` existant,
interpretation guard, `needs_review`, `EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN`…).
Le bilan de l'Interaction Reader y porte en plus `analysis_complete`.

## 5. Avertissements de locuteur

Les `speaker_warning` de l'audit (étape 3.5) sont transmis **uniquement** aux
appels dont l'extrait contient le tour concerné : le bloc (ou les deux blocs, si
le tour est dans un chevauchement) et, si le tour est sélectionné, la lecture à
longue distance. Le transcript original n'est jamais modifié.

## 6. Cache

Cache **par bloc** : la clé d'un bloc (et de la lecture à longue distance) est
l'empreinte du message exact envoyé + l'identité de l'agent (consignes + gabarit
du message de bloc, schéma, version) + modèle + paramètres. Elle ne dépend pas de
l'empreinte du fichier entier :

- relance identique → **aucun appel** (statut `CACHED`) ;
- un tour modifié → seul son bloc (et ceux dont les frontières bougent, s'il
  change la taille estimée au point de déplacer une frontière) est rappelé, puis
  la lecture à longue distance si sa sélection change ;
- un bloc en échec n'est jamais mis en cache ; les blocs réussis le sont.

Le cache de l'auditeur n'est pas touché. (Étape 3.7 : le Practice Extractor a
désormais, pour un entretien long, le même cache par bloc.)

## 7. Bloc tronqué ou en échec

- Le bloc est marqué `TRUNCATED` (ou `FAILED`) dans `chunking.chunks`.
- Le statut global devient **`PARTIAL`** (`analysis_complete: false` dans le
  document, le manifest et `evidence_validation.json`) : le résultat n'est
  jamais présenté comme complet. `error.code = PARTIAL_ANALYSIS` détaille les
  blocs manquants.
- Les signaux des blocs réussis sont écrits, validés et mis en cache.
- La lecture à longue distance est reportée (`SKIPPED_INCOMPLETE`).
- Relancer l'analyse ne rappelle que le(s) bloc(s) manquant(s) et la lecture à
  longue distance. Si un bloc reste tronqué, réduire `TRACE_INTERACTION_CHUNK_TOKENS`.
- Si **tous** les blocs échouent : `FAILED`, sans sortie (comme avant).

## 8. Provenance (manifest et document)

`interaction_manifest.json` et `interaction_signals.json` portent un objet
`chunking` (toujours présent, `chunking_used: false` pour un entretien court) :
`chunking_used`, `chunk_count`, `chunk_ranges` (premier/dernier tour, tours,
chevauchement, tokens estimés), `overlap_turns`, `target_tokens`,
`estimated_tokens`, `llm_calls_this_run`, `interaction_calls_max`,
`chunks_succeeded`, `failed_chunks`, `truncated_chunks`, `signals_before_dedup`,
`signals_after_dedup`, `duplicates_removed`, le détail de chaque bloc (statut,
clé de cache, tokens, `stop_reason`, nombre d'avertissements transmis…) et de la
lecture à longue distance. Le schéma public des signaux est inchangé (seul le
champ interne `provenance` s'ajoute, pour un entretien long).

## 9. Interface

Avant exécution : « Interaction Reader — <entretien> : entretien long : analyse
en 5 blocs (chevauchement inclus), puis 1 lecture à longue distance, soit au
plus 6 appels Interaction Reader », et le total maximal pour la sélection.
Après exécution : colonne « Blocs (Interaction) » (`5/5`), blocs réussis,
signaux finaux (et avant dédoublonnage), état de la lecture à longue distance,
blocs tronqués en rouge. Le téléchargement de `interaction_signals.json` est inchangé.

## 10. Limites connues

- L'estimation des tokens est une approximation (3,5 caractères par token) : un
  bloc réel peut être un peu plus grand ou plus petit que prévu.
- Un tour unique plus long que la limite de sortie ne peut pas être découpé.
- La lecture à longue distance ne voit qu'une sélection : une contradiction entre
  deux passages qu'aucun bloc n'a signalés et qui ne contiennent aucun marqueur
  lexical fort peut lui échapper.
- La déduplication est volontairement prudente : deux relevés du même phénomène
  formulés très différemment par deux blocs restent deux signaux.
- Une contradiction locale située dans un chevauchement peut être relevée par
  les deux blocs avec des citations différentes et rester en double.
- Tant que l'audit des locuteurs n'est pas en cache, le nombre d'appels annoncé
  est un maximum (« au plus »).
