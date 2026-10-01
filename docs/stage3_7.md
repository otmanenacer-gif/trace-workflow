# Étape 3.7 — stabilisation de l'étape 3 sur les entretiens longs

Dernier patch de l'étape 3 avant l'étape 4. Rien de l'étape 4 n'est construit ici.

## Problèmes observés sur le vrai entretien long

| Agent | Résultat réel |
|---|---|
| Speaker Attribution Auditor | SUCCESS — 4 tours suspects, 4 à vérifier |
| Interaction Reader | SUCCESS_WITH_WARNINGS — 6/6 blocs, longue distance SUCCESS, **498 signaux** avant dédoublonnage, **478** après, **24 anomalies** de validation |
| Practice Extractor | **FAILED — réponse tronquée (max_tokens)** |
| Run | ≈ 125 586 tokens d'entrée / 193 610 de sortie, 9 appels |

### Cause 1 — Practice Extractor tronqué

Le Practice Extractor faisait **un seul appel par entretien** : la réponse devait
contenir, en une fois, **toutes** les pratiques de l'entretien. Depuis la
version 1.2, ses consignes demandent (à juste titre) de chercher
systématiquement usages ET non-usages, refus, usages passés, usages personnels
et professionnels : un long entretien donne plusieurs dizaines de pratiques, et
chacune est un objet de ≈ 20 champs avec citations (plusieurs centaines de
tokens). La sortie croît avec la longueur de l'entretien et a dépassé
`TRACE_LLM_MAX_TOKENS` (32 000). L'étape 3.6 n'avait découpé que l'Interaction Reader.

### Cause 2 — surcodage de l'Interaction Reader

≈ 83 signaux par bloc (≈ 26 000 tokens de sortie par bloc, d'après le total du
run : déjà près de la limite). Les consignes 1.2 décrivaient les types de façon
**lexicale** et poussaient à l'exhaustivité :

- « Il n'y a pas de nombre maximal de signaux : ne les sélectionne pas selon leur
  importance et ne fusionne pas des signaux distincts » ;
- « Un même passage peut porter plusieurs signaux : fais un objet par signal » ;
- `hesitation` = « euh », « bah », « ben », points de suspension (tous) ;
  `minimization` = « juste », « seulement », « un peu » (tous) ;
- l'exemple « Euh… enfin… je l'utilise juste pour reformuler » autorisait
  explicitement trois signaux ;
- le message de bloc demandait : « Relève tous les signaux présents dans l'extrait ».

Le modèle a donc fait un inventaire linguistique : un signal par « euh », par
« juste », par « un peu »…

### Causes probables des 24 anomalies de validation

Les sorties réelles ne sont pas versionnées : les causes ci-dessous sont
déduites du code et des consignes, pas d'une lecture du run réel.

- `NO_INTERVIEWEE_EVIDENCE` : signaux appuyés sur la seule question de
  l'enquêteur ; et le validateur ignorait les `speaker_warning` (un tour
  « enquêteur » signalé par l'audit comme réponse probable de l'enquêté·e
  déclenchait l'avertissement).
- `EVIDENCE_TURN_NOT_LISTED` : citation du tour voisin (souvent la question)
  sans l'ajouter à `turn_ids` ; rien ne complétait `turn_ids` avant validation.
- `METADISCURSIVE_NO_SELF_REFERENCE` : citation réduite au fragment évaluatif
  (« c'est un peu facile ») sans « ce que je dis », alors que la phrase le
  contient ; ou évaluation de l'outil / d'une règle classée métadiscursive.
- `AFFECT_NOT_IN_QUOTES` : affect normalisé par le modèle (« énervement » pour
  « ça m'énerve », « j'avais peur » pour « j'ai peur ») avec une comparaison qui
  n'admettait que la variation d'une lettre finale ; ou affect inféré.
- `INTERPRETIVE_VOCABULARY` : noms qui qualifient l'opération de l'enquêté·e
  (« justification », « stratégie »…) dans `description`.

## A. Practice Extractor : lecture par blocs

Même agent, **mêmes consignes système** (prompt inchangé, version 1.2), même
schéma, mêmes catégories (`use`, `past_use`, `non_use`, `refusal`,
`hypothetical`, `practice_domain`, `non_use_reason`) : seul le message
utilisateur d'un bloc annonce un extrait (`CHUNK_USER_TEMPLATE`).

- Découpage entre les tours, jamais au milieu d'un tour (utilitaires de
  l'étape 3.6, `core/interaction_chunking.plan_chunks`, réutilisés par
  `core/practice_chunking.py`).
- **Taille cible : `TRACE_PRACTICE_CHUNK_TOKENS` = 4 000 tokens estimés**
  (≈ 3,5 caractères par token ; ≈ 60 à 90 tours selon la longueur des tours).
  Plus petite que celle de l'Interaction Reader (5 000) : une pratique est plus
  longue qu'un signal, et le rapport sortie / entrée du Practice Extractor
  n'est connu que par sa borne basse (l'appel unique a dépassé 32 000 tokens).
  Un bloc de 4 000 tokens d'entrée devrait produire de l'ordre de 4 000 à
  12 000 tokens de sortie : 2,5 à 8 fois sous la limite, **qui n'est pas augmentée**.
- Entretien de moins de 1,4 × la cible (≈ 5 600 tokens, ≈ 85 à 130 tours) : **un
  seul appel, exactement comme avant** (même message, même clé de cache).
- **Chevauchement : 8 tours** (≈ 4 échanges question/réponse), annoncé au
  modèle : une situation racontée à la jonction reste lisible en entier.
- Un appel par bloc ; aucune lecture complémentaire (le Practice Extractor
  n'interprète pas les contradictions).

| Entretien | Tokens estimés | Blocs Practice | Appels Practice au plus |
|---|---|---|---|
| court | ≤ 5 600 | 1 | 1 |
| synthétique 3.7 (330 tours) | ≈ 14 200 | 4 | 4 |
| synthétique 3.6 (349 tours) | ≈ 22 400 | 6 | 6 |
| type « Barnabé » (6 blocs Interaction à 5 000) | ≈ 27 000–32 000 | 7–8 | 7–8 |

### Relations distantes sans interprétation

Chaque bloc décrit **chaque conduite déclarée dans l'extrait**, même si
l'enquêté·e dit autre chose ailleurs (le message de bloc le rappelle) ; la
fusion ne réunit jamais deux conduites différentes. « Il m'arrive de lui faire
rédiger » (T0020) et « Normalement mes travaux je les fais moi-même » (T0150)
donnent donc deux pratiques, `use` et `non_use`, sans aucune lecture de l'écart
(ni « il se justifie », ni « il répare »).

Garde-fou léger et informatif (`non_use_cues`) : les tours de l'enquêté·e qui
contiennent une formulation typique de non-usage (« je ne l'utilise pas »,
« moi-même », « à ma place », « je refuse », « je préfère le faire »…) et
qu'aucune pratique `non_use` / `refusal` ne couvre sont listés dans
`practice_extractor.json` (`non_use_cues.uncovered_turn_ids`) et signalés dans
l'interface, pour relecture. Rien n'est ajouté ni corrigé.

## B. Practice Extractor : fusion et dédoublonnage (`merge_practices`)

Sans LLM. Deux pratiques ne sont un doublon que si **toutes** ces conditions
sont réunies :

1. elles viennent de deux blocs **différents** (deux pratiques d'un même bloc
   ne sont jamais fusionnées : l'agent les a distinguées) ;
2. même `use_status` — jamais `use`/`non_use`, `use`/`refusal`,
   `past_use`/`non_use`… ;
3. même `non_use_reason`, même `practice_domain` (jamais études / vie personnelle) ;
4. `assessment_context` identique ou inconnu d'un côté ; `academic_task`
   identique, inclus l'un dans l'autre ou absent d'un côté ; `ai_tool` commun
   ou absent d'un côté ;
5. un **ancrage commun** : une citation de chacune, du **même tour**, situé dans
   la zone commune aux deux blocs, l'une incluse dans l'autre (casse,
   apostrophes, blancs normalisés).

Chaque pratique absorbe au plus une pratique d'un autre bloc ; les paires sont
appariées de la plus appuyée (nombre d'ancrages, tâche identique, résumés
proches) à la moins appuyée. Dans le doute, les deux pratiques restent.

Pratique conservée : la première relevée, enrichie de **toutes les citations**,
de l'intervalle `turn_start`–`turn_end` le plus large, des éléments de liste
nouveaux, de la lecture la plus prudente (`explicitness` : `unclear` l'emporte),
de **toutes** les `uncertainty_note` et d'une provenance
(`"provenance": {"chunks": [1, 2], "merged_duplicates": 1}`). `needs_review`
est recalculé par la validation des preuves sur le résultat fusionné.
Identifiants finaux stables : ordre de l'entretien (`turn_start`), puis bloc,
puis rang ; `P001`, `P002`… attribués par la validation.

## C. Interaction Reader : sélectivité (consignes 1.3)

Schéma inchangé (1.2, mêmes types) ; consignes réécrites :

- **Règle centrale** : un élément de langage ne devient un signal que s'il
  intervient dans la manière dont l'enquêté·e décrit sa conduite, la limite, la
  reformule, l'évalue (ou évalue sa façon de la dire), dit ce qui la rend
  acceptable ou non à ses yeux, distingue des usages, exprime un trouble, une
  réserve ou une préférence, modifie une affirmation, rend compte d'un usage ou
  d'un non-usage. « Tu ne fais pas un inventaire linguistique. »
- « **Préférer un signal plus substantiel à plusieurs micro-signaux redondants.** »
  Ni quota ni plafond (supprimé : « il n'y a pas de nombre maximal de signaux…
  ne les sélectionne pas »).
- **Micro-marqueurs** (« euh », « ben », « voilà », « enfin », « juste », « un
  peu », « on va dire »…) : jamais relevés seuls. Hésitation seulement si elle
  accompagne une reformulation importante, précède un changement de position,
  marque une difficulté à qualifier sa pratique ou participe à une réponse
  délicate sur un usage / non-usage. Minimisation, intensification,
  modalisation : seulement si elles bornent la présentation de la pratique ;
  une intensité descriptive (« énormément ») n'est pas un signal à elle seule.
  `surface_form` reprend le marqueur **avec ce qu'il borne** (« juste reformuler,
  jamais écrire »).
- Plusieurs signaux sur un passage seulement pour des **opérations différentes**
  (« J'utilise pas ça normalement… enfin sauf si je suis vraiment en retard » →
  `self_correction` + `exception`, rien de plus).
- Exemples négatifs (« Ben oui. », « Euh je l'utilise beaucoup. », « J'utilise
  juste ChatGPT pour chercher des films. ») et positifs (autocorrection +
  exception, contraste, restriction, `metadiscursive_self_evaluation`,
  `reference_to_teacher_judgment` + `explicit_emotion`, `preference_statement`,
  « je ne veux pas qu'il réfléchisse à ma place » → `normative_formulation`).
- Métadiscours : référence **explicite** à sa propre parole, citer la phrase
  qui la contient ; « ChatGPT est ridicule », « cette règle est absurde », « ce
  devoir est nul » n'en relèvent pas.
- `explicit_affect` : mot présent **mot pour mot** dans une citation, sinon `null`.
- `turn_ids` contient **chaque** tour cité ; au moins une citation d'un tour
  `enquete` (ou d'un tour signalé par un `speaker_warning`, avec
  `needs_human_review`) ; une question de l'enquêteur n'est jamais une position
  de l'enquêté·e.
- Vocabulaire : verbes de constat (« dit », « ajoute », « précise », « corrige »,
  « oppose »…), aucun nom qui qualifie l'opération (« justification »,
  « stratégie », « défense », « légitimation », « normalisation »,
  « rationalisation ») sauf s'il est employé par l'enquêté·e. Les concepts
  théoriques ne sont toujours pas nommés dans les consignes.
- Message de bloc : « relève les signaux pertinents pour le récit des
  pratiques, pas chaque marqueur » (au lieu de « relève tous les signaux »).

Lecture à longue distance (consignes 1.1) : seulement `cross_turn_contradiction`,
`significant_repetition`, `vocabulary_shift` entre deux passages réellement
éloignés ; contradiction = deux propositions sur **le même objet**, réellement
incompatibles ou en forte tension ; « je l'utilise beaucoup » / « je l'utilise
surtout pour réviser » n'en est pas une ; en cas de doute, rien. Côté TRACE, un
signal n'est gardé que si **les tours cités** (au moins deux) ne tiennent dans
aucun bloc ; un micro-marqueur nu ne rend plus un tour candidat à la sélection.

## D. Couche déterministe avant validation (`core/signal_selectivity.py`)

Appliquée au résultat de l'agent (entretien court) ou fusionné (entretien long),
**avant** la validation des preuves. Rien n'est supprimé en silence : chaque
signal écarté est conservé dans `interaction_signals.json`
(`set_aside_signals`, avec `set_aside_reason` et la vérification de ses
citations) et compté (`selectivity`).

1. **`turn_ids` complétés** par les tours cités existants (`turn_ids_added`) :
   plus de `EVIDENCE_TURN_NOT_LISTED` évitable. La fusion des blocs ajoute aussi
   le tour de toute citation fusionnée.
2. **`INTERVIEWER_ONLY_EVIDENCE`** : toutes les citations (valides) viennent de
   tours `enqueteur` sans `speaker_warning` → écarté. Un tour signalé par
   l'audit reste un appui possible (à revoir).
3. **`ISOLATED_MICRO_MARKER`** : type `hesitation`, `minimization`,
   `intensification`, `modalization`, `self_correction`, `self_reformulation`
   ou `other` ; `surface_form` faite **uniquement** de remplisseurs ou d'adverbes
   de degré ; **aucun autre signal sur le même tour**.
4. **`REDUNDANT_MICRO_MARKER`** : même définition, mais le marqueur figure déjà
   dans la `surface_form` d'un autre signal du même tour (« juste » seul à côté
   de la restriction « juste reformuler, jamais écrire »).

Jamais écarté : un signal avec `needs_human_review`, `explicit_affect`,
`cross_turn_reference` ou une citation invalide (il doit être vu et validé) ; un
« Euh… » sur un tour qui porte aussi une autocorrection « Enfin non » (il
accompagne une reformulation). Ce n'est **pas un quota** : aucun plafond,
aucun tri par importance.

## E. Validation (validator 1.2, garde-fou 1.3)

- `NO_INTERVIEWEE_EVIDENCE` n'est plus émis quand l'appui « enquêteur » est un
  tour signalé par l'audit des locuteurs (`EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN`,
  objet à revoir) ; il l'est toujours sinon (et pour une pratique).
- `METADISCURSIVE_NO_SELF_REFERENCE` : la référence à sa propre parole est
  cherchée dans la **phrase** du tour qui contient la citation (« c'est un peu
  facile » dans « Bon, c'est un peu facile ce que je dis, je sais. ») ; « ChatGPT
  est ridicule. » reste signalé.
- `AFFECT_NOT_IN_QUOTES` : comparaison par radical (6 lettres au plus, mots-outils
  et auxiliaires ignorés) : « énervement » ↔ « m'énerve », « j'avais peur » ↔
  « j'ai peur », « stressé » ↔ « stressée » ; « honte » pour « Euh… oui » reste signalé.
- `INTERPRETIVE_VOCABULARY` : ajout de « légitimation », « normalisation » (deux
  agents descriptifs) et « défense » (Interaction Reader) ; l'auditeur des
  locuteurs n'est pas concerné. Un mot employé par l'enquêté·e (présent dans
  les citations) n'est pas signalé.

## F. Cache

- Practice Extractor : entretien court → clé historique inchangée ; entretien
  long → **cache par bloc** (empreinte du message exact + identité de l'agent),
  comme l'Interaction Reader : relance identique = 0 appel ; un tour modifié ne
  rappelle que son bloc (et ceux dont les frontières bougeraient) ; un bloc en
  échec n'est jamais mis en cache.
- Interaction Reader : le changement de consignes (1.3) invalide une fois les
  résultats en cache de l'Interaction Reader (nouvelle empreinte de prompt) ;
  ensuite, même comportement qu'à l'étape 3.6. La sélectivité est recalculée à
  chaque lecture du cache (déterministe, gratuite).
- Cache de l'auditeur des locuteurs : non touché.
- Cache de prompt de l'API : consignes système identiques d'un bloc à l'autre,
  en tête du message (`cache_control`) : réutilisables entre les blocs d'un agent.

## G. Statuts

`SUCCESS`, `SUCCESS_WITH_WARNINGS`, `PARTIAL`, `FAILED` (et `CACHED`) pour les
deux agents. Un bloc Practice tronqué → `PARTIAL`, `analysis_complete: false`
(document, manifest, `evidence_validation.json`), message
`PARTIAL_ANALYSIS` (« Analyse INCOMPLÈTE : 3/4 bloc(s) réussi(s)… réduisez
TRACE_PRACTICE_CHUNK_TOKENS ») : la liste n'est jamais présentée comme
complète ; relancer ne refait que le bloc manquant. Tous les blocs en échec →
`FAILED`, sans sortie. Le nombre d'anomalies de validation
(`validation_issue_count` : erreurs + avertissements) est affiché.

## H. Interface

Avant exécution : « Practice Extractor — X : entretien long : analyse en N blocs
(chevauchement inclus), soit au plus N appels » et la ligne équivalente pour
l'Interaction Reader, plus « Practice Extractor : au plus … appel(s) ·
Interaction Reader : au plus … appel(s) ». Après : colonnes « Anomalies
validation », « Blocs (Practice) », « Blocs (Interaction) » ; dans l'aperçu,
blocs réussis, pratiques / signaux finaux et avant dédoublonnage, statut,
lecture à longue distance, anomalies de validation, signaux écartés, indices de
non-usage non couverts. Blocs tronqués en rouge, par agent.

## I. Estimation des appels (au plus, avant cache)

Entretien type « Barnabé » (lu en 6 blocs par l'Interaction Reader, donc
≈ 27 000 à 32 000 tokens estimés) :

| Étape | Appels |
|---|---|
| Speaker Attribution Auditor | 1 (0 sans tour suspect) |
| Practice Extractor | 7–8 (un par bloc de 4 000) |
| Interaction Reader | 6 (un par bloc de 5 000) |
| Lecture à longue distance | 1 |
| **Total** | **≈ 15–16 au plus** ; **0** à la relance identique |

Plus d'appels qu'avant (9), mais chacun borné : aucune réponse ne doit plus
approcher la limite de 32 000 tokens, et les tokens de sortie de l'Interaction
Reader — l'essentiel du coût du run réel — doivent baisser nettement avec la
règle de pertinence. Aucun chiffre réel n'est garanti : ces tests n'appellent
pas l'API.

## J. Limites connues

- Les tests vérifient les consignes et la couche déterministe ; la sélectivité
  **réelle** du modèle ne sera mesurée qu'au prochain run réel.
- La règle des micro-marqueurs est lexicale et prudente : un micro-signal
  redondant formulé autrement (« un petit peu » à côté de « un peu facile ») ou
  un `contrast` réduit à « mais » ne sont pas écartés.
- Une pratique racontée sur plus de 8 tours à cheval sur deux blocs peut donner
  deux descriptions partielles sans citation commune : elles restent deux
  pratiques (aucune fusion sans ancrage).
- Les indices de non-usage (`non_use_cues`) sont lexicaux : faux positifs et
  oublis possibles ; ils ne servent qu'à la relecture.
- Le rapport sortie / entrée du Practice Extractor sur un vrai entretien reste
  inconnu : si un bloc est encore tronqué, réduire `TRACE_PRACTICE_CHUNK_TOKENS`
  (jamais compenser en augmentant `TRACE_LLM_MAX_TOKENS`).
- L'estimation des tokens (3,5 caractères par token) est approximative.
