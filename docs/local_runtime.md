# Exécution locale de TRACE (Ollama)

TRACE exécute lui-même les étapes 1 à 6 sur l'ordinateur de l'utilisateur. Les agents sémantiques sont joués par
un **modèle local servi par Ollama** : aucune clé, aucune API externe, aucune donnée envoyée hors de l'ordinateur,
aucun outil extérieur nécessaire à l'exécution.

```
Streamlit (app.py) ou CLI (scripts/trace_local.py)
→ TRACE : ingestion déterministe (étapes 1-2)
→ TRACE : préparation déterministe de l'étape (blocs, candidats, ancrages, corpus…)       inchangée
→ LocalAgentRunner (core/local_agent_runner.py)
     → Ollama local, POST /api/chat (en flux), prompt de l'agent (prompts/*.md), JSON Schema strict de l'agent,
       fenêtre de contexte dimensionnée pour CET appel, réponse bornée
     → JSON → Pydantic → validateur méthodologique de l'étape (core/agent_checks.py)
     → en cas d'anomalie : nouvelles tentatives LOCALES, erreurs à l'appui (budgets ci-dessous)
→ TRACE : validateurs habituels, fusion, sorties habituelles                                inchangés
→ étape suivante
```

## Installation (une fois, avec Internet)

1. Installer Ollama : <https://ollama.com/download>.
2. Installer le modèle recommandé : `ollama pull qwen2.5:7b`.
3. `pip install -r requirements.txt` (Streamlit, pypdf, python-docx, Pydantic, pytest ; aucun SDK de modèle).

Ensuite, Internet n'est plus nécessaire.

## Utilisation

```bash
ollama serve                 # si Ollama n'est pas déjà lancé (l'application Ollama le fait aussi)
streamlit run app.py         # importer un entretien, puis « Exécuter l'étape 3 / 4 / 5 (local, Ollama) »
```

Le bouton lance l'étape **en arrière-plan** (processus indépendant de la page, voir « Exécution en arrière-plan et
reprise ») : la page affiche l'agent en cours, le temps écoulé, les tokens générés, la liste des appels
(✓ validé, ⏳ en cours, ○ à faire) et, pour chaque appel terminé, ses mesures. La page peut être rafraîchie ou fermée.

Étape 6 : section « Étape 6 », importer les sorties de l'étape 5 de plusieurs entretiens (ou inclure celles du
run), puis « Exécuter l'étape 6 (local, Ollama) ». L'interface affiche en permanence « Exécution : locale ·
Runtime : Ollama · Modèle : … · Données externes : aucune » et l'état d'Ollama (actif, modèle installé ou commande
d'installation).

Sans interface :

```bash
python scripts/trace_local.py runtime                     # Ollama actif ? modèle installé ?
python scripts/trace_local.py ingest entretien.docx       # run + ingestion
python scripts/trace_local.py run latest --until 5        # étapes 3 → 5
python scripts/trace_local.py stage6 <run A> <run B>      # étape 6 sur les sorties de l'étape 5
python scripts/trace_local.py status latest
python scripts/trace_local.py job start latest --until 5  # étapes 3 → 5 en arrière-plan (terminal fermable)
python scripts/trace_local.py job status latest           # agent en cours, appels validés
```

Chaque appel au modèle écrit une ligne de mesure (terminal de la CLI, `job.log`, journal de l'appel), par exemple :
`Practice Extractor — ENTRETIEN/bloc2 — 9 900 tokens entrée — 620 sortie — 41 s — 15 tok/s — contexte 20480 — 1 tentative(s) — accepted`.

## Réglages (facultatifs, `.env` ou environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `TRACE_LOCAL_MODEL` | `qwen2.5:7b` | modèle Ollama utilisé par tous les agents |
| `TRACE_OLLAMA_URL` | `http://localhost:11434` | adresse d'Ollama — **boucle locale uniquement** (localhost, 127.0.0.1, ::1) |
| `TRACE_OLLAMA_NUM_CTX` | `32768` | fenêtre de contexte **maximale** : chaque appel reçoit la plus petite fenêtre qui contient sa requête et sa réponse (voir « Performances ») ; une requête plus longue est refusée (`PROMPT_TOO_LONG`), jamais tronquée |
| `TRACE_OLLAMA_TIMEOUT` | `1800` | délai maximal d'une génération, en secondes |
| `TRACE_OLLAMA_KEEP_ALIVE` | `30m` | durée pendant laquelle Ollama garde le modèle chargé entre deux appels |
| `TRACE_LOCAL_MAX_CORRECTIONS` | `2` | nouvelles tentatives locales après un JSON invalide, hors schéma ou tronqué |
| `TRACE_LOCAL_MAX_METHOD_CORRECTIONS` | `1` | nouvelles tentatives locales après une anomalie **bloquante** du validateur méthodologique (réponse conforme au schéma) ; `2` reproduit l'ancien comportement |
| `TRACE_LOCAL_TEMPERATURE` | `0` | température (reproductibilité) |
| `TRACE_LOCAL_CONCURRENCY` | `1` | appels simultanés au modèle local |
| `TRACE_INTERACTION_CHUNK_TOKENS`, `TRACE_PRACTICE_CHUNK_TOKENS` | `5000`, `4000` | taille des blocs des entretiens longs |

Une adresse hors de la boucle locale est refusée (`NON_LOCAL_URL`) ; les proxys HTTP de l'environnement ne sont
jamais utilisés pour joindre Ollama.

## Réponses non conformes : corrections locales, validateurs jamais contournés

1. La réponse doit être un JSON conforme au schéma de l'agent (Ollama reçoit ce schéma : sortie structurée).
2. Elle est validée par Pydantic, puis par le validateur de l'étape (citations mot pour mot, identifiants,
   épisodes, ancrages, appuis du corpus…).
3. JSON invalide ou schéma non respecté : le modèle reçoit sa réponse et la liste des erreurs, et renvoie l'objet
   complet corrigé (au plus `TRACE_LOCAL_MAX_CORRECTIONS` fois). Anomalie **bloquante** du validateur sur une
   réponse conforme au schéma : même principe, au plus `TRACE_LOCAL_MAX_METHOD_CORRECTIONS` fois. Une réponse
   **tronquée** (réponse maximale atteinte) est redemandée en entier avec une réserve doublée, sans être renvoyée
   au modèle. Chaque correction régénère toute la réponse : c'est l'opération la plus coûteuse (voir les mesures).
4. Après les corrections : une réponse conforme au schéma est rendue telle quelle à l'étape, dont les validateurs
   habituels signalent ou rejettent ce qui reste (needs_review, rejet) ; une réponse toujours non conforme au schéma
   est un échec de l'agent (statut FAILED ou PARTIAL), jamais une réponse inventée.

## Erreurs du runtime : claires, sans repli

| Situation | Message | Que faire |
|---|---|---|
| Ollama arrêté | « Ollama ne répond pas … » | `ollama serve` (ou lancer l'application Ollama) |
| Modèle absent | « Le modèle local demandé n'est pas installé … » | `ollama pull <modèle>` |
| Adresse non locale | « Adresse d'Ollama refusée … » | utiliser `http://localhost:11434` |
| Réponse tronquée | « Réponse du modèle local tronquée … » | augmenter `TRACE_OLLAMA_NUM_CTX` |
| Délai dépassé | « Le modèle local n'a pas répondu … » | augmenter `TRACE_OLLAMA_TIMEOUT` ou prendre un modèle plus petit |

Le runtime est vérifié **avant** toute exécution d'une étape : s'il n'est pas prêt, rien n'est écrit. Aucun autre
fournisseur n'est jamais utilisé.

## Performances : contexte, génération, ordre des appels

- **Fenêtre de contexte par appel.** Le cache K/V qu'Ollama alloue est proportionnel à `num_ctx` (qwen2.5:7b :
  ≈ 56 Ko par token, soit ≈ 1,75 Go à 32 768 contre ≈ 1,1 Go à 20 480, en plus des ≈ 4,7 Go du modèle). Une
  fenêtre de 32 768 pour TOUS les appels peut faire déborder une carte graphique de 8 Go : Ollama calcule alors une
  partie du modèle sur le processeur (`ollama ps` affiche par exemple « 8%/92% CPU/GPU »), ce qui divise la vitesse
  de génération. TRACE choisit donc, pour chaque appel, la plus petite fenêtre (4 096, 8 192, 12 288, 16 384,
  20 480, 24 576, 32 768…) qui contient la requête (estimée prudemment à 3 caractères par token, corrigée si Ollama
  en compte davantage), la réponse réservée et une marge ; la fenêtre ne diminue jamais au cours d'une exécution
  (changer `num_ctx` recharge le modèle). Rien n'est jamais tronqué : `TRACE_OLLAMA_NUM_CTX` est un maximum.
- **Réponse bornée** (`num_predict`) : 2 048 tokens (audit des locuteurs) à 8 192 (étapes 5 et 6) ; une génération
  qui s'emballe s'arrête, et une réponse réellement plus longue est redemandée avec une réserve doublée.
- **Modèle gardé chargé** (`keep_alive`, défaut 30 min) : pas de rechargement entre deux appels.
- **Ordre des appels** : les blocs d'un même agent sont envoyés à la suite ; Ollama réutilise le prompt système déjà
  lu (4 000 à 5 500 tokens) au lieu de le relire à chaque alternance Practice / Interaction. Les résultats sont
  identiques (appels indépendants, fusion déterministe).
- **Vérifier sur sa machine** : pendant une exécution, `ollama ps` doit indiquer `100% GPU`. Réglages facultatifs
  d'Ollama lui-même (variables d'environnement du serveur Ollama, pas de TRACE) : `OLLAMA_FLASH_ATTENTION=1` et
  `OLLAMA_KV_CACHE_TYPE=q8_0` réduisent encore la mémoire du contexte.

### Benchmark

```bash
python scripts/trace_benchmark.py plan                 # sans Ollama : appels prévus, taille et fenêtre de chaque appel
python scripts/trace_benchmark.py run --legacy         # mesure « avant » (fenêtre fixe, génération non bornée)
python scripts/trace_benchmark.py run --compare-format # mesure « après » + coût du JSON Schema strict (diagnostic)
```

Entretien synthétique représentatif (≈ 1 h, 349 tours, ≈ 56 000 caractères) : `benchmarks/entretien_synthetique_1h.txt`.
Le benchmark travaille dans un dossier temporaire (aucun résultat en cache réutilisé), n'utilise que le modèle
configuré (aucun téléchargement) et écrit un tableau Agent | tokens entrée | tokens sortie | durée | tentatives |
contexte, les totaux, l'estimation pour 17 entretiens et un rapport JSON dans `benchmarks/results/`.

## Exécution en arrière-plan et reprise

- Streamlit (bouton d'une étape) et `trace_local.py job start` lancent un **processus détaché**
  (`python -m core.local_jobs <dossier>`) : rafraîchir la page, fermer le navigateur, redémarrer Streamlit ou fermer
  le terminal n'interrompt pas l'exécution.
- Le processus tient à jour `local_runs/job/job.json` (run) ou `cross_interview/local_jobs/stage6/job.json`
  (étape 6) : agent en cours, tentative, tokens générés, liste des appels attendus, mesures de chaque appel, et un
  battement toutes les 2 secondes. La page le relit toutes les 2 secondes.
- Après un refresh, l'adresse de la page (`?run=…`) rouvre le run ; à la réouverture du navigateur, TRACE rouvre de
  lui-même le run dont une exécution est en cours.
- Chaque réponse validée est enregistrée **immédiatement** dans le cache TRACE, avant l'appel suivant. Si
  l'exécution s'arrête (ordinateur éteint, Ollama arrêté, bouton « Arrêter l'exécution »), la page affiche
  « interrompue » et un bouton « Reprendre l'exécution » : l'étape repart au premier appel non terminé ; un résultat
  validé n'est jamais recalculé (il apparaît « ✓ déjà validé (repris, non recalculé) »).

## Garde, cache et reprise

- Une étape complète et à jour n'est jamais rejouée (étape 4 calculée sur l'étape 3 actuelle, étape 5 sur les
  étapes 3 et 4 actuelles ; étape 6 : même corpus, mêmes prompt, schéma et versions). `--force` (CLI) la rejoue.
- Le cache TRACE (`data/cache/analysis/`) reprend les réponses validées d'une requête identique avec le même modèle :
  après un échec partiel (un bloc), seuls les appels manquants sont refaits.

## Fichiers produits

```
data/outputs/<run>/interviews/<id>/analysis/        sorties habituelles des étapes 3, 4 et 5
data/outputs/<run>/local_runs/stage<N>/status.json  bilan de la dernière exécution de l'étape N
data/outputs/<run>/local_runs/stage<N>/<appel>.json journal d'un appel : modèle, tentatives (mesures de chacune :
                                                    tokens lus et générés, durées, vitesse, fenêtre, réponse maximale,
                                                    motif de correction), erreurs, réponse finale
data/outputs/<run>/local_runs/job/job.json          exécution en arrière-plan : état, agent en cours, liste des appels
data/outputs/<run>/local_runs/job/job.log           une ligne de mesure par tentative et par appel
data/outputs/cross_interview/<corpus_id>/analysis/  sorties habituelles de l'étape 6
data/outputs/cross_interview/<corpus_id>/local_runs/stage6/
```

Les manifests portent `"model": "<modèle Ollama>"`, `api_calls: 0`, `billed_this_run: false`, `usage` (tokens
traités par le modèle local) et, comme `request_id`, l'identifiant de l'appel (son journal).
