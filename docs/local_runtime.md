# Exécution locale de TRACE (Ollama)

TRACE exécute lui-même les étapes 1 à 6 sur l'ordinateur de l'utilisateur. Les agents sémantiques sont joués par
un **modèle local servi par Ollama** : aucune clé, aucune API externe, aucune donnée envoyée hors de l'ordinateur,
aucun outil extérieur nécessaire à l'exécution.

```
Streamlit (app.py) ou CLI (scripts/trace_local.py)
→ TRACE : ingestion déterministe (étapes 1-2)
→ TRACE : préparation déterministe de l'étape (blocs, candidats, ancrages, corpus…)       inchangée
→ LocalAgentRunner (core/local_agent_runner.py)
     → Ollama local, POST /api/chat, prompt de l'agent (prompts/*.md), JSON Schema strict de l'agent
     → JSON → Pydantic → validateur méthodologique de l'étape (core/agent_checks.py)
     → en cas d'anomalie : au plus TRACE_LOCAL_MAX_CORRECTIONS nouvelles tentatives LOCALES, erreurs à l'appui
→ TRACE : validateurs habituels, fusion, sorties habituelles                                inchangés
→ étape suivante
```

## Installation (une fois, avec Internet)

1. Installer Ollama : <https://ollama.com/download>.
2. Installer le modèle recommandé : `ollama pull qwen2.5:14b`.
3. `pip install -r requirements.txt` (Streamlit, pypdf, python-docx, Pydantic, pytest ; aucun SDK de modèle).

Ensuite, Internet n'est plus nécessaire.

## Utilisation

```bash
ollama serve                 # si Ollama n'est pas déjà lancé (l'application Ollama le fait aussi)
streamlit run app.py         # importer un entretien, puis « Exécuter l'étape 3 / 4 / 5 (local, Ollama) »
```

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
```

## Réglages (facultatifs, `.env` ou environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `TRACE_LOCAL_MODEL` | `qwen2.5:14b` | modèle Ollama utilisé par tous les agents |
| `TRACE_OLLAMA_URL` | `http://localhost:11434` | adresse d'Ollama — **boucle locale uniquement** (localhost, 127.0.0.1, ::1) |
| `TRACE_OLLAMA_NUM_CTX` | `32768` | fenêtre de contexte demandée (les requêtes des étapes 5 et 6 peuvent atteindre ~24 000 tokens) |
| `TRACE_OLLAMA_TIMEOUT` | `1800` | délai maximal d'une génération, en secondes |
| `TRACE_LOCAL_MAX_CORRECTIONS` | `2` | nouvelles tentatives locales après une réponse non conforme |
| `TRACE_LOCAL_TEMPERATURE` | `0` | température (reproductibilité) |
| `TRACE_LOCAL_CONCURRENCY` | `1` | appels simultanés au modèle local |
| `TRACE_INTERACTION_CHUNK_TOKENS`, `TRACE_PRACTICE_CHUNK_TOKENS` | `5000`, `4000` | taille des blocs des entretiens longs |

Une adresse hors de la boucle locale est refusée (`NON_LOCAL_URL`) ; les proxys HTTP de l'environnement ne sont
jamais utilisés pour joindre Ollama.

## Réponses non conformes : corrections locales, validateurs jamais contournés

1. La réponse doit être un JSON conforme au schéma de l'agent (Ollama reçoit ce schéma : sortie structurée).
2. Elle est validée par Pydantic, puis par le validateur de l'étape (citations mot pour mot, identifiants,
   épisodes, ancrages, appuis du corpus…).
3. JSON invalide, schéma non respecté ou anomalie **bloquante** : le modèle reçoit sa réponse et la liste des
   erreurs, et renvoie l'objet complet corrigé (au plus `TRACE_LOCAL_MAX_CORRECTIONS` fois).
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

## Garde, cache et reprise

- Une étape complète et à jour n'est jamais rejouée (étape 4 calculée sur l'étape 3 actuelle, étape 5 sur les
  étapes 3 et 4 actuelles ; étape 6 : même corpus, mêmes prompt, schéma et versions). `--force` (CLI) la rejoue.
- Le cache TRACE (`data/cache/analysis/`) reprend les réponses validées d'une requête identique avec le même modèle :
  après un échec partiel (un bloc), seuls les appels manquants sont refaits.

## Fichiers produits

```
data/outputs/<run>/interviews/<id>/analysis/        sorties habituelles des étapes 3, 4 et 5
data/outputs/<run>/local_runs/stage<N>/status.json  bilan de la dernière exécution de l'étape N
data/outputs/<run>/local_runs/stage<N>/<appel>.json journal d'un appel : modèle, tentatives, erreurs, réponse finale
data/outputs/cross_interview/<corpus_id>/analysis/  sorties habituelles de l'étape 6
data/outputs/cross_interview/<corpus_id>/local_runs/stage6/
```

Les manifests portent `"model": "<modèle Ollama>"`, `api_calls: 0`, `billed_this_run: false`, `usage` (tokens
traités par le modèle local) et, comme `request_id`, l'identifiant de l'appel (son journal).
