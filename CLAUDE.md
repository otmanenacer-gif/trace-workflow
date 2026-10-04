# TRACE — notes pour le développement du dépôt

Ce fichier ne sert qu'au DÉVELOPPEMENT. Claude Code n'est pas nécessaire pour utiliser TRACE : les étapes 1 à 6
s'exécutent localement (Streamlit ou `python scripts/trace_local.py`), les agents étant joués par un modèle local
servi par Ollama (voir `docs/local_runtime.md`).

- Architecture : `app.py` → `core/local_pipeline.py` (gardes, étapes 3 à 6) → orchestrateurs habituels
  (`core/analysis.py`, `accountability.py`, `trajectory.py`, `cross_interview.py`) → `core/local_agent_runner.py`
  (Ollama local, JSON Schema, Pydantic, `core/agent_checks.py`, corrections locales) → validateurs habituels.
- Aucun appel à une API de modèle externe (Anthropic, OpenAI, Gemini…), aucun SDK, aucune clé, aucun repli ;
  Ollama uniquement sur la boucle locale. N'en ajouter aucun.
- Les prompts `prompts/*.md`, les schémas (`agents/`) et les validateurs (`core/`) font foi : ne jamais les modifier
  pour faire passer une réponse ; aucune règle méthodologique ne se modifie sans demande explicite.
- Exécution depuis Streamlit : en arrière-plan (`core/local_jobs.py`, processus détaché, `job.json`) ; reprise par le
  cache TRACE (chaque réponse validée est enregistrée immédiatement). Mesures : `scripts/trace_benchmark.py`.
- Étape 3 : une réponse lisible n'est jamais régénérée en entier ; citations corrigées de façon déterministe
  (`core/citation_resolver.py`), au plus UNE réparation par le modèle et par objet pour les autres anomalies locales
  (`core/stage3_repair.py`, état persistant `local_runs/stage3/partial/`).
- Sélectivité de l'étape 3 : `core/practice_selectivity.py` (lien explicite avec une IAG, `non_use_reason`) et
  `core/signal_selectivity.py` (contradictions démontrées) ; `trace_local.py refilter` les réapplique sans modèle.
- Tests : `python -m pytest` (aucun modèle réel, aucun réseau : faux Ollama derrière le vrai runner local).
- Test navigateur (à la main) : `python tests/e2e/browser_check.py` (Playwright + Chromium).
- Lot sans surveillance : `trace_local.py batch <corpus> --until 5` (`core/batch.py`) — run retrouvé à chaque relance,
  étapes 3 → 5 par entretien, garde et cache habituels, une nouvelle tentative par étape, échec isolé, manifeste
  `<run>/batch/batch_manifest.json`.
- Étape 6 d'un run : `trace_local.py stage6 <run_id>` (`core/stage6_blocks.py`) — entretiens `stage5_valid` du lot,
  blocs d'au plus 6 entretiens (étape 6 habituelle par bloc, cache et reprise), fusion déterministe dans
  `<run>/stage6/stage6_corpus.json`.
- Étape 6 avec un seul entretien exploitable : `NOT_APPLICABLE_SINGLE_INTERVIEW`, aucune comparaison, aucun appel.
- Étape 7 : théorisation transversale (`core/stage7_theory.py`, agents `agents/theory_builder.py`) à partir de
  `<run>/stage6/stage6_corpus.json` — blocs thématiques puis une synthèse, preuves reconstituées par TRACE, cache et
  reprise ; `trace_local.py stage7 RUN` → `<run>/stage7/stage7_theory.json`.
- Rapport individuel expérimental (`core/final_report.py`, déterministe, Markdown + JSON dans `<run>/final_report/`) :
  `trace_local.py report RUN`, Streamlit « Générer le rapport final ». Pas d'étape 8 ou au-delà sans demande explicite.
