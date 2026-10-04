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
- Étape 6 avec un seul entretien exploitable : `NOT_APPLICABLE_SINGLE_INTERVIEW`, aucune comparaison, aucun appel.
- Étape 7 : rapport final DÉTERMINISTE (`core/final_report.py`, Markdown + JSON dans `<run>/final_report/`), assemblé à
  partir des sorties validées des étapes 3 à 5, sans aucun appel au modèle ; Streamlit « Générer le rapport final »,
  CLI `trace_local.py report RUN`. Pas d'étape 8 ou au-delà sans demande explicite.
