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
- Tests : `python -m pytest` (aucun modèle réel, aucun réseau : faux Ollama derrière le vrai runner local).
- Test navigateur (à la main) : `python tests/e2e/browser_check.py` (Playwright + Chromium).
- Pas d'étape 7 sans demande explicite.
