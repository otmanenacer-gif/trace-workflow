# TRACE — consignes pour Claude Code

- Exécuter TRACE sur un entretien, ou les tâches TRACE en attente d'un run : suivre **`TRACE_WORKFLOW.md`**
  (commandes `python scripts/trace_workflow.py …`, un sous-agent `trace-agent` par tâche).
- Les **étapes 3 à 6** passent par ce workflow : étapes 3 à 5 par run (`run <run> --until 5`), étape 6 par
  corpus de sorties de l'étape 5 (`stage6 <runs, dossiers ou fichiers>`, puis `stage6 --corpus <corpus_id>`). Une
  étape (ou un corpus) déjà complète n'est jamais rejouée. Pas d'étape 7 : s'arrêter après l'étape demandée.
- TRACE est un workflow multi-agents exécuté dans Claude Code : aucun appel à une API de modèle (Anthropic ou
  autre), aucun SDK, aucune clé ; n'en ajouter aucun. Ne jamais utiliser `--force` sans demande explicite.
- Les prompts `prompts/*.md`, les schémas (`agents/`) et les validateurs (`core/`) font foi : ne jamais les modifier
  pour faire passer une réponse.
- Tests : `python -m pytest` (aucun appel réseau, agents simulés).
