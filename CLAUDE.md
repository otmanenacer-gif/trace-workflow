# TRACE — consignes pour Claude Code

- Exécuter TRACE sur un entretien, ou les tâches TRACE en attente d'un run : suivre **`TRACE_WORKFLOW.md`**
  (commandes `python scripts/trace_workflow.py …`, un sous-agent `trace-agent` par tâche).
- Les **étapes 3, 4 et 5** passent par ce workflow (`run <run> --until 5`, une étape déjà complète n'est jamais
  rejouée). L'étape 6 n'est pas encore migrée : s'arrêter après l'étape 5 et le dire.
- Aucun appel à une API de modèle (Anthropic ou autre) ; ne jamais renseigner `TRACE_STAGE3_BACKEND`,
  `TRACE_STAGE4_BACKEND` ni `TRACE_STAGE5_BACKEND` à `anthropic` ; ne jamais utiliser `--force` sans demande
  explicite.
- Les prompts `prompts/*.md`, les schémas (`agents/`) et les validateurs (`core/`) font foi : ne jamais les modifier
  pour faire passer une réponse.
- Tests : `python -m pytest` (aucun appel réseau, LLM simulé).
