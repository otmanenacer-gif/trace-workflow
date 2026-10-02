# TRACE — consignes pour Claude Code

- Exécuter TRACE sur un entretien, ou les tâches TRACE en attente d'un run : suivre **`TRACE_WORKFLOW.md`**
  (commandes `python scripts/trace_workflow.py …`, un sous-agent `trace-agent` par tâche).
- Les **étapes 3 et 4** passent par ce workflow (`run <run> --until 4`). Les étapes 5 et 6 ne sont pas encore
  migrées : s'arrêter après l'étape 4 et le dire.
- Aucun appel à une API de modèle (Anthropic ou autre) ; ne jamais renseigner `TRACE_STAGE3_BACKEND=anthropic`
  ni `TRACE_STAGE4_BACKEND=anthropic`.
- Les prompts `prompts/*.md`, les schémas (`agents/`) et les validateurs (`core/`) font foi : ne jamais les modifier
  pour faire passer une réponse.
- Tests : `python -m pytest` (aucun appel réseau, LLM simulé).
