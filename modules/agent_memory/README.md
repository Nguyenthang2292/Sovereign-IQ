# agent_memory

MemOS-backed workflow and project-conventions context for agents: **Cursor, OpenCode, Claude Code, Antigravity**. Agents read a single context file; store is automatic (git hook) or via CLI.

## Setup

1. **MemOS API key**  
   Set `MEMOS_API_KEY` in the environment (or in `.env`). Optional: `MEMOS_USER_ID` (default `crypto_probability_repo`), `MEMOS_CONVERSATION_ID` (default today's date), `AGENT_MEMORY_CONTEXT_PATH` (default `.cursor/agent_memory_context.md`).

2. **Optional dependency**  
   Install MemOS client: `pip install MemoryOS`  
   If not installed, recall/store no-op (no crash).

## Commands

- **Recall (session start)**  
  Run when starting work so the agent sees up-to-date context:
  ```bash
  python -m modules.agent_memory recall
  ```
  Writes `.cursor/agent_memory_context.md` with "Project conventions" and "Recent workflow" from MemOS.

- **Store**
  - Store last git commit as workflow (e.g. from post-commit hook):
    ```bash
    python -m modules.agent_memory store
    ```
  - Store a custom summary:
    ```bash
    python -m modules.agent_memory store "Refactored auto_trade execution; added trailing stop step."
    ```

## Cursor / IDE

Add a rule or instruction: *"When starting a session, read `.cursor/agent_memory_context.md` for project conventions and recent workflow."*  
You can run `python -m modules.agent_memory recall` manually or from a VS Code/Cursor task when opening the project.

## Design

See `docs/plans/2026-02-06-agent-memos-workflow-design.md`.
