# maestro

Personal chess trainer: upload your own game history, get it evaluated
by an engine, then get a specific game analyzed against that history -
recurring strategic mistakes tied to your own play patterns, with
advice, not generic engine commentary.

Status: early, active development. Built on the same LLM pipeline infra
as the `lector` project (prompt composition, backend-agnostic inference,
a resumable task loop) - currently vendored in `llm_kit/`, planned to
move into a shared [`toolkit`](https://github.com/danrd/toolkit) repo
once dependency isolation between toolkit modules is settled.

## Setup

```bash
uv sync --extra test
cp .env.example .env   # fill in OPENROUTER_API_KEY / WANDB_API_KEY as needed
uv run pytest
```

## License

MIT
