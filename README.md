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

## Pieces

- **`maestro/chess_analysis.py`** - analyze one game move-by-move via a
  UCI engine (Stockfish): for every position, ask for the top `multipv`
  candidate lines (not just the single best move - "was there only one
  good option here" matters for telling a blunder from an understandable
  choice), and score the move actually played against them, even when
  it isn't one of the top candidates.
- **`maestro/analysis_pool.py`** - run that over many games at once, one
  Stockfish process per game task in a process pool, not one process per
  worker kept alive across tasks (see that module's docstring - the
  latter reliably deadlocked on pool shutdown; verified empirically, not
  just avoided out of caution).
- **LLM-generated, personalized feedback on top of the raw engine
  numbers** - not built yet.

## Setup

```bash
uv sync --extra test
cp .env.example .env   # fill in OPENROUTER_API_KEY / WANDB_API_KEY as needed
uv run pytest
```

Stockfish itself isn't a Python package - install it separately (e.g.
`apt install stockfish`, or a static build from
[stockfishchess.org](https://stockfishchess.org/download/)) and pass its
path to `chess_analysis`/`analysis_pool`. Tests that need a real engine
are skipped automatically if no `stockfish` binary is found.

## License

MIT
