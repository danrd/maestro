# maestro

Personal chess trainer: upload your own game history, get it evaluated
by an engine, then get a specific game analyzed against that history -
recurring strategic mistakes tied to your own play patterns, with
advice, not generic engine commentary.

Status: early, active development. The single-game pipeline (import ->
engine analysis -> report -> LLM coaching feedback, each step cached) is
complete end to end - see `maestro/pipeline.py`. What's not built yet:
finding patterns *across* a player's whole game history rather than one
game at a time, which is the actual "tied to your own play patterns"
part of the pitch above - see "Not built yet" below.

Built on the same LLM pipeline infra as the `lector` project (prompt
composition, backend-agnostic inference, a resumable task loop) -
currently vendored in `llm_kit/`, planned to move into a shared
[`toolkit`](https://github.com/danrd/toolkit) repo once dependency
isolation between toolkit modules is settled.

## Pieces

- **`maestro/chess_analysis.py`** - analyze one game move-by-move via a
  UCI engine (Stockfish): for every position, ask for the top `multipv`
  candidate lines (not just the single best move - "was there only one
  good option here" matters for telling a blunder from an understandable
  choice), and score the move actually played against them, even when
  it isn't one of the top candidates.
  `chess_analysis.py` also flags, for mistakes past a configurable
  opening-move cutoff, how many *other* legal moves in that position
  were themselves safe (`safe_alternatives`, capped) - an unforced
  error among many good options is a different kind of mistake than
  missing the one move that mattered.
- **`maestro/analysis_pool.py`** - run that over many games at once, one
  Stockfish process per game task in a process pool, not one process per
  worker kept alive across tasks (see that module's docstring - the
  latter reliably deadlocked on pool shutdown; verified empirically, not
  just avoided out of caution).
- **`maestro/game_report.py`** - pure derivation (no engine calls) from
  an already-computed game analysis into a `GameReport`: flagged
  mistakes sorted worst-first, opening (from PGN `ECO`/`Opening` tags),
  which color the tracked player had, and per-move time spent (parsed
  from PGN `%clk` comments, when present in the export).
- **`maestro/game_store.py`** - SQLite-backed storage: games are
  deduplicated by a hash of their parsed moves so re-importing a PGN
  export only picks up what's actually new, and reports are cached by
  game hash + a hash of the analysis settings that produced them, so
  changing depth/multipv/threshold doesn't silently serve a stale
  result under different settings.
- **`maestro/coaching.py`** - turns a `GameReport` into readable coaching
  feedback via an LLM (built on `llm_kit`, same as `lector`): uses
  `safe_alternatives` directly - a mistake with several safe alternatives
  is framed as an unforced error worth calling out, one with few or none
  as a genuinely hard moment - rather than just restating the numbers.
- **`maestro/pipeline.py`** - ties everything above into two entry
  points: `analyze_and_cache_games(...)` (import what's new, reuse a
  cached report wherever one already exists for the exact settings
  given, run Stockfish only on what's actually missing) and
  `analyze_and_coach_games(...)` (the same, then generate - or reuse
  cached - coaching feedback per game; a report being cached doesn't
  imply its feedback is too, so that's checked independently).
- **`maestro/lichess_import.py`** - pull a Lichess user's games straight
  from the Lichess API instead of a manually-exported PGN file, with
  clocks/opening/evals included (opening comes from Lichess's own
  ECO/Opening tags - no need for our own opening-book classifier; evals
  are Lichess's stored Stockfish analysis when a game has one, a free
  skip on recomputing that one game).

## Not built yet

- **Cross-game pattern finding.** Every game is analyzed independently
  right now - nothing aggregates mistakes *across* a player's game
  history to surface recurring habits (same opening, same phase of the
  game, same kind of tactical miss). This is the part of the original
  idea ("advice tied to your own play patterns") that's still missing;
  needs its own design pass before building.
- **Opening classification for manually-imported PGNs.** `opening` in a
  `GameReport` only gets filled in when the source PGN already has
  `ECO`/`Opening` tags (true for Lichess API imports, not guaranteed for
  hand-exported files). No opening-book classifier of our own -
  deliberately deferred, not a small task.

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
