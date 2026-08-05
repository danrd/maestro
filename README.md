# maestro

Personal chess trainer: upload your own game history, get it evaluated
by an engine, then get a specific game analyzed against that history -
recurring strategic mistakes tied to your own play patterns, with
advice, not generic engine commentary.

Status: early, active development. Both the single-game pipeline (import
-> engine analysis -> report -> LLM coaching feedback) and cross-game
pattern finding (phase-of-game habits, recurring opening lines, both
compared against a reference opening idea where one is known) are
complete end to end - see `maestro/pipeline.py`. What's actually missing
now is running it against a real player's real game history instead of
test fixtures - see "Not built yet" below.

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
- **`maestro/lichess_import.py`** - pull a Lichess user's games straight
  from the Lichess API instead of a manually-exported PGN file, with
  clocks/opening/evals included (opening comes from Lichess's own
  ECO/Opening tags - no need for our own opening-book classifier; evals
  are Lichess's stored Stockfish analysis when a game has one, a free
  skip on recomputing that one game).
- **`maestro/player_profile.py`** - aggregates mistakes across many
  `GameReport`s into per-phase, per-color statistics: the opening (first
  5 moves per side, matching `chess_analysis.py`'s cutoff), then 5-move
  chunks, with everything from move 40 onward folded into one final
  bucket rather than thinning out into near-empty buckets for rare very
  long games. A bucket is only reported once at least `min_games` games
  actually reached it - otherwise it's noise from one or two games, not
  a real pattern.
- **`maestro/opening_signature.py`** - groups games by their own move
  sequence (both colors - a line is co-defined by the reply, not just
  one side's moves) instead of a fixed classification, so grouping works
  even for lines the reference data below doesn't name. Depth is
  adaptive: tries 5 full moves per side first, shrinking to 4 then 3
  only if not enough *other* games share the longer prefix; below that,
  a game gets no signature rather than being forced into a
  low-confidence group.
- **`maestro/opening_reference.py`** - classifies a signature two ways:
  a *name* (vendored from
  [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings),
  CC0/public domain, ~3800 entries - broad but names only) and, for a
  bounded set of well-known top-level systems, a hand-curated *idea*
  (no open dataset actually explains the strategic idea in prose, only
  classifies - not sourced from anywhere specific, written directly as
  established chess theory). Everything outside that bounded set gets a
  signature and statistics but no idea text - filled in by hand later as
  cases accumulate, not auto-generated.
- **`maestro/opening_profile.py`** - combines the two above with each
  group's own mistake statistics into one `OpeningGroupSummary` per
  recurring line.
- **`maestro/profile_coaching.py`** - the cross-game counterpart to
  `coaching.py`: turns a `PlayerProfile` + `OpeningGroupSummary` list
  into prose, distinguishing a general phase-of-game habit from an
  opening-specific pattern, and referencing the opening's idea against
  the player's actual mistakes there when one is known.
- **`maestro/pipeline.py`** - ties everything above into three entry
  points: `analyze_and_cache_games(...)` (import what's new, reuse a
  cached report wherever one already exists, run Stockfish only on
  what's actually missing), `analyze_and_coach_games(...)` (the same,
  then per-game coaching feedback, cached independently of the report),
  and `analyze_and_coach_profile(...)` (the same, then aggregate across
  every game given into a `PlayerProfile` + opening groups + one
  cross-game feedback text, cached by the exact set of games + settings
  involved).
- **`maestro/app.py`** - a minimal [Gradio](https://gradio.app) front end:
  paste a username and PGN, get per-game analysis plus (once enough games
  are pasted) the cross-game pattern summary. No LLM key is required to
  use it - without `OPENROUTER_API_KEY` set it still runs the full
  engine analysis and shows the structured statistics, just without the
  prose commentary; the key only unlocks the LLM coaching text on top.
  Run with `uv run --extra ui python -m maestro.app`.

## Not built yet

- **Running against a real player's real history.** Every piece above
  is built and tested (including two full pipelines, and the Gradio app
  itself, run end to end against real Stockfish - see
  `tests/test_pipeline.py`/`tests/test_app.py`), but only ever against a
  handful of test games. Nobody has pointed it at an actual Lichess
  account's full game history yet.
- **Opening classification for manually-imported PGNs.** A game's
  `opening` field (`game_report.py`) only gets filled in when the source
  PGN already has `ECO`/`Opening` tags (true for Lichess API imports,
  not guaranteed for hand-exported files) - separate from
  `opening_signature.py`'s own grouping, which doesn't need tags at all.
- **Growing the curated opening-idea set.** Deliberately bounded to
  well-known systems for now (see `opening_reference.py`); everything
  else just accumulates statistics under its own signature until
  someone adds a description by hand.

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
