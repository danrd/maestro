"""Tests for maestro/analysis_pool.py.

load_pgn_games is pure PGN-text splitting, tested directly. The actual
process-pool + real-engine path (analyze_pgn_games_parallel) is an
opt-in integration test - skipped unless a real Stockfish binary is
found, since that's the one thing here no fake can stand in for: what's
under test in that one is specifically that a persistent-across-tasks
engine problem doesn't come back (see the module docstring in
analysis_pool.py - this exact scenario deadlocked ProcessPoolExecutor's
shutdown before switching to one engine per task).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import chess.engine
import pytest

from maestro.analysis_pool import analyze_pgn_games_parallel, load_pgn_games

SCHOLARS_MATE_PGN = """[Event "Test"]
[White "A"]
[Black "B"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""


def _find_stockfish() -> Optional[str]:
    for candidate in (shutil.which("stockfish"), "/usr/games/stockfish",
                      "/usr/local/bin/stockfish", "/usr/bin/stockfish"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


STOCKFISH_PATH = _find_stockfish()
requires_stockfish = pytest.mark.skipif(STOCKFISH_PATH is None, reason="no stockfish binary found")


# -- load_pgn_games -----------------------------------------------------

def test_load_pgn_games_splits_a_multi_game_file(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(SCHOLARS_MATE_PGN + "\n" + SCHOLARS_MATE_PGN, encoding="utf-8")

    games = load_pgn_games(str(path))

    assert len(games) == 2
    for text in games:
        assert "Qxf7#" in text


def test_load_pgn_games_returns_empty_list_for_an_empty_file(tmp_path):
    path = tmp_path / "empty.pgn"
    path.write_text("", encoding="utf-8")

    assert load_pgn_games(str(path)) == []


# -- analyze_pgn_games_parallel (real Stockfish, opt-in) ---------------

@requires_stockfish
def test_analyze_pgn_games_parallel_runs_real_games_and_returns_them_in_order():
    results = analyze_pgn_games_parallel(
        [SCHOLARS_MATE_PGN, SCHOLARS_MATE_PGN], STOCKFISH_PATH,
        chess.engine.Limit(depth=5), multipv=2, num_workers=2,
    )

    assert len(results) == 2
    for result in results:
        assert len(result.moves) == 7
        # 3...Nf6 (index 5) walks into forced mate - a real, large loss,
        # not just "some positive number" - this is what a genuine
        # engine evaluation of this exact trap should produce.
        assert result.moves[5].played_move == "Nf6"
        assert result.moves[5].loss_cp > 50_000
        # the final move (mate) should show as the engine's own best move
        assert result.moves[-1].loss_cp == 0
