"""Tests for maestro/pipeline.py.

The orchestration logic (import, cache hit/miss, dedup, order
preservation) is tested against a monkeypatched analyze_pgn_games_parallel
- fast and deterministic, no engine needed, since what's under test is
whether the pipeline skips cached games and calls the analyzer only for
what's actually new. A separate opt-in test at the bottom runs the same
function against real Stockfish, skipped automatically if none is found.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import chess.engine
import pytest

import maestro.pipeline as pipeline_module
from maestro.chess_analysis import GameAnalysis, MoveAnalysis
from maestro.game_store import compute_game_hash, get_cached_report, open_store

GAME_A = """[Event "Test"]
[White "Alice"]
[Black "Bob"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

GAME_B = """[Event "Test"]
[White "Carol"]
[Black "Dave"]

1. d4 d5 1-0
"""


class _FakeAnalyzer:
    """Stands in for analyze_pgn_games_parallel: records what it was
    asked to analyze and returns one trivial GameAnalysis per input,
    in order."""
    def __init__(self):
        self.calls = []

    def __call__(self, pgn_texts, stockfish_path, limit, multipv, num_workers,
                  mistake_threshold_cp, safe_alternatives_cap, opening_ply_cutoff):
        self.calls.append(list(pgn_texts))
        return [
            GameAnalysis(game_id=compute_game_hash(text), moves=[
                MoveAnalysis(move_number=1, color=chess.WHITE, played_move="e4",
                              played_score_cp=40, best_move="e4", best_score_cp=40),
            ])
            for text in pgn_texts
        ]


def test_analyzes_new_games_and_caches_the_result(monkeypatch):
    fake_analyzer = _FakeAnalyzer()
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", fake_analyzer)
    conn = open_store(":memory:")

    reports = pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path",
                                                        chess.engine.Limit(depth=5))

    assert len(reports) == 1
    assert len(fake_analyzer.calls) == 1
    assert fake_analyzer.calls[0] == [GAME_A]
    # the report actually landed in the cache
    game_hash = compute_game_hash(GAME_A)
    from maestro.game_store import compute_params_hash
    params_hash = compute_params_hash(multipv=3, mistake_threshold_cp=50, safe_alternatives_cap=5,
                                       opening_ply_cutoff=10, depth=5, time=None, nodes=None)
    assert get_cached_report(conn, game_hash, params_hash) is not None


def test_second_call_with_the_same_settings_is_a_cache_hit_not_a_reanalysis(monkeypatch):
    fake_analyzer = _FakeAnalyzer()
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", fake_analyzer)
    conn = open_store(":memory:")
    limit = chess.engine.Limit(depth=5)

    pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path", limit)
    pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path", limit)

    assert len(fake_analyzer.calls) == 1  # second call was a pure cache hit


def test_different_settings_bypass_the_cache(monkeypatch):
    fake_analyzer = _FakeAnalyzer()
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", fake_analyzer)
    conn = open_store(":memory:")

    pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path",
                                             chess.engine.Limit(depth=5))
    pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path",
                                             chess.engine.Limit(depth=10))  # different depth

    assert len(fake_analyzer.calls) == 2


def test_only_analyzes_the_games_not_already_cached(monkeypatch):
    fake_analyzer = _FakeAnalyzer()
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", fake_analyzer)
    conn = open_store(":memory:")
    limit = chess.engine.Limit(depth=5)

    pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path", limit)
    reports = pipeline_module.analyze_and_cache_games(conn, [GAME_A, GAME_B], "fake-stockfish-path", limit)

    assert len(reports) == 2
    assert len(fake_analyzer.calls) == 2
    assert fake_analyzer.calls[1] == [GAME_B]  # only the new one was (re)analyzed


def test_results_preserve_input_order_regardless_of_cache_mix(monkeypatch):
    fake_analyzer = _FakeAnalyzer()
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", fake_analyzer)
    conn = open_store(":memory:")
    limit = chess.engine.Limit(depth=5)

    pipeline_module.analyze_and_cache_games(conn, [GAME_B], "fake-stockfish-path", limit)  # cache B first
    reports = pipeline_module.analyze_and_cache_games(conn, [GAME_A, GAME_B], "fake-stockfish-path", limit)

    assert [r.game_id for r in reports] == [compute_game_hash(GAME_A), compute_game_hash(GAME_B)]


# -- real Stockfish, opt-in ------------------------------------------------

def _find_stockfish() -> Optional[str]:
    for candidate in (shutil.which("stockfish"), "/usr/games/stockfish",
                      "/usr/local/bin/stockfish", "/usr/bin/stockfish"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


STOCKFISH_PATH = _find_stockfish()


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="no stockfish binary found")
def test_analyze_and_cache_games_end_to_end_with_real_stockfish():
    conn = open_store(":memory:")

    reports = pipeline_module.analyze_and_cache_games(
        conn, [GAME_A], STOCKFISH_PATH, chess.engine.Limit(depth=6),
        mistake_threshold_cp=50, opening_ply_cutoff=0, player_name="Bob",
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.player_color == chess.BLACK
    # 3...Nf6 walks into forced mate - a real mistake, worst of the game
    assert report.mistakes[0].played_move == "Nf6"
    assert report.mistakes[0].safe_alternatives is not None
