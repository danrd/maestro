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
from maestro.coaching import CoachingConfig
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


# -- analyze_and_coach_games ------------------------------------------------

class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


class _FakeRunner:
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self._responses.pop(0)


def test_analyze_and_coach_games_generates_and_caches_feedback(monkeypatch):
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", _FakeAnalyzer())
    conn = open_store(":memory:")
    runner = _FakeRunner(responses=["Coaching text for game A."])

    results = pipeline_module.analyze_and_coach_games(
        conn, [GAME_A], "fake-stockfish-path", chess.engine.Limit(depth=5),
        _FakeTokenizer(), runner,
    )

    assert len(results) == 1
    assert results[0].feedback == "Coaching text for game A."
    assert len(runner.calls) == 1


def test_analyze_and_coach_games_second_call_is_a_feedback_cache_hit(monkeypatch):
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", _FakeAnalyzer())
    conn = open_store(":memory:")
    runner = _FakeRunner(responses=["Coaching text."])
    limit = chess.engine.Limit(depth=5)

    pipeline_module.analyze_and_coach_games(conn, [GAME_A], "fake-stockfish-path", limit,
                                             _FakeTokenizer(), runner)
    results = pipeline_module.analyze_and_coach_games(conn, [GAME_A], "fake-stockfish-path", limit,
                                                        _FakeTokenizer(), runner)

    assert len(runner.calls) == 1  # second call never touched the runner
    assert results[0].feedback == "Coaching text."


def test_analyze_and_coach_games_different_coaching_settings_bypass_the_feedback_cache(monkeypatch):
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", _FakeAnalyzer())
    conn = open_store(":memory:")
    runner = _FakeRunner(responses=["First.", "Second."])
    limit = chess.engine.Limit(depth=5)

    pipeline_module.analyze_and_coach_games(conn, [GAME_A], "fake-stockfish-path", limit,
                                             _FakeTokenizer(), runner,
                                             coaching_config=CoachingConfig(max_mistakes=3))
    pipeline_module.analyze_and_coach_games(conn, [GAME_A], "fake-stockfish-path", limit,
                                             _FakeTokenizer(), runner,
                                             coaching_config=CoachingConfig(max_mistakes=7))

    assert len(runner.calls) == 2


def test_analyze_and_coach_games_reusing_a_cached_report_still_generates_feedback(monkeypatch):
    """A report can already be cached (from a plain analyze_and_cache_games
    call) without feedback ever having been generated for it - the
    feedback step must still run in that case, not assume "report
    cached" implies "feedback cached"."""
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", _FakeAnalyzer())
    conn = open_store(":memory:")
    limit = chess.engine.Limit(depth=5)
    pipeline_module.analyze_and_cache_games(conn, [GAME_A], "fake-stockfish-path", limit)

    runner = _FakeRunner(responses=["Fresh feedback."])
    results = pipeline_module.analyze_and_coach_games(conn, [GAME_A], "fake-stockfish-path", limit,
                                                        _FakeTokenizer(), runner)

    assert len(runner.calls) == 1
    assert results[0].feedback == "Fresh feedback."


def test_analyze_and_coach_games_saves_feedback_under_the_game_hash(monkeypatch):
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", _FakeAnalyzer())
    conn = open_store(":memory:")
    runner = _FakeRunner(responses=["Some text."])

    pipeline_module.analyze_and_coach_games(conn, [GAME_A], "fake-stockfish-path",
                                             chess.engine.Limit(depth=5), _FakeTokenizer(), runner)

    game_hash = compute_game_hash(GAME_A)
    # whatever the exact params hash is, *something* got cached for this game
    row = conn.execute("SELECT feedback_text FROM feedback WHERE game_hash = ?", (game_hash,)).fetchone()
    assert row is not None
    assert row[0] == "Some text."


def test_analyze_and_coach_games_does_not_cache_a_prompt_that_did_not_fit(monkeypatch):
    monkeypatch.setattr(pipeline_module, "analyze_pgn_games_parallel", _FakeAnalyzer())
    conn = open_store(":memory:")
    runner = _FakeRunner()  # never called if the prompt doesn't fit
    config = CoachingConfig(token_limit=1)

    results = pipeline_module.analyze_and_coach_games(
        conn, [GAME_A], "fake-stockfish-path", chess.engine.Limit(depth=5),
        _FakeTokenizer(), runner, coaching_config=config,
    )

    assert results[0].feedback is None
    assert runner.calls == []
    game_hash = compute_game_hash(GAME_A)
    row = conn.execute("SELECT 1 FROM feedback WHERE game_hash = ?", (game_hash,)).fetchone()
    assert row is None


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
