"""Tests for maestro/chess_analysis.py.

No real Stockfish here - a fake engine stub (queued per-call responses,
records what it was called with) stands in, since what's under test is
the move-comparison/fallback logic, not engine search quality. Uses
real python-chess Board/Move/PovScore objects throughout - only the
engine boundary itself is faked.
"""
from __future__ import annotations

import chess
import chess.engine
import chess.pgn

from maestro.chess_analysis import (
    MoveAnalysis,
    _score_to_cp,
    analyze_game,
    analyze_move,
    analyze_position,
)


class _FakeEngine:
    """Returns queued responses (one per .analyse() call) in order;
    records every (board, multipv) it was called with."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def analyse(self, board, limit, multipv=1):
        self.calls.append((board.copy(), multipv))
        return self._responses.pop(0)


def _line(move: chess.Move, cp: int, turn: chess.Color) -> dict:
    return {"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(cp), turn)}


E4 = chess.Move.from_uci("e2e4")
NF3 = chess.Move.from_uci("g1f3")
D4 = chess.Move.from_uci("d2d4")
A3 = chess.Move.from_uci("a2a3")
E5 = chess.Move.from_uci("e7e5")


# -- _score_to_cp -------------------------------------------------------

def test_score_to_cp_returns_the_centipawn_value():
    score = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
    assert _score_to_cp(score) == 150


def test_score_to_cp_maps_mate_scores_to_a_large_finite_value():
    score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert _score_to_cp(score) > 50_000


# -- analyze_position ---------------------------------------------------

def test_analyze_position_returns_move_score_pairs_best_first():
    board = chess.Board()
    engine = _FakeEngine(responses=[[_line(E4, 50, chess.WHITE), _line(NF3, 30, chess.WHITE)]])

    lines = analyze_position(engine, board, chess.engine.Limit(depth=1), multipv=2)

    assert lines == [(E4, 50), (NF3, 30)]


# -- analyze_move ---------------------------------------------------------

def test_analyze_move_when_played_move_is_the_engines_best():
    board = chess.Board()
    engine = _FakeEngine(responses=[
        [_line(E4, 50, chess.WHITE), _line(NF3, 30, chess.WHITE), _line(D4, 20, chess.WHITE)],
    ])

    result = analyze_move(engine, board, E4, chess.engine.Limit(depth=1), multipv=3)

    assert result.played_move == "e4"
    assert result.best_move == "e4"
    assert result.played_score_cp == 50
    assert result.best_score_cp == 50
    assert result.loss_cp == 0
    assert result.candidate_lines == [("e4", 50), ("Nf3", 30), ("d4", 20)]
    assert len(engine.calls) == 1  # played move found directly - no fallback call needed


def test_analyze_move_when_played_move_is_a_worse_candidate():
    board = chess.Board()
    engine = _FakeEngine(responses=[[_line(E4, 50, chess.WHITE), _line(NF3, 30, chess.WHITE)]])

    result = analyze_move(engine, board, NF3, chess.engine.Limit(depth=1), multipv=2)

    assert result.played_move == "Nf3"
    assert result.best_move == "e4"
    assert result.played_score_cp == 30
    assert result.best_score_cp == 50
    assert result.loss_cp == 20


def test_analyze_move_falls_back_to_the_reply_position_when_played_move_is_not_in_multipv():
    board = chess.Board()
    engine = _FakeEngine(responses=[
        [_line(E4, 50, chess.WHITE), _line(NF3, 30, chess.WHITE)],  # before a3 - a3 isn't here
        [_line(E5, 45, chess.BLACK)],  # after a3, Black to move, multipv=1
    ])

    result = analyze_move(engine, board, A3, chess.engine.Limit(depth=1), multipv=2)

    assert result.played_move == "a3"
    assert result.played_score_cp == -45  # negated: good for Black = bad for the mover (White)
    assert result.best_move == "e4"
    assert result.best_score_cp == 50
    assert result.loss_cp == 95
    assert len(engine.calls) == 2
    assert engine.calls[1][1] == 1  # fallback call used multipv=1
    assert engine.calls[1][0].fen() != engine.calls[0][0].fen()  # analyzed the position after a3


def test_analyze_move_does_not_mutate_the_boards_final_state():
    board = chess.Board()
    engine = _FakeEngine(responses=[
        [_line(E4, 50, chess.WHITE)],
        [_line(E5, 45, chess.BLACK)],
    ])
    starting_fen = board.fen()

    analyze_move(engine, board, A3, chess.engine.Limit(depth=1), multipv=1)

    assert board.fen() == starting_fen  # push/pop inside the fallback path left it unchanged


# -- MoveAnalysis.loss_cp -----------------------------------------------

def test_loss_cp_is_none_when_a_score_is_missing():
    move = MoveAnalysis(move_number=1, color=chess.WHITE, played_move="e4",
                         played_score_cp=None, best_move="e4", best_score_cp=50)
    assert move.loss_cp is None


def test_loss_cp_is_clamped_to_zero_rather_than_negative():
    move = MoveAnalysis(move_number=1, color=chess.WHITE, played_move="e4",
                         played_score_cp=60, best_move="d4", best_score_cp=50)
    assert move.loss_cp == 0


# -- analyze_game ---------------------------------------------------------

def _two_ply_game(white="Alice", black="Bob") -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = black
    node = game.add_variation(E4)
    node.add_variation(E5)
    return game


def test_analyze_game_analyzes_every_move_in_order():
    game = _two_ply_game()
    engine = _FakeEngine(responses=[
        [_line(E4, 40, chess.WHITE)],
        [_line(E5, -35, chess.BLACK)],
    ])

    result = analyze_game(engine, game, chess.engine.Limit(depth=1), multipv=1)

    assert len(result.moves) == 2
    assert result.moves[0].played_move == "e4"
    assert result.moves[1].played_move == "e5"
    assert result.game_id == "Alice vs Bob"


def test_analyze_game_prefers_the_event_header_for_game_id():
    game = _two_ply_game()
    game.headers["Event"] = "World Championship"
    engine = _FakeEngine(responses=[
        [_line(E4, 40, chess.WHITE)],
        [_line(E5, -35, chess.BLACK)],
    ])

    result = analyze_game(engine, game, chess.engine.Limit(depth=1), multipv=1)

    assert result.game_id == "World Championship"
