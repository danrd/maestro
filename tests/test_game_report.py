"""Tests for maestro/game_report.py.

No engine, no fakes needed - this module is pure derivation from an
already-computed GameAnalysis plus PGN headers/comments, both built
directly here as plain data.
"""
from __future__ import annotations

import chess
import chess.pgn

from maestro.chess_analysis import GameAnalysis, MoveAnalysis
from maestro.game_report import (
    GameReport,
    _move_clocks,
    _move_times,
    _parse_clock_seconds,
    _resolve_opening,
    _resolve_player_color,
    build_game_report,
)


# -- _parse_clock_seconds -----------------------------------------------

def test_parse_clock_seconds_extracts_hms():
    assert _parse_clock_seconds("[%clk 0:05:30]") == 330.0


def test_parse_clock_seconds_returns_none_when_absent():
    assert _parse_clock_seconds("just a regular comment") is None


def test_parse_clock_seconds_handles_fractional_seconds():
    assert _parse_clock_seconds("[%clk 1:00:02.5]") == 3602.5


# -- _move_clocks / _move_times ------------------------------------------

def _game_with_clocks(clock_strings):
    """Build a game with one comment per ply, in order."""
    game = chess.pgn.Game()
    node = game
    moves = [chess.Move.from_uci(uci) for uci in ("e2e4", "e7e5", "g1f3", "b8c6")]
    for move, clock in zip(moves, clock_strings):
        node = node.add_variation(move)
        if clock is not None:
            node.comment = f"[%clk {clock}]"
    return game


def test_move_clocks_parses_each_plys_comment_in_order():
    game = _game_with_clocks(["0:05:00", "0:05:00", "0:04:50", "0:04:55"])

    clocks = _move_clocks(game)

    assert clocks == [300.0, 300.0, 290.0, 295.0]


def test_move_clocks_has_none_for_plies_without_a_clock_comment():
    game = _game_with_clocks(["0:05:00", None, "0:04:50", None])

    assert _move_clocks(game) == [300.0, None, 290.0, None]


def test_move_times_diffs_each_colors_own_consecutive_clocks():
    # White: 300 -> 290 (spent 10s on move 3, ply index 2)
    # Black: 300 -> 295 (spent 5s on move 4, ply index 3)
    clocks = [300.0, 300.0, 290.0, 295.0]

    times = _move_times(clocks)

    assert times == [None, None, 10.0, 5.0]


def test_move_times_is_none_when_this_plys_own_clock_reading_is_missing():
    clocks = [300.0, 300.0, None, 295.0]

    assert _move_times(clocks)[2] is None


def test_move_times_is_none_when_the_same_colors_prior_clock_reading_is_missing():
    # index 3 (Black) would diff against index 1, which is missing here -
    # index 2 (White) is unaffected since it diffs against index 0.
    clocks = [300.0, None, 290.0, 295.0]

    times = _move_times(clocks)

    assert times[2] == 10.0
    assert times[3] is None


# -- _resolve_player_color / _resolve_opening ----------------------------

def test_resolve_player_color_matches_white_case_insensitively():
    game = chess.pgn.Game()
    game.headers["White"] = "Magnus"
    game.headers["Black"] = "Hikaru"

    assert _resolve_player_color(game, "magnus") == chess.WHITE


def test_resolve_player_color_matches_black():
    game = chess.pgn.Game()
    game.headers["White"] = "Magnus"
    game.headers["Black"] = "Hikaru"

    assert _resolve_player_color(game, "Hikaru") == chess.BLACK


def test_resolve_player_color_is_none_when_name_does_not_match_either_side():
    game = chess.pgn.Game()
    game.headers["White"] = "Magnus"
    game.headers["Black"] = "Hikaru"

    assert _resolve_player_color(game, "Fabiano") is None


def test_resolve_player_color_is_none_when_no_name_given():
    game = chess.pgn.Game()
    assert _resolve_player_color(game, None) is None


def test_resolve_opening_combines_eco_and_name_when_both_present():
    game = chess.pgn.Game()
    game.headers["ECO"] = "C50"
    game.headers["Opening"] = "Italian Game"

    assert _resolve_opening(game) == "C50 - Italian Game"


def test_resolve_opening_falls_back_to_whichever_tag_is_present():
    game = chess.pgn.Game()
    game.headers["ECO"] = "C50"

    assert _resolve_opening(game) == "C50"


def test_resolve_opening_is_none_when_neither_tag_is_set():
    game = chess.pgn.Game()
    assert _resolve_opening(game) is None


# -- build_game_report ---------------------------------------------------

def _move(move_number, color, played_move, loss_cp, safe_alternatives=None, best_move="best"):
    best_score = 100
    played_score = best_score - loss_cp if loss_cp is not None else None
    return MoveAnalysis(
        move_number=move_number, color=color, played_move=played_move,
        played_score_cp=played_score, best_move=best_move, best_score_cp=best_score,
        safe_alternatives=safe_alternatives,
    )


def test_build_game_report_filters_by_threshold_and_sorts_worst_first():
    analysis = GameAnalysis(game_id="Test", moves=[
        _move(1, chess.WHITE, "e4", loss_cp=0),
        _move(2, chess.BLACK, "a6", loss_cp=30),    # below threshold - excluded
        _move(3, chess.WHITE, "Qh5", loss_cp=91, safe_alternatives=3),
        _move(4, chess.BLACK, "Nf6", loss_cp=150, safe_alternatives=0),
    ])
    game = chess.pgn.Game()

    report = build_game_report(analysis, game, mistake_threshold_cp=50)

    assert isinstance(report, GameReport)
    assert [m.played_move for m in report.mistakes] == ["Nf6", "Qh5"]  # worst first
    assert report.mistakes[0].loss_cp == 150
    assert report.mistakes[0].safe_alternatives == 0
    assert report.total_moves == 4


def test_build_game_report_attaches_move_time_by_position():
    game = _game_with_clocks(["0:05:00", "0:05:00", "0:04:40", "0:04:59"])
    analysis = GameAnalysis(game_id="Test", moves=[
        _move(1, chess.WHITE, "e4", loss_cp=0),
        _move(1, chess.BLACK, "e5", loss_cp=0),
        _move(2, chess.WHITE, "Nf3", loss_cp=200),  # spent 20s (300 -> 280... wait see clocks)
        _move(2, chess.BLACK, "Nc6", loss_cp=0),
    ])

    report = build_game_report(analysis, game, mistake_threshold_cp=50)

    assert len(report.mistakes) == 1
    assert report.mistakes[0].move_time_seconds == 20.0


def test_build_game_report_includes_opening_and_player_color():
    game = chess.pgn.Game()
    game.headers["ECO"] = "C50"
    game.headers["White"] = "Magnus"
    game.headers["Black"] = "Hikaru"
    analysis = GameAnalysis(game_id="Test", moves=[])

    report = build_game_report(analysis, game, player_name="Magnus")

    assert report.opening == "C50"
    assert report.player_color == chess.WHITE


def test_build_game_report_returns_no_mistakes_for_a_clean_game():
    analysis = GameAnalysis(game_id="Test", moves=[_move(1, chess.WHITE, "e4", loss_cp=0)])
    game = chess.pgn.Game()

    report = build_game_report(analysis, game)

    assert report.mistakes == []
