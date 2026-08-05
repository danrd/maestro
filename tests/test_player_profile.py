"""Tests for maestro/player_profile.py. Pure aggregation over plain
GameReport/Mistake objects - no engine, no LLM, no fakes needed."""
from __future__ import annotations

import chess

from maestro.game_report import GameReport, Mistake
from maestro.player_profile import (
    OPENING_CUTOFF_MOVE,
    _all_phase_labels,
    _game_reaches_move,
    _phase_for_move,
    build_player_profile,
)


def _mistake(move_number, color=chess.WHITE, loss_cp=100, safe_alternatives=None, move_time_seconds=None):
    return Mistake(move_number=move_number, color=color, played_move="x", best_move="y",
                    loss_cp=loss_cp, safe_alternatives=safe_alternatives, move_time_seconds=move_time_seconds)


def _report(mistakes=None, player_color=chess.WHITE, total_moves=40, game_id="g"):
    return GameReport(game_id=game_id, opening=None, player_color=player_color,
                       total_moves=total_moves, mistakes=mistakes or [])


# -- _phase_for_move --------------------------------------------------------

def test_phase_for_move_within_opening_cutoff():
    assert _phase_for_move(1) == ("opening", 1)
    assert _phase_for_move(OPENING_CUTOFF_MOVE) == ("opening", 1)


def test_phase_for_move_first_chunk_after_opening():
    assert _phase_for_move(OPENING_CUTOFF_MOVE + 1) == ("6-10", 6)
    assert _phase_for_move(10) == ("6-10", 6)


def test_phase_for_move_chunks_continue_in_fives():
    assert _phase_for_move(11) == ("11-15", 11)
    assert _phase_for_move(15) == ("11-15", 11)
    assert _phase_for_move(16) == ("16-20", 16)


def test_phase_for_move_boundary_just_below_long_game_cutoff():
    assert _phase_for_move(39) == ("36-40", 36)


def test_phase_for_move_long_game_catch_all():
    assert _phase_for_move(40) == ("40+", 40)
    assert _phase_for_move(75) == ("40+", 40)


def test_all_phase_labels_are_contiguous_and_end_with_catch_all():
    labels = _all_phase_labels()
    assert labels[0] == ("opening", 1)
    assert labels[-1] == ("40+", 40)
    assert labels[1] == ("6-10", 6)


# -- _game_reaches_move -------------------------------------------------

def test_game_reaches_move_for_white():
    # White's move 3 is ply 5 (1:W 2:B 3:W 4:B 5:W)
    assert _game_reaches_move(total_plies=5, move_number=3, color=chess.WHITE) is True
    assert _game_reaches_move(total_plies=4, move_number=3, color=chess.WHITE) is False


def test_game_reaches_move_for_black():
    # Black's move 3 is ply 6
    assert _game_reaches_move(total_plies=6, move_number=3, color=chess.BLACK) is True
    assert _game_reaches_move(total_plies=5, move_number=3, color=chess.BLACK) is False


# -- build_player_profile ------------------------------------------------

def test_skips_reports_without_a_player_color():
    reports = [_report(player_color=None, mistakes=[_mistake(20)] * 10)]

    profile = build_player_profile(reports, min_games=1)

    assert profile.total_games == 1
    assert profile.phase_stats == []


def test_only_counts_the_tracked_players_own_mistakes():
    # player is White; a Black mistake in the same game must not count
    reports = [_report(player_color=chess.WHITE, total_moves=40,
                        mistakes=[_mistake(20, color=chess.BLACK)] * 10)]

    profile = build_player_profile(reports, min_games=1)

    # the game still "reaches" every phase bucket (it's 40 plies long) -
    # just with zero mistakes counted, since the only mistake was Black's
    assert profile.phase_stats  # buckets exist...
    assert all(bucket.mistake_count == 0 for bucket in profile.phase_stats)  # ...but none attribute a mistake


def test_bucket_omitted_below_min_games():
    # only 2 games reach move 20 ("16-20" phase), threshold is 5
    reports = [_report(total_moves=40, mistakes=[_mistake(18)]) for _ in range(2)]

    profile = build_player_profile(reports, min_games=5)

    assert profile.phase_stats == []


def test_bucket_included_once_min_games_is_met():
    reports = [_report(total_moves=40, mistakes=[_mistake(18, loss_cp=100)]) for _ in range(5)]

    profile = build_player_profile(reports, min_games=5)

    bucket = next(s for s in profile.phase_stats if s.phase == "16-20" and s.color == chess.WHITE)
    assert bucket.games_with_data == 5
    assert bucket.mistake_count == 5
    assert bucket.avg_loss_cp == 100


def test_games_with_data_counts_games_reaching_the_phase_even_without_a_mistake_there():
    # 5 games reach move 18, only 2 of them have a mistake there
    reports = [_report(total_moves=40, mistakes=[_mistake(18)] if i < 2 else []) for i in range(5)]

    profile = build_player_profile(reports, min_games=5)

    bucket = next(s for s in profile.phase_stats if s.phase == "16-20" and s.color == chess.WHITE)
    assert bucket.games_with_data == 5
    assert bucket.mistake_count == 2


def test_unforced_and_forced_counts():
    mistakes = [
        _mistake(18, safe_alternatives=3),  # unforced
        _mistake(18, safe_alternatives=0),  # forced
        _mistake(18, safe_alternatives=None),  # not computed - neither bucket
    ]
    reports = [_report(total_moves=40, mistakes=mistakes) for _ in range(5)]

    profile = build_player_profile(reports, min_games=5)

    bucket = next(s for s in profile.phase_stats if s.phase == "16-20" and s.color == chess.WHITE)
    assert bucket.unforced_count == 5  # one per game * 5 games
    assert bucket.forced_count == 5


def test_avg_move_time_ignores_missing_values():
    mistakes = [_mistake(18, move_time_seconds=10.0), _mistake(19, move_time_seconds=None)]
    reports = [_report(total_moves=40, mistakes=mistakes) for _ in range(5)]

    profile = build_player_profile(reports, min_games=5)

    bucket = next(s for s in profile.phase_stats if s.phase == "16-20" and s.color == chess.WHITE)
    assert bucket.avg_move_time_seconds == 10.0


def test_separates_stats_by_color():
    white_reports = [_report(player_color=chess.WHITE, total_moves=40,
                              mistakes=[_mistake(18, color=chess.WHITE, loss_cp=200)]) for _ in range(5)]
    black_reports = [_report(player_color=chess.BLACK, total_moves=40,
                              mistakes=[_mistake(18, color=chess.BLACK, loss_cp=50)]) for _ in range(5)]

    profile = build_player_profile(white_reports + black_reports, min_games=5)

    white_bucket = next(s for s in profile.phase_stats if s.phase == "16-20" and s.color == chess.WHITE)
    black_bucket = next(s for s in profile.phase_stats if s.phase == "16-20" and s.color == chess.BLACK)
    assert white_bucket.avg_loss_cp == 200
    assert black_bucket.avg_loss_cp == 50
