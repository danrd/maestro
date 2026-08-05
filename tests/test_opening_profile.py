"""Tests for maestro/opening_profile.py. Pure derivation over plain
GameReport objects and move tuples - no engine, no LLM."""
from __future__ import annotations

import chess

from maestro.game_report import GameReport, Mistake
from maestro.opening_profile import build_opening_groups

RUY = ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7")
CARO = ("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7", "Nf3", "Ngf6")


def _report(game_hash, mistakes=None, player_color=chess.WHITE):
    return GameReport(game_id=game_hash, opening=None, player_color=player_color,
                       total_moves=40, mistakes=mistakes or [])


def _mistake(loss_cp=100, color=chess.WHITE):
    return Mistake(move_number=3, color=color, played_move="x", best_move="y",
                    loss_cp=loss_cp, safe_alternatives=None)


def test_groups_games_sharing_a_signature_and_reports_the_known_name_and_idea():
    reports = {f"g{i}": _report(f"g{i}", mistakes=[_mistake(100)]) for i in range(5)}
    moves = {game_hash: RUY for game_hash in reports}

    groups = build_opening_groups(reports, moves, min_games=5)

    assert len(groups) == 1
    group = groups[0]
    assert group.signature == RUY
    assert group.name == "Ruy Lopez: Closed"
    assert group.idea is not None and "Ruy Lopez" in group.idea
    assert group.game_count == 5
    assert group.mistake_count == 5
    assert group.avg_loss_cp == 100


def test_excludes_groups_below_min_games():
    reports = {f"g{i}": _report(f"g{i}") for i in range(3)}
    moves = {game_hash: RUY for game_hash in reports}

    groups = build_opening_groups(reports, moves, min_games=5)

    assert groups == []


def test_separates_distinct_openings_into_distinct_groups():
    reports = {}
    moves = {}
    for i in range(5):
        reports[f"ruy{i}"] = _report(f"ruy{i}", mistakes=[_mistake(200)])
        moves[f"ruy{i}"] = RUY
        reports[f"caro{i}"] = _report(f"caro{i}", mistakes=[_mistake(50)])
        moves[f"caro{i}"] = CARO

    groups = build_opening_groups(reports, moves, min_games=5)

    assert len(groups) == 2
    by_name = {g.name: g for g in groups}
    assert by_name["Ruy Lopez: Closed"].avg_loss_cp == 200
    assert by_name["Caro-Kann Defense"].avg_loss_cp == 50


def test_only_counts_the_tracked_players_own_mistakes():
    reports = {f"g{i}": _report(f"g{i}", player_color=chess.WHITE,
                                 mistakes=[_mistake(999, color=chess.BLACK)]) for i in range(5)}
    moves = {game_hash: RUY for game_hash in reports}

    groups = build_opening_groups(reports, moves, min_games=5)

    assert groups[0].mistake_count == 0
    assert groups[0].avg_loss_cp == 0.0


def test_unrecognized_signature_still_produces_a_group_without_name_or_idea():
    # not real SAN - guaranteed not to match anything in the vendored data
    unknown = ("zz1", "zz2", "zz3", "zz4", "zz5", "zz6", "zz7", "zz8", "zz9", "zz10")
    reports = {f"g{i}": _report(f"g{i}", mistakes=[_mistake(10)]) for i in range(5)}
    moves = {game_hash: unknown for game_hash in reports}

    groups = build_opening_groups(reports, moves, min_games=5)

    assert len(groups) == 1
    assert groups[0].name is None
    assert groups[0].idea is None
