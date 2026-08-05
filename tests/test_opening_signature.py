"""Tests for maestro/opening_signature.py."""
from __future__ import annotations

from maestro.opening_signature import (
    assign_opening_signatures,
    extract_move_prefix,
    group_by_signature,
)

# 10-ply (5 full moves per side) Ruy Lopez-ish line
RUY = ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7")
# A completely different 10-ply line
CARO = ("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7", "Nf3", "Ngf6")


def _n_games(prefix, n, start=0):
    return {f"g{i}": prefix for i in range(start, start + n)}


# -- assign_opening_signatures -------------------------------------------

def test_assigns_full_depth_signature_when_min_games_is_met():
    games = _n_games(RUY, 5)

    signatures = assign_opening_signatures(games, min_games=5)

    assert all(sig == RUY for sig in signatures.values())


def test_shrinks_to_a_shorter_prefix_when_full_depth_lacks_support():
    # 5 games share the first 6 plies (3 full moves), but diverge after that
    games = {}
    common = RUY[:6]
    for i, tail in enumerate([RUY[6:], CARO[6:], ("h3", "h6"), ("c3", "b5"), ("d3", "d6")]):
        games[f"g{i}"] = common + tail

    signatures = assign_opening_signatures(games, min_games=5)

    assert all(sig == common for sig in signatures.values())


def test_returns_none_when_no_depth_reaches_min_games():
    games = {"g0": RUY, "g1": CARO, "g2": ("d4", "d5", "c4", "e6", "Nc3", "Nf6")}

    signatures = assign_opening_signatures(games, min_games=5)

    assert all(sig is None for sig in signatures.values())


def test_different_groups_get_different_signatures_when_each_meets_the_threshold():
    games = {**_n_games(RUY, 5), **_n_games(CARO, 5, start=5)}

    signatures = assign_opening_signatures(games, min_games=5)

    ruy_sigs = {signatures[f"g{i}"] for i in range(5)}
    caro_sigs = {signatures[f"g{i}"] for i in range(5, 10)}
    assert ruy_sigs == {RUY}
    assert caro_sigs == {CARO}


def test_games_shorter_than_the_minimum_depth_get_no_signature():
    games = _n_games(RUY[:4], 10)  # only 4 plies - below MIN_SIGNATURE_FULL_MOVES*2=6

    signatures = assign_opening_signatures(games, min_games=5)

    assert all(sig is None for sig in signatures.values())


# -- group_by_signature ---------------------------------------------------

def test_group_by_signature_inverts_the_mapping():
    signatures = {"g0": RUY, "g1": RUY, "g2": None}

    groups = group_by_signature(signatures)

    assert set(groups[RUY]) == {"g0", "g1"}
    assert groups[None] == ["g2"]


# -- extract_move_prefix ---------------------------------------------------

def test_extract_move_prefix_reads_san_moves_in_order():
    pgn = """[Event "Test"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0
"""
    prefix = extract_move_prefix(pgn)

    assert prefix == RUY  # capped at the default depth (10 plies)


def test_extract_move_prefix_handles_a_short_game():
    pgn = """[Event "Test"]

1. e4 e5 1-0
"""
    assert extract_move_prefix(pgn) == ("e4", "e5")


def test_extract_move_prefix_returns_empty_tuple_for_an_unparseable_game():
    assert extract_move_prefix("") == ()
