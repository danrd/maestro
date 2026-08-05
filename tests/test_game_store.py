"""Tests for maestro/game_store.py. Real SQLite (a fresh :memory: or
tmp_path DB per test) - no reason to fake something this cheap and
self-contained."""
from __future__ import annotations

import chess

from maestro.game_report import GameReport, Mistake
from maestro.game_store import (
    compute_game_hash,
    compute_params_hash,
    get_all_game_hashes,
    get_cached_feedback,
    get_cached_report,
    get_pgn,
    import_games,
    open_store,
    save_feedback,
    save_report,
)

GAME_A = """[Event "Test"]
[White "Alice"]
[Black "Bob"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

GAME_B = """[Event "Test"]
[White "Carol"]
[Black "Dave"]

1. d4 d5 1-0
"""


# -- compute_game_hash / compute_params_hash -----------------------------

def test_compute_game_hash_is_stable_across_incidental_formatting():
    padded = GAME_A.replace("1. e4 e5", "1.  e4   e5")  # extra whitespace, still the same moves

    assert compute_game_hash(GAME_A) == compute_game_hash(padded)


def test_compute_game_hash_differs_for_different_games():
    assert compute_game_hash(GAME_A) != compute_game_hash(GAME_B)


def test_compute_params_hash_differs_when_a_param_changes():
    a = compute_params_hash(depth=10, multipv=3)
    b = compute_params_hash(depth=12, multipv=3)

    assert a != b


def test_compute_params_hash_is_order_independent():
    a = compute_params_hash(depth=10, multipv=3)
    b = compute_params_hash(multipv=3, depth=10)

    assert a == b


# -- import_games ---------------------------------------------------------

def test_import_games_inserts_new_games_and_returns_their_hashes():
    conn = open_store(":memory:")

    new_hashes = import_games(conn, [GAME_A, GAME_B])

    assert len(new_hashes) == 2
    assert set(get_all_game_hashes(conn)) == set(new_hashes)


def test_import_games_skips_already_present_games():
    conn = open_store(":memory:")
    import_games(conn, [GAME_A])

    new_hashes = import_games(conn, [GAME_A, GAME_B])

    assert new_hashes == [compute_game_hash(GAME_B)]
    assert len(get_all_game_hashes(conn)) == 2


def test_get_pgn_returns_the_stored_text():
    conn = open_store(":memory:")
    import_games(conn, [GAME_A])

    assert get_pgn(conn, compute_game_hash(GAME_A)) is not None
    assert "Alice" in get_pgn(conn, compute_game_hash(GAME_A))


def test_get_pgn_returns_none_for_an_unknown_hash():
    conn = open_store(":memory:")
    assert get_pgn(conn, "nonexistent") is None


# -- reports: save/get roundtrip ------------------------------------------

def _sample_report():
    return GameReport(
        game_id="Alice vs Bob", opening="C50 - Italian Game", player_color=chess.WHITE,
        total_moves=10, mistakes=[
            Mistake(move_number=3, color=chess.WHITE, played_move="Qh5", best_move="Nf3",
                    loss_cp=91, safe_alternatives=3, move_time_seconds=12.5),
        ],
    )


def test_save_and_get_cached_report_roundtrips():
    conn = open_store(":memory:")
    game_hash = compute_game_hash(GAME_A)
    params_hash = compute_params_hash(depth=10)
    report = _sample_report()

    save_report(conn, game_hash, params_hash, report)
    loaded = get_cached_report(conn, game_hash, params_hash)

    assert loaded == report


def test_get_cached_report_is_none_for_a_miss():
    conn = open_store(":memory:")
    assert get_cached_report(conn, "some-hash", "some-params") is None


def test_get_cached_report_does_not_match_across_different_params():
    conn = open_store(":memory:")
    game_hash = compute_game_hash(GAME_A)
    save_report(conn, game_hash, compute_params_hash(depth=10), _sample_report())

    assert get_cached_report(conn, game_hash, compute_params_hash(depth=12)) is None


# -- feedback: save/get roundtrip ------------------------------------------

def test_save_and_get_cached_feedback_roundtrips():
    conn = open_store(":memory:")
    game_hash = compute_game_hash(GAME_A)
    params_hash = compute_params_hash(max_mistakes=5)

    save_feedback(conn, game_hash, params_hash, "Some coaching text.")

    assert get_cached_feedback(conn, game_hash, params_hash) == "Some coaching text."


def test_get_cached_feedback_is_none_for_a_miss():
    conn = open_store(":memory:")
    assert get_cached_feedback(conn, "some-hash", "some-params") is None


def test_get_cached_feedback_does_not_match_across_different_params():
    conn = open_store(":memory:")
    game_hash = compute_game_hash(GAME_A)
    save_feedback(conn, game_hash, compute_params_hash(max_mistakes=5), "Text A")

    assert get_cached_feedback(conn, game_hash, compute_params_hash(max_mistakes=10)) is None


def test_save_feedback_overwrites_the_same_key():
    conn = open_store(":memory:")
    game_hash = compute_game_hash(GAME_A)
    params_hash = compute_params_hash(max_mistakes=5)

    save_feedback(conn, game_hash, params_hash, "First version")
    save_feedback(conn, game_hash, params_hash, "Second version")

    assert get_cached_feedback(conn, game_hash, params_hash) == "Second version"
