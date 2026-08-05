"""Tests for maestro/app.py.

Requires gradio (the `ui` extra) to import at all - skipped otherwise,
same pattern used for other optional-dependency-gated tests here.
"""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

import chess  # noqa: E402

from maestro.app import (  # noqa: E402
    build_app,
    build_runner,
    find_stockfish,
    format_game_report,
    format_profile,
)
from maestro.game_report import GameReport, Mistake  # noqa: E402
from maestro.opening_profile import OpeningGroupSummary  # noqa: E402
from maestro.player_profile import PhaseStats, PlayerProfile  # noqa: E402


def _mistake(loss_cp=100, color=chess.WHITE, safe_alternatives=None):
    return Mistake(move_number=3, color=color, played_move="Qh5", best_move="Nf3",
                    loss_cp=loss_cp, safe_alternatives=safe_alternatives)


# -- find_stockfish -------------------------------------------------------

def test_find_stockfish_uses_the_env_var_when_it_points_to_a_real_file(tmp_path, monkeypatch):
    fake_binary = tmp_path / "stockfish"
    fake_binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("STOCKFISH_PATH", str(fake_binary))

    assert find_stockfish() == str(fake_binary)


def test_find_stockfish_ignores_a_nonexistent_env_var_path(monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", "/definitely/not/a/real/path")

    result = find_stockfish()

    assert result != "/definitely/not/a/real/path"


# -- build_runner -----------------------------------------------------

def test_build_runner_returns_none_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert build_runner() is None


def test_build_runner_constructs_a_runner_when_a_key_is_set(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-construction-only")

    runner = build_runner()

    assert runner is not None


# -- format_game_report ---------------------------------------------------

def test_format_game_report_reports_no_mistakes_for_a_clean_game():
    report = GameReport(game_id="g1", opening="C50", player_color=chess.WHITE,
                         total_moves=20, mistakes=[])

    text = format_game_report(report)

    assert "No mistakes" in text
    assert "C50" in text


def test_format_game_report_lists_mistakes_with_key_details():
    report = GameReport(game_id="g1", opening=None, player_color=chess.WHITE, total_moves=20,
                         mistakes=[_mistake(loss_cp=150, safe_alternatives=2)])

    text = format_game_report(report)

    assert "unknown" in text  # no opening tag
    assert "Qh5" in text
    assert "Nf3" in text
    assert "1.50 pawns" in text
    assert "2 safe alternative" in text


# -- format_profile -------------------------------------------------------

def test_format_profile_reports_not_enough_data_when_empty():
    text = format_profile(PlayerProfile(total_games=1, phase_stats=[]), [])

    assert "Not enough games" in text


def test_format_profile_includes_phase_and_opening_sections():
    profile = PlayerProfile(total_games=5, phase_stats=[
        PhaseStats(phase="16-20", color=chess.WHITE, games_with_data=5, mistake_count=3,
                   avg_loss_cp=120.0, unforced_count=2, forced_count=1),
    ])
    groups = [OpeningGroupSummary(signature=("e4", "c5"), name="Sicilian Defense",
                                   idea="Fights for d4.", game_count=5, mistake_count=4,
                                   avg_loss_cp=90.0)]

    text = format_profile(profile, groups)

    assert "16-20" in text
    assert "Sicilian Defense" in text
    assert "Fights for d4." in text


# -- build_app --------------------------------------------------------

def test_build_app_returns_a_gradio_blocks_instance():
    app = build_app()

    assert isinstance(app, gr.Blocks)
