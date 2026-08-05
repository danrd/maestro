"""Tests for maestro/profile_coaching.py.

No real LLM - a fake tokenizer/runner stand in, same pattern as
test_coaching.py. What's under test is prompt assembly from already-
computed PlayerProfile/OpeningGroupSummary data, not generation quality.
"""
from __future__ import annotations

import chess

from maestro.opening_profile import OpeningGroupSummary
from maestro.player_profile import PhaseStats, PlayerProfile
from maestro.profile_coaching import (
    ProfileCoachingConfig,
    _format_opening_notes,
    _format_phase_stats,
    build_profile_coaching_prompt,
    generate_profile_coaching_feedback,
)


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


def _phase_stat(phase="16-20", color=chess.WHITE, avg_loss_cp=100.0, games_with_data=5,
                 mistake_count=5, unforced_count=0, forced_count=0, avg_move_time_seconds=None):
    return PhaseStats(phase=phase, color=color, games_with_data=games_with_data,
                       mistake_count=mistake_count, avg_loss_cp=avg_loss_cp,
                       unforced_count=unforced_count, forced_count=forced_count,
                       avg_move_time_seconds=avg_move_time_seconds)


def _opening_group(name="Sicilian Defense", idea="Fights for d4 from the flank.",
                    avg_loss_cp=150.0, game_count=5, mistake_count=5,
                    signature=("e4", "c5")):
    return OpeningGroupSummary(signature=signature, name=name, idea=idea,
                                game_count=game_count, mistake_count=mistake_count,
                                avg_loss_cp=avg_loss_cp)


# -- _format_phase_stats -----------------------------------------------

def test_format_phase_stats_reports_no_data_when_empty():
    profile = PlayerProfile(total_games=3, phase_stats=[])
    assert "Not enough games" in _format_phase_stats(profile, max_buckets=6)


def test_format_phase_stats_sorts_worst_first():
    profile = PlayerProfile(total_games=10, phase_stats=[
        _phase_stat(phase="opening", avg_loss_cp=20.0),
        _phase_stat(phase="16-20", avg_loss_cp=300.0),
    ])

    text = _format_phase_stats(profile, max_buckets=6)

    assert text.index("16-20") < text.index("opening")


def test_format_phase_stats_caps_at_max_buckets():
    profile = PlayerProfile(total_games=10, phase_stats=[
        _phase_stat(phase=f"{i}", avg_loss_cp=float(i)) for i in range(10)
    ])

    text = _format_phase_stats(profile, max_buckets=3)

    assert text.count("Moves") == 3


def test_format_phase_stats_includes_unforced_forced_and_time_when_present():
    profile = PlayerProfile(total_games=5, phase_stats=[
        _phase_stat(unforced_count=2, forced_count=1, avg_move_time_seconds=15.0),
    ])

    text = _format_phase_stats(profile, max_buckets=6)

    assert "unforced" in text
    assert "forced" in text
    assert "15s" in text


# -- _format_opening_notes -------------------------------------------------

def test_format_opening_notes_reports_no_data_when_empty():
    assert "Not enough games" in _format_opening_notes([], max_openings=5)


def test_format_opening_notes_sorts_worst_first():
    groups = [_opening_group(name="Caro-Kann Defense", avg_loss_cp=20.0),
              _opening_group(name="Sicilian Defense", avg_loss_cp=300.0)]

    text = _format_opening_notes(groups, max_openings=5)

    assert text.index("Sicilian") < text.index("Caro-Kann")


def test_format_opening_notes_uses_moves_when_no_name_is_known():
    groups = [_opening_group(name=None, idea=None, signature=("e4", "z9"))]

    text = _format_opening_notes(groups, max_openings=5)

    assert "e4 z9" in text


def test_format_opening_notes_includes_the_idea_when_present():
    groups = [_opening_group(idea="Fights for d4 from the flank.")]

    text = _format_opening_notes(groups, max_openings=5)

    assert "Fights for d4 from the flank." in text


def test_format_opening_notes_caps_at_max_openings():
    groups = [_opening_group(name=f"Opening {i}", avg_loss_cp=float(i)) for i in range(10)]

    text = _format_opening_notes(groups, max_openings=3)

    assert text.count("Opening ") == 3


# -- build_profile_coaching_prompt / generate_profile_coaching_feedback -----

def test_build_profile_coaching_prompt_includes_phase_and_opening_data():
    profile = PlayerProfile(total_games=12, phase_stats=[_phase_stat()])
    groups = [_opening_group()]

    prompt = build_profile_coaching_prompt(profile, groups, _FakeTokenizer())

    assert prompt is not None
    assert "12" in prompt
    assert "Sicilian" in prompt


def test_build_profile_coaching_prompt_returns_none_when_it_does_not_fit():
    config = ProfileCoachingConfig(token_limit=1)
    prompt = build_profile_coaching_prompt(
        PlayerProfile(total_games=1, phase_stats=[_phase_stat()]), [_opening_group()],
        _FakeTokenizer(), config,
    )
    assert prompt is None


def test_generate_profile_coaching_feedback_happy_path():
    runner = _FakeRunner(responses=["Some cross-game feedback."])
    profile = PlayerProfile(total_games=5, phase_stats=[_phase_stat()])

    feedback = generate_profile_coaching_feedback(profile, [_opening_group()], _FakeTokenizer(), runner)

    assert feedback == "Some cross-game feedback."
    assert len(runner.calls) == 1


def test_generate_profile_coaching_feedback_skips_generation_when_prompt_does_not_fit():
    config = ProfileCoachingConfig(token_limit=1)
    runner = _FakeRunner()

    feedback = generate_profile_coaching_feedback(
        PlayerProfile(total_games=1, phase_stats=[_phase_stat()]), [_opening_group()],
        _FakeTokenizer(), runner, config,
    )

    assert feedback is None
    assert runner.calls == []
