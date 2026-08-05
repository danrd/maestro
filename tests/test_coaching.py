"""Tests for maestro/coaching.py.

No real LLM - a fake tokenizer/runner stand in, since what's under test
is prompt assembly (does the unforced-error-vs-forced-moment distinction
actually reach the prompt, does capping/fit-checking work) not
generation quality.
"""
from __future__ import annotations

import chess

from maestro.coaching import (
    CoachingConfig,
    _format_mistake,
    _format_mistakes,
    build_coaching_prompt,
    generate_coaching_feedback,
)
from maestro.game_report import GameReport, Mistake


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


def _mistake(loss_cp=91, safe_alternatives=None, move_time_seconds=None,
             move_number=3, color=chess.WHITE, played_move="Qh5", best_move="Nf3"):
    return Mistake(move_number=move_number, color=color, played_move=played_move,
                    best_move=best_move, loss_cp=loss_cp, safe_alternatives=safe_alternatives,
                    move_time_seconds=move_time_seconds)


def _report(mistakes=None, opening="C50 - Italian Game", player_color=chess.WHITE, total_moves=20):
    return GameReport(game_id="test-game", opening=opening, player_color=player_color,
                       total_moves=total_moves, mistakes=mistakes or [])


# -- _format_mistake ------------------------------------------------------

def test_format_mistake_flags_an_unforced_error_when_safe_alternatives_exist():
    text = _format_mistake(_mistake(safe_alternatives=3), 1)
    assert "unforced error" in text
    assert "3" in text


def test_format_mistake_flags_a_forced_moment_when_no_safe_alternatives():
    text = _format_mistake(_mistake(safe_alternatives=0), 1)
    assert "narrow, forced" in text
    assert "unforced" not in text


def test_format_mistake_omits_the_alternatives_framing_when_not_computed():
    text = _format_mistake(_mistake(safe_alternatives=None), 1)
    assert "unforced" not in text
    assert "forced" not in text


def test_format_mistake_includes_move_time_when_present():
    text = _format_mistake(_mistake(move_time_seconds=12.0), 1)
    assert "12s" in text


def test_format_mistake_omits_move_time_when_absent():
    text = _format_mistake(_mistake(move_time_seconds=None), 1)
    assert "Time spent" not in text


# -- _format_mistakes -------------------------------------------------------

def test_format_mistakes_reports_none_found_for_a_clean_game():
    assert "No mistakes" in _format_mistakes(_report(mistakes=[]), max_mistakes=5)


def test_format_mistakes_caps_at_max_mistakes_and_notes_the_remainder():
    mistakes = [_mistake(loss_cp=100 - i, move_number=i + 1) for i in range(7)]
    text = _format_mistakes(_report(mistakes=mistakes), max_mistakes=3)

    assert text.count("Move ") == 3
    assert "4 more mistake" in text


# -- build_coaching_prompt ----------------------------------------------

def test_build_coaching_prompt_includes_opening_color_and_mistakes():
    report = _report(mistakes=[_mistake()])
    prompt = build_coaching_prompt(report, _FakeTokenizer())

    assert prompt is not None
    assert "Italian Game" in prompt
    assert "White" in prompt
    assert "Qh5" in prompt


def test_build_coaching_prompt_includes_language_instruction_when_set():
    config = CoachingConfig(language="Russian")
    prompt = build_coaching_prompt(_report(), _FakeTokenizer(), config)

    assert "Russian" in prompt


def test_build_coaching_prompt_omits_language_instruction_when_unset():
    prompt = build_coaching_prompt(_report(), _FakeTokenizer())
    assert "Write your feedback in" not in prompt


def test_build_coaching_prompt_returns_none_when_it_does_not_fit():
    config = CoachingConfig(token_limit=1)
    prompt = build_coaching_prompt(_report(mistakes=[_mistake()]), _FakeTokenizer(), config)
    assert prompt is None


# -- generate_coaching_feedback -------------------------------------------

def test_generate_coaching_feedback_happy_path():
    report = _report(mistakes=[_mistake()])
    runner = _FakeRunner(responses=["Some coaching text."])

    feedback = generate_coaching_feedback(report, _FakeTokenizer(), runner)

    assert feedback == "Some coaching text."
    assert len(runner.calls) == 1


def test_generate_coaching_feedback_skips_generation_when_prompt_does_not_fit():
    config = CoachingConfig(token_limit=1)
    runner = _FakeRunner()

    feedback = generate_coaching_feedback(_report(mistakes=[_mistake()]), _FakeTokenizer(), runner, config)

    assert feedback is None
    assert runner.calls == []
