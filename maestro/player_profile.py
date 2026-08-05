"""Aggregate mistakes across many GameReports into a PlayerProfile:
per-phase, per-color statistics - pure derivation, no engine or LLM
calls, same as game_report.py. The LLM step downstream (coaching.py)
turns these already-computed numbers into prose; it never sees raw
per-move data for dozens of games at once.

Phase buckets mirror chess_analysis.py's opening_ply_cutoff convention
(5 full moves per side = the opening), then chunk the rest of the game
in 5-move steps, with every move from 40 onward folded into one final
"40+" bucket rather than producing ever-thinner buckets for the rare
very long game.

A bucket is only reported if at least `min_games` games actually
reached it (by move count) for that color - otherwise the "statistic"
is really just noise from one or two games, which is worse than not
reporting anything.

Only reports with `player_color` set (see game_report.build_game_report's
`player_name` param) contribute here - without knowing which side the
tracked player had, a mistake can't be attributed to them at all.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess

from maestro.game_report import GameReport, Mistake

OPENING_CUTOFF_MOVE = 5  # full moves per side - matches opening_ply_cutoff=10 (10 plies)
CHUNK_SIZE = 5
LONG_GAME_CUTOFF_MOVE = 40


@dataclass
class PhaseStats:
    phase: str
    color: chess.Color
    games_with_data: int  # distinct games that reached this phase for this color
    mistake_count: int
    avg_loss_cp: float
    unforced_count: int  # mistakes with safe_alternatives > 0
    forced_count: int  # mistakes with safe_alternatives == 0
    avg_move_time_seconds: Optional[float] = None


@dataclass
class PlayerProfile:
    total_games: int
    phase_stats: List[PhaseStats] = field(default_factory=list)


def _phase_for_move(move_number: int) -> Tuple[str, int]:
    """Returns (phase_label, phase_start_move) for `move_number`."""
    if move_number <= OPENING_CUTOFF_MOVE:
        return "opening", 1
    if move_number >= LONG_GAME_CUTOFF_MOVE:
        return "40+", LONG_GAME_CUTOFF_MOVE
    chunk_index = (move_number - OPENING_CUTOFF_MOVE - 1) // CHUNK_SIZE
    start = OPENING_CUTOFF_MOVE + 1 + chunk_index * CHUNK_SIZE
    end = start + CHUNK_SIZE - 1
    return f"{start}-{end}", start


def _all_phase_labels() -> List[Tuple[str, int]]:
    labels = [("opening", 1)]
    start = OPENING_CUTOFF_MOVE + 1
    while start < LONG_GAME_CUTOFF_MOVE:
        end = start + CHUNK_SIZE - 1
        labels.append((f"{start}-{end}", start))
        start += CHUNK_SIZE
    labels.append(("40+", LONG_GAME_CUTOFF_MOVE))
    return labels


def _game_reaches_move(total_plies: int, move_number: int, color: chess.Color) -> bool:
    """Whether a game with `total_plies` half-moves played included a
    move by `color` at `move_number` - i.e. whether it's part of this
    bucket's sample for that color at all, mistake or not."""
    ply = 2 * move_number - 1 if color else 2 * move_number
    return total_plies >= ply


def build_player_profile(reports: List[GameReport], min_games: int = 5) -> PlayerProfile:
    """Aggregate `reports` into per-phase, per-color statistics. Reports
    without `player_color` set are skipped entirely - see this module's
    docstring."""
    phase_labels = _all_phase_labels()
    usable_reports = [r for r in reports if r.player_color is not None]

    reach_counts: Dict[Tuple[str, chess.Color], int] = Counter()
    for report in usable_reports:
        for phase_label, start_move in phase_labels:
            if _game_reaches_move(report.total_moves, start_move, report.player_color):
                reach_counts[(phase_label, report.player_color)] += 1

    mistakes_by_bucket: Dict[Tuple[str, chess.Color], List[Mistake]] = defaultdict(list)
    for report in usable_reports:
        for mistake in report.mistakes:
            if mistake.color != report.player_color:
                continue  # only the tracked player's own mistakes count
            phase_label, _ = _phase_for_move(mistake.move_number)
            mistakes_by_bucket[(phase_label, mistake.color)].append(mistake)

    phase_stats = []
    for phase_label, _ in phase_labels:
        for color in (chess.WHITE, chess.BLACK):
            games_with_data = reach_counts.get((phase_label, color), 0)
            if games_with_data < min_games:
                continue

            bucket_mistakes = mistakes_by_bucket.get((phase_label, color), [])
            unforced = sum(1 for m in bucket_mistakes if m.safe_alternatives)
            forced = sum(1 for m in bucket_mistakes if m.safe_alternatives == 0)
            times = [m.move_time_seconds for m in bucket_mistakes if m.move_time_seconds is not None]

            phase_stats.append(PhaseStats(
                phase=phase_label, color=color, games_with_data=games_with_data,
                mistake_count=len(bucket_mistakes),
                avg_loss_cp=(sum(m.loss_cp for m in bucket_mistakes) / len(bucket_mistakes)
                             if bucket_mistakes else 0.0),
                unforced_count=unforced, forced_count=forced,
                avg_move_time_seconds=(sum(times) / len(times) if times else None),
            ))

    return PlayerProfile(total_games=len(reports), phase_stats=phase_stats)
