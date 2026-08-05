"""Turn an already-computed GameAnalysis into a GameReport: the flagged
mistakes (loss_cp >= mistake_threshold_cp), sorted worst first, plus
per-game context - opening, which color the tracked player had, and
per-move time spent (parsed from PGN %clk annotations, when present).

No engine calls here - chess_analysis.py already did all of that
(including safe_alternatives, computed inline while the engine and
board were already in hand); this module is pure derivation from data
already computed, so it needs no fakes to test, just real objects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import chess
import chess.pgn

from maestro.chess_analysis import GameAnalysis

_CLOCK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")


@dataclass
class Mistake:
    move_number: int
    color: chess.Color
    played_move: str
    best_move: Optional[str]
    loss_cp: int
    safe_alternatives: Optional[int]
    move_time_seconds: Optional[float] = None


@dataclass
class GameReport:
    game_id: str
    opening: Optional[str]
    player_color: Optional[chess.Color]
    total_moves: int
    mistakes: List[Mistake] = field(default_factory=list)


def _parse_clock_seconds(comment: str) -> Optional[float]:
    match = _CLOCK_RE.search(comment)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _move_clocks(game: chess.pgn.Game) -> List[Optional[float]]:
    """Remaining clock time (seconds) after each mainline move, in ply
    order. None wherever %clk wasn't present in that move's comment -
    common, since clocks are an opt-in field on export, not a
    guarantee."""
    return [_parse_clock_seconds(node.comment) for node in game.mainline()]


def _move_times(clocks: List[Optional[float]]) -> List[Optional[float]]:
    """Seconds spent per move, derived by differencing each side's own
    consecutive clock readings (previous remaining time minus current
    remaining time for that same color, i.e. two plies back). Ignores
    any increment - a real but accepted imprecision, since increment
    isn't reliably present in PGN either."""
    times: List[Optional[float]] = [None] * len(clocks)
    for i, clock in enumerate(clocks):
        prev_index = i - 2  # this color's previous move
        if clock is None or prev_index < 0 or clocks[prev_index] is None:
            continue
        times[i] = max(0.0, clocks[prev_index] - clock)
    return times


def _resolve_player_color(game: chess.pgn.Game, player_name: Optional[str]) -> Optional[chess.Color]:
    if player_name is None:
        return None
    name = player_name.strip().lower()
    if game.headers.get("White", "").strip().lower() == name:
        return chess.WHITE
    if game.headers.get("Black", "").strip().lower() == name:
        return chess.BLACK
    return None


def _resolve_opening(game: chess.pgn.Game) -> Optional[str]:
    opening = game.headers.get("Opening", "")
    eco = game.headers.get("ECO", "")
    if opening and eco:
        return f"{eco} - {opening}"
    return opening or eco or None


def build_game_report(analysis: GameAnalysis, game: chess.pgn.Game,
                       mistake_threshold_cp: int = 50,
                       player_name: Optional[str] = None) -> GameReport:
    """Derive a GameReport from `analysis` (from chess_analysis.analyze_game,
    run with the same mistake_threshold_cp) and `game` (the same game the
    analysis was run on - needed here only for headers/comments, not
    replayed). `mistake_threshold_cp` filters which already-scored moves
    count as mistakes; it does not itself trigger new engine work."""
    move_times = _move_times(_move_clocks(game))

    mistakes = []
    for i, move in enumerate(analysis.moves):
        loss = move.loss_cp
        if loss is None or loss < mistake_threshold_cp:
            continue
        mistakes.append(Mistake(
            move_number=move.move_number, color=move.color, played_move=move.played_move,
            best_move=move.best_move, loss_cp=loss, safe_alternatives=move.safe_alternatives,
            move_time_seconds=move_times[i] if i < len(move_times) else None,
        ))
    mistakes.sort(key=lambda m: m.loss_cp, reverse=True)

    return GameReport(
        game_id=analysis.game_id, opening=_resolve_opening(game),
        player_color=_resolve_player_color(game, player_name),
        total_moves=len(analysis.moves), mistakes=mistakes,
    )
