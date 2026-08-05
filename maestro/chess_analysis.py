"""Analyze a single game move-by-move with Stockfish (or any UCI engine):
for each position, ask for the top `multipv` candidate lines rather than
just the single best move - "was there only one good move here, or
several" matters for telling a real blunder from an understandable
choice among options, which a best-move-only comparison can't tell
apart.

No engine process management here - callers pass in an already-started
`chess.engine.SimpleEngine` (see analysis_pool.py for running many games
across a pool of persistent engine processes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import chess
import chess.engine
import chess.pgn


@dataclass
class MoveAnalysis:
    move_number: int
    color: chess.Color
    played_move: str  # SAN
    played_score_cp: Optional[int]  # from the mover's own perspective
    best_move: Optional[str]  # SAN
    best_score_cp: Optional[int]
    candidate_lines: List[Tuple[str, int]] = field(default_factory=list)  # (SAN, score_cp), best first

    @property
    def loss_cp(self) -> Optional[int]:
        """How many centipawns worse the played move was than the
        engine's best - 0 if it *was* the best (or tied), None if no
        score is available for either side of the comparison."""
        if self.best_score_cp is None or self.played_score_cp is None:
            return None
        return max(0, self.best_score_cp - self.played_score_cp)


@dataclass
class GameAnalysis:
    game_id: str
    moves: List[MoveAnalysis] = field(default_factory=list)


def _score_to_cp(score: chess.engine.PovScore) -> int:
    """Centipawn value from the mover's perspective. Mate scores are
    mapped to a large-but-finite value (via python-chess's own
    mate_score= convention) so they still sort/compare sensibly
    against ordinary centipawn scores instead of needing separate
    handling everywhere."""
    return score.relative.score(mate_score=100_000)


def analyze_position(engine: chess.engine.SimpleEngine, board: chess.Board,
                      limit: chess.engine.Limit, multipv: int = 3) -> List[Tuple[chess.Move, int]]:
    """Top `multipv` engine lines for `board`, as (move, score_cp)
    pairs from the perspective of the side to move, best first."""
    info = engine.analyse(board, limit, multipv=multipv)
    lines = []
    for entry in info:
        pv = entry.get("pv")
        if not pv:
            continue
        lines.append((pv[0], _score_to_cp(entry["score"])))
    return lines


def analyze_move(engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                  limit: chess.engine.Limit, multipv: int = 3) -> MoveAnalysis:
    """Analyze the position before `move` is played: the engine's top
    candidates, and how `move` itself compares - even when it isn't
    one of those candidates. Does not mutate `board` past its original
    state (pushes/pops internally as needed)."""
    lines = analyze_position(engine, board, limit, multipv)
    played_score = next((score for candidate, score in lines if candidate == move), None)

    if played_score is None:
        # Played move wasn't among the top `multipv` lines - evaluate it
        # directly: the position after it is scored from the opponent's
        # perspective, so negate to get it back to the mover's.
        board.push(move)
        reply_lines = analyze_position(engine, board, limit, multipv=1)
        board.pop()
        played_score = -reply_lines[0][1] if reply_lines else None

    played_san = board.san(move)
    candidate_lines = [(board.san(candidate), score) for candidate, score in lines]
    best_move, best_score = candidate_lines[0] if candidate_lines else (None, None)

    return MoveAnalysis(
        move_number=board.fullmove_number, color=board.turn, played_move=played_san,
        played_score_cp=played_score, best_move=best_move, best_score_cp=best_score,
        candidate_lines=candidate_lines,
    )


def analyze_game(engine: chess.engine.SimpleEngine, game: chess.pgn.Game,
                  limit: chess.engine.Limit, multipv: int = 3) -> GameAnalysis:
    """Replay `game`'s mainline and analyze every move in order."""
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        moves.append(analyze_move(engine, board, move, limit, multipv))
        board.push(move)

    # python-chess's default Headers pre-fills every Seven Tag Roster
    # field (including Event) with the PGN "unknown value" placeholder
    # "?" rather than leaving it empty - so "?" has to be treated as
    # not-really-set here too, not just an empty/missing key.
    event = game.headers.get("Event", "?")
    if event and event != "?":
        game_id = event
    else:
        white = game.headers.get("White", "?")
        black = game.headers.get("Black", "?")
        game_id = f"{white} vs {black}"
    return GameAnalysis(game_id=game_id, moves=moves)
