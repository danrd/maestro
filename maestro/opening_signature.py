"""Group games by their own move sequence rather than a fixed opening
classification - works even for lines the curated reference (see
opening_reference.py) or the vendored lichess dataset doesn't cover,
since a signature is just "moves this player's games actually share",
not a lookup into anything external.

Uses the full move sequence (both colors) - a specific opening line is
co-defined by the reply, not just the tracked player's own moves (the
Sicilian only exists because Black played ...c5; White's own first move
alone doesn't distinguish it from any other 1.e4 game).

Signature depth is adaptive and measured in full moves per side (same
unit as player_profile.py's OPENING_CUTOFF_MOVE, and the same boundary
- 5 full moves - since a signature is meant to describe exactly how a
player gets through the opening as that boundary defines it): try 5
first, shrinking to 4 then 3 only if not enough *other* games share
that longer prefix. A game whose prefix never reaches `min_games`
support even at 3 full moves gets no signature (None) - reported as
such, not forced into a low-confidence group.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

MAX_SIGNATURE_FULL_MOVES = 5
MIN_SIGNATURE_FULL_MOVES = 3


def _candidate_ply_depths() -> List[int]:
    """[10, 8, 6] - 5, 4, 3 full moves per side, in plies (2 plies per
    full move), longest first."""
    return [full_moves * 2 for full_moves in
            range(MAX_SIGNATURE_FULL_MOVES, MIN_SIGNATURE_FULL_MOVES - 1, -1)]


def _prefix(moves: Sequence[str], ply_depth: int) -> Tuple[str, ...]:
    return tuple(moves[:ply_depth])


def assign_opening_signatures(games_moves: Dict[str, Sequence[str]],
                               min_games: int = 5) -> Dict[str, Optional[Tuple[str, ...]]]:
    """`games_moves` maps some game identifier (e.g. a game hash) to
    that game's ordered SAN moves, both colors interleaved, at least
    MAX_SIGNATURE_FULL_MOVES*2 plies long where available. Returns the
    same keys mapped to a signature (a move-prefix tuple) or None."""
    ply_depths = _candidate_ply_depths()

    prefix_counts: Dict[Tuple[str, ...], int] = Counter()
    for moves in games_moves.values():
        for ply_depth in ply_depths:
            if len(moves) >= ply_depth:
                prefix_counts[_prefix(moves, ply_depth)] += 1

    signatures: Dict[str, Optional[Tuple[str, ...]]] = {}
    for game_id, moves in games_moves.items():
        chosen = None
        for ply_depth in ply_depths:  # longest first
            if len(moves) < ply_depth:
                continue
            prefix = _prefix(moves, ply_depth)
            if prefix_counts[prefix] >= min_games:
                chosen = prefix
                break
        signatures[game_id] = chosen

    return signatures


def group_by_signature(
    signatures: Dict[str, Optional[Tuple[str, ...]]],
) -> Dict[Optional[Tuple[str, ...]], List[str]]:
    """Invert assign_opening_signatures's output: signature -> the game
    ids that share it. The None key collects every game that didn't
    reach minimum support at any depth."""
    groups: Dict[Optional[Tuple[str, ...]], List[str]] = {}
    for game_id, signature in signatures.items():
        groups.setdefault(signature, []).append(game_id)
    return groups


def extract_move_prefix(pgn_text: str,
                         max_ply_depth: int = MAX_SIGNATURE_FULL_MOVES * 2) -> Tuple[str, ...]:
    """Parse the first `max_ply_depth` SAN moves (both colors) out of a
    PGN game's mainline - the raw material assign_opening_signatures
    groups games by."""
    import io

    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return ()

    board = game.board()
    moves = []
    for move in game.mainline_moves():
        if len(moves) >= max_ply_depth:
            break
        moves.append(board.san(move))
        board.push(move)
    return tuple(moves)
