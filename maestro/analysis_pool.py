"""Analyze many games concurrently: each game is analyzed in its own
worker-process task, which opens its own Stockfish process, analyzes
the game, and closes it again before returning.

Games cross the process boundary as plain PGN text, not chess.pgn.Game
objects: simpler and more robustly picklable than shipping the parsed
game-tree object itself, and re-parsing one game's text in the worker
is cheap next to the engine analysis it's about to do anyway.
"""
from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import chess.engine
import chess.pgn

from maestro.chess_analysis import GameAnalysis, analyze_game


def split_pgn_games(pgn_text: str) -> List[str]:
    """Split one multi-game PGN blob (games concatenated one after
    another, as any PGN database export - a file on disk, a Lichess API
    response - naturally is) into individual per-game PGN text blocks."""
    games = []
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games.append(str(game))
    return games


def load_pgn_games(path: str) -> List[str]:
    """Split a multi-game PGN database file into individual per-game PGN
    text blocks, ready to hand to analyze_pgn_games_parallel."""
    with open(path, encoding="utf-8") as f:
        return split_pgn_games(f.read())


def _analyze_game_text(pgn_text: str, stockfish_path: str, limit: chess.engine.Limit, multipv: int,
                        mistake_threshold_cp: Optional[int], safe_alternatives_cap: int,
                        opening_ply_cutoff: int) -> GameAnalysis:
    """Open a fresh engine for this one game, analyze it, and close the
    engine again before returning.

    Deliberately NOT a persistent engine reused across multiple tasks in
    the same worker process: chess.engine.SimpleEngine keeps a
    non-daemon background thread alive for as long as it's open, and
    letting that thread's lifetime span multiple separate
    ProcessPoolExecutor task calls reliably deadlocked the pool's own
    shutdown in testing (confirmed empirically - a genuine, reproducible
    hang, not a theoretical concern - even when the engine was
    explicitly quit via a dedicated last task per worker instead of
    atexit). Opening and closing the engine within one task call
    sidesteps it entirely, at the cost of one Stockfish process startup
    per game rather than per worker - negligible next to the seconds of
    search time a real game actually takes to analyze.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        return analyze_game(engine, game, limit, multipv,
                             mistake_threshold_cp, safe_alternatives_cap, opening_ply_cutoff)


def analyze_pgn_games_parallel(pgn_texts: List[str], stockfish_path: str,
                                limit: chess.engine.Limit, multipv: int = 3,
                                num_workers: Optional[int] = None,
                                mistake_threshold_cp: Optional[int] = None,
                                safe_alternatives_cap: int = 5,
                                opening_ply_cutoff: int = 10) -> List[GameAnalysis]:
    """Analyze every game in `pgn_texts` (e.g. from load_pgn_games)
    concurrently across a process pool. Results come back in the same
    order as `pgn_texts`, regardless of completion order."""
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_analyze_game_text, text, stockfish_path, limit, multipv,
                        mistake_threshold_cp, safe_alternatives_cap, opening_ply_cutoff)
            for text in pgn_texts
        ]
        return [future.result() for future in futures]
