"""Tie storage, the analysis pool, and report-building together:
import whatever's new, analyze only what isn't already cached for these
exact settings, and return every requested game's report either way.
"""
from __future__ import annotations

import io
import sqlite3
from typing import List, Optional

import chess.engine
import chess.pgn

from maestro.analysis_pool import analyze_pgn_games_parallel
from maestro.game_report import GameReport, build_game_report
from maestro.game_store import (
    compute_game_hash,
    compute_params_hash,
    get_cached_report,
    import_games,
    save_report,
)


def analyze_and_cache_games(conn: sqlite3.Connection, pgn_texts: List[str], stockfish_path: str,
                             limit: chess.engine.Limit, multipv: int = 3,
                             mistake_threshold_cp: int = 50, safe_alternatives_cap: int = 5,
                             opening_ply_cutoff: int = 10, player_name: Optional[str] = None,
                             num_workers: Optional[int] = None) -> List[GameReport]:
    """Import any of `pgn_texts` not already stored, then return every
    one of them as a GameReport - reusing a cached report wherever one
    already exists for this exact combination of engine/analysis
    settings, and only running Stockfish on what's actually new (or
    was previously analyzed with different settings).
    """
    params_hash = compute_params_hash(
        multipv=multipv, mistake_threshold_cp=mistake_threshold_cp,
        safe_alternatives_cap=safe_alternatives_cap, opening_ply_cutoff=opening_ply_cutoff,
        depth=limit.depth, time=limit.time, nodes=limit.nodes,
    )
    import_games(conn, pgn_texts)

    game_hashes = [compute_game_hash(text) for text in pgn_texts]
    reports_by_hash = {}
    to_analyze_hashes: List[str] = []
    to_analyze_texts: List[str] = []

    for game_hash, text in zip(game_hashes, pgn_texts):
        cached = get_cached_report(conn, game_hash, params_hash)
        if cached is not None:
            reports_by_hash[game_hash] = cached
        elif game_hash not in reports_by_hash and game_hash not in to_analyze_hashes:
            to_analyze_hashes.append(game_hash)
            to_analyze_texts.append(text)

    if to_analyze_texts:
        analyses = analyze_pgn_games_parallel(
            to_analyze_texts, stockfish_path, limit, multipv, num_workers,
            mistake_threshold_cp, safe_alternatives_cap, opening_ply_cutoff,
        )
        for game_hash, text, analysis in zip(to_analyze_hashes, to_analyze_texts, analyses):
            game = chess.pgn.read_game(io.StringIO(text))
            report = build_game_report(analysis, game, mistake_threshold_cp, player_name)
            save_report(conn, game_hash, params_hash, report)
            reports_by_hash[game_hash] = report

    return [reports_by_hash[game_hash] for game_hash in game_hashes]
