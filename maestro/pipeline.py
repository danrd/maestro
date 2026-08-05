"""Tie storage, the analysis pool, report-building, and coaching
feedback together: import whatever's new, analyze (and coach) only
what isn't already cached for these exact settings, and return every
requested game's result either way.
"""
from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import chess.engine
import chess.pgn

from maestro.analysis_pool import analyze_pgn_games_parallel
from maestro.coaching import CoachingConfig, generate_coaching_feedback
from maestro.game_report import GameReport, build_game_report
from maestro.game_store import (
    compute_game_hash,
    compute_params_hash,
    get_cached_feedback,
    get_cached_report,
    get_pgn,
    import_games,
    save_feedback,
    save_report,
)
from maestro.opening_profile import OpeningGroupSummary, build_opening_groups
from maestro.opening_signature import extract_move_prefix
from maestro.player_profile import PlayerProfile, build_player_profile
from maestro.profile_coaching import ProfileCoachingConfig, generate_profile_coaching_feedback


def _analysis_params_hash(multipv: int, mistake_threshold_cp: int, safe_alternatives_cap: int,
                           opening_ply_cutoff: int, limit: chess.engine.Limit) -> str:
    """Shared by analyze_and_cache_games and analyze_and_coach_games so
    the two never drift into computing this differently."""
    return compute_params_hash(
        multipv=multipv, mistake_threshold_cp=mistake_threshold_cp,
        safe_alternatives_cap=safe_alternatives_cap, opening_ply_cutoff=opening_ply_cutoff,
        depth=limit.depth, time=limit.time, nodes=limit.nodes,
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
    params_hash = _analysis_params_hash(multipv, mistake_threshold_cp, safe_alternatives_cap,
                                         opening_ply_cutoff, limit)
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


@dataclass
class CoachedGame:
    report: GameReport
    feedback: Optional[str]  # None if the coaching prompt didn't fit its token_limit


def analyze_and_coach_games(conn: sqlite3.Connection, pgn_texts: List[str], stockfish_path: str,
                             limit: chess.engine.Limit, tokenizer, runner,
                             multipv: int = 3, mistake_threshold_cp: int = 50,
                             safe_alternatives_cap: int = 5, opening_ply_cutoff: int = 10,
                             player_name: Optional[str] = None, num_workers: Optional[int] = None,
                             coaching_config: Optional[CoachingConfig] = None) -> List[CoachedGame]:
    """analyze_and_cache_games(), then generate coaching feedback for
    each report - the same "don't recompute what's already been done
    for these exact settings" discipline extended to the LLM step,
    which costs a real call and isn't free either. A report whose
    feedback was already generated for this exact combination of
    analysis + coaching settings is a cache hit; everything else calls
    `runner` once per game.
    """
    coaching_config = coaching_config or CoachingConfig()
    reports = analyze_and_cache_games(
        conn, pgn_texts, stockfish_path, limit, multipv, mistake_threshold_cp,
        safe_alternatives_cap, opening_ply_cutoff, player_name, num_workers,
    )

    analysis_hash = _analysis_params_hash(multipv, mistake_threshold_cp, safe_alternatives_cap,
                                           opening_ply_cutoff, limit)
    feedback_params_hash = compute_params_hash(
        analysis_params=analysis_hash, max_mistakes=coaching_config.max_mistakes,
        language=coaching_config.language, token_limit=coaching_config.token_limit,
    )

    results = []
    for text, report in zip(pgn_texts, reports):
        game_hash = compute_game_hash(text)
        cached_feedback = get_cached_feedback(conn, game_hash, feedback_params_hash)
        if cached_feedback is not None:
            results.append(CoachedGame(report=report, feedback=cached_feedback))
            continue

        feedback = generate_coaching_feedback(report, tokenizer, runner, coaching_config)
        if feedback is not None:
            save_feedback(conn, game_hash, feedback_params_hash, feedback)
        results.append(CoachedGame(report=report, feedback=feedback))

    return results


@dataclass
class ProfileCoachingResult:
    profile: PlayerProfile
    opening_groups: List[OpeningGroupSummary]
    feedback: Optional[str]  # None if the coaching prompt didn't fit its token_limit


def analyze_and_coach_profile(conn: sqlite3.Connection, pgn_texts: List[str], stockfish_path: str,
                               limit: chess.engine.Limit, tokenizer, runner,
                               multipv: int = 3, mistake_threshold_cp: int = 50,
                               safe_alternatives_cap: int = 5, opening_ply_cutoff: int = 10,
                               player_name: Optional[str] = None, num_workers: Optional[int] = None,
                               profile_min_games: int = 5,
                               coaching_config: Optional[ProfileCoachingConfig] = None) -> ProfileCoachingResult:
    """Analyze (or reuse cached reports for) every game in `pgn_texts`,
    aggregate them into a PlayerProfile (phase-of-game patterns, see
    player_profile.py) and opening groups (recurring lines, see
    opening_profile.py), then generate cross-game coaching feedback -
    reusing a cached feedback text if one already exists for this exact
    combination of game set + settings.

    `player_name` is required in practice: without it, reports have no
    `player_color`, and player_profile.py / opening_profile.py can't
    attribute any mistake to the tracked player at all.
    """
    coaching_config = coaching_config or ProfileCoachingConfig()
    reports = analyze_and_cache_games(
        conn, pgn_texts, stockfish_path, limit, multipv, mistake_threshold_cp,
        safe_alternatives_cap, opening_ply_cutoff, player_name, num_workers,
    )

    game_hashes = [compute_game_hash(text) for text in pgn_texts]
    reports_by_hash = dict(zip(game_hashes, reports))
    moves_by_hash = {
        game_hash: extract_move_prefix(get_pgn(conn, game_hash) or text)
        for game_hash, text in zip(game_hashes, pgn_texts)
    }

    profile = build_player_profile(reports, min_games=profile_min_games)
    opening_groups = build_opening_groups(reports_by_hash, moves_by_hash, min_games=profile_min_games)

    analysis_hash = _analysis_params_hash(multipv, mistake_threshold_cp, safe_alternatives_cap,
                                           opening_ply_cutoff, limit)
    profile_params_hash = compute_params_hash(
        game_hashes=sorted(game_hashes), analysis_params=analysis_hash,
        profile_min_games=profile_min_games, max_phase_buckets=coaching_config.max_phase_buckets,
        max_openings=coaching_config.max_openings, language=coaching_config.language,
        token_limit=coaching_config.token_limit,
    )
    # Not a real per-game hash - the feedback table's schema doesn't
    # care what the key means, and reusing it here avoids a whole extra
    # table just for this one aggregate-level cache entry. Prefixed so
    # it can never collide with an actual game hash (a hex sha256).
    profile_cache_key = "profile:" + profile_params_hash

    cached_feedback = get_cached_feedback(conn, profile_cache_key, profile_params_hash)
    if cached_feedback is not None:
        return ProfileCoachingResult(profile=profile, opening_groups=opening_groups, feedback=cached_feedback)

    feedback = generate_profile_coaching_feedback(profile, opening_groups, tokenizer, runner, coaching_config)
    if feedback is not None:
        save_feedback(conn, profile_cache_key, profile_params_hash, feedback)

    return ProfileCoachingResult(profile=profile, opening_groups=opening_groups, feedback=feedback)
