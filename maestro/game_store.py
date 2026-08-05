"""SQLite-backed storage for imported games and their computed reports,
so re-running analysis on a game already processed with the same
settings is a cache hit, not a re-run through Stockfish.

Two tables: `games` (raw PGN text, deduplicated by a hash of the
parsed-and-re-rendered game - not the raw bytes, so incidental
whitespace differences between sources don't create duplicates) and
`reports` (a GameReport as JSON, keyed by game hash + a hash of the
analysis parameters that produced it, so changing depth/multipv/engine
invalidates the cache instead of silently serving a stale result under
different settings).
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from dataclasses import asdict
from typing import Any, List, Optional

import chess.pgn

from maestro.game_report import GameReport, Mistake

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_hash TEXT PRIMARY KEY,
    pgn_text TEXT NOT NULL,
    white TEXT,
    black TEXT,
    date TEXT,
    imported_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reports (
    game_hash TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (game_hash, params_hash)
);
"""


def compute_game_hash(pgn_text: str) -> str:
    """Stable identity for a game, independent of incidental
    formatting differences between sources - hashes the
    parsed-and-re-rendered PGN (headers + moves), not the raw text
    byte-for-byte."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    canonical = str(game) if game is not None else pgn_text
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_params_hash(**params: Any) -> str:
    """Stable identity for a set of analysis parameters (depth, multipv,
    mistake_threshold_cp, engine path/version, ...) - part of the
    report cache key so different settings never collide under the
    same game hash."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def open_store(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def import_games(conn: sqlite3.Connection, pgn_texts: List[str]) -> List[str]:
    """Insert every game in `pgn_texts` not already present (by hash) -
    "just pull in what's new". Returns the hashes of the newly-inserted
    games only, so a caller doing a bulk incremental import knows
    exactly what still needs analyzing without re-scanning the whole
    table for what changed."""
    new_hashes = []
    for text in pgn_texts:
        game_hash = compute_game_hash(text)
        already_present = conn.execute(
            "SELECT 1 FROM games WHERE game_hash = ?", (game_hash,)
        ).fetchone()
        if already_present:
            continue

        game = chess.pgn.read_game(io.StringIO(text))
        white = game.headers.get("White") if game else None
        black = game.headers.get("Black") if game else None
        date = game.headers.get("Date") if game else None
        conn.execute(
            "INSERT INTO games (game_hash, pgn_text, white, black, date) VALUES (?, ?, ?, ?, ?)",
            (game_hash, text, white, black, date),
        )
        new_hashes.append(game_hash)

    conn.commit()
    return new_hashes


def get_pgn(conn: sqlite3.Connection, game_hash: str) -> Optional[str]:
    row = conn.execute("SELECT pgn_text FROM games WHERE game_hash = ?", (game_hash,)).fetchone()
    return row[0] if row else None


def get_all_game_hashes(conn: sqlite3.Connection) -> List[str]:
    return [row[0] for row in conn.execute("SELECT game_hash FROM games")]


def get_cached_report(conn: sqlite3.Connection, game_hash: str, params_hash: str) -> Optional[GameReport]:
    row = conn.execute(
        "SELECT report_json FROM reports WHERE game_hash = ? AND params_hash = ?",
        (game_hash, params_hash),
    ).fetchone()
    return _report_from_json(row[0]) if row else None


def save_report(conn: sqlite3.Connection, game_hash: str, params_hash: str, report: GameReport) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO reports (game_hash, params_hash, report_json) VALUES (?, ?, ?)",
        (game_hash, params_hash, _report_to_json(report)),
    )
    conn.commit()


def _report_to_json(report: GameReport) -> str:
    # chess.Color is a plain bool (True = white), so it round-trips
    # through JSON as-is - nothing here needs custom (de)serialization.
    return json.dumps(asdict(report))


def _report_from_json(text: str) -> GameReport:
    payload = json.loads(text)
    mistakes = [Mistake(**m) for m in payload.pop("mistakes")]
    return GameReport(mistakes=mistakes, **payload)
