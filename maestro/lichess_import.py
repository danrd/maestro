"""Import a Lichess user's games directly from the Lichess API - an
alternative to loading a manually-exported PGN file (both feed into the
same place: analysis_pool.split_pgn_games / game_store.import_games).

clocks/evals/opening default on: clocks and opening are what
game_report.py already knows how to use (move timing, PGN ECO/Opening
tags instead of building our own opening-book classifier), and evals
are Lichess's own stored Stockfish analysis, included whenever a game
has one - a free skip on recomputation for that game when present,
though most games don't have one (only ones actually analyzed on
Lichess), so pipeline.py's own caching still does most of the work.

Uses stdlib urllib rather than adding an HTTP client dependency - this
is one GET request with query params, nothing more is needed. `opener`
is injectable (defaults to urllib.request.urlopen) so request
construction can be tested without a real network call.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from typing import List, Optional

from maestro.analysis_pool import split_pgn_games

LICHESS_GAMES_URL = "https://lichess.org/api/games/user"


def fetch_user_games_pgn(username: str, *, max_games: Optional[int] = None,
                          since: Optional[int] = None, until: Optional[int] = None,
                          clocks: bool = True, evals: bool = True, opening: bool = True,
                          api_token: Optional[str] = None, timeout: float = 60.0,
                          opener=urllib.request.urlopen) -> str:
    """Fetch `username`'s games as one PGN blob (multiple games
    concatenated, in Lichess's export order) - split it with
    analysis_pool.split_pgn_games before handing it to the rest of the
    pipeline.

    `since`/`until` are Unix timestamps in milliseconds (Lichess's own
    convention). No `api_token` is required for a public profile's
    public games - pass one to raise rate limits or reach private games.
    """
    params = {
        "clocks": str(clocks).lower(), "evals": str(evals).lower(), "opening": str(opening).lower(),
        "pgnInJson": "false",
    }
    if max_games is not None:
        params["max"] = str(max_games)
    if since is not None:
        params["since"] = str(since)
    if until is not None:
        params["until"] = str(until)

    # quote()'s default safe="/" leaves "/" unescaped - fine for a path
    # segment that's supposed to contain one, wrong here where the
    # username itself is the whole segment and a literal "/" in it
    # would otherwise be read as a path separator.
    url = f"{LICHESS_GAMES_URL}/{urllib.parse.quote(username, safe='')}?{urllib.parse.urlencode(params)}"
    headers = {"Accept": "application/x-chess-pgn"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def import_user_games(username: str, **fetch_kwargs) -> List[str]:
    """Fetch and split `username`'s games in one call - ready to pass
    straight to game_store.import_games. `fetch_kwargs` are forwarded to
    fetch_user_games_pgn (max_games, since, until, api_token, ...)."""
    pgn_text = fetch_user_games_pgn(username, **fetch_kwargs)
    return split_pgn_games(pgn_text)
