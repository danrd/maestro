"""Tests for maestro/lichess_import.py.

No real network call - a fake opener (matching urllib.request.urlopen's
call signature and context-manager protocol) stands in, so what's under
test is request construction (URL, query params, headers) and response
handling, not Lichess's actual API.
"""
from __future__ import annotations

from maestro.lichess_import import fetch_user_games_pgn, import_user_games

SAMPLE_PGN = """[Event "Test"]
[White "A"]
[Black "B"]

1. e4 e5 1-0

[Event "Test 2"]
[White "C"]
[Black "D"]

1. d4 d5 1-0
"""


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeOpener:
    def __init__(self, data: bytes):
        self._data = data
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append((request, timeout))
        return _FakeResponse(self._data)


# -- fetch_user_games_pgn -------------------------------------------------

def test_fetch_user_games_pgn_returns_the_decoded_response_body():
    opener = _FakeOpener(SAMPLE_PGN.encode("utf-8"))

    result = fetch_user_games_pgn("magnus", opener=opener)

    assert result == SAMPLE_PGN


def test_fetch_user_games_pgn_url_encodes_the_username():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("weird user/name", opener=opener)

    request = opener.calls[0][0]
    assert "weird%20user%2Fname" in request.full_url


def test_fetch_user_games_pgn_defaults_clocks_evals_opening_to_true():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", opener=opener)

    request = opener.calls[0][0]
    assert "clocks=true" in request.full_url
    assert "evals=true" in request.full_url
    assert "opening=true" in request.full_url


def test_fetch_user_games_pgn_can_disable_any_of_the_flags():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", clocks=False, evals=False, opening=False, opener=opener)

    request = opener.calls[0][0]
    assert "clocks=false" in request.full_url
    assert "evals=false" in request.full_url
    assert "opening=false" in request.full_url


def test_fetch_user_games_pgn_omits_optional_params_by_default():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", opener=opener)

    request = opener.calls[0][0]
    assert "max=" not in request.full_url
    assert "since=" not in request.full_url
    assert "until=" not in request.full_url


def test_fetch_user_games_pgn_includes_max_since_until_when_given():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", max_games=50, since=1000, until=2000, opener=opener)

    request = opener.calls[0][0]
    assert "max=50" in request.full_url
    assert "since=1000" in request.full_url
    assert "until=2000" in request.full_url


def test_fetch_user_games_pgn_sets_no_authorization_header_without_a_token():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", opener=opener)

    request = opener.calls[0][0]
    assert "Authorization" not in request.headers


def test_fetch_user_games_pgn_sets_bearer_authorization_when_a_token_is_given():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", api_token="secret-token", opener=opener)

    request = opener.calls[0][0]
    assert request.headers["Authorization"] == "Bearer secret-token"


def test_fetch_user_games_pgn_passes_the_timeout_through():
    opener = _FakeOpener(b"")

    fetch_user_games_pgn("magnus", timeout=30.0, opener=opener)

    assert opener.calls[0][1] == 30.0


# -- import_user_games ------------------------------------------------------

def test_import_user_games_fetches_and_splits_into_individual_games():
    opener = _FakeOpener(SAMPLE_PGN.encode("utf-8"))

    games = import_user_games("magnus", opener=opener)

    assert len(games) == 2
    assert "e4" in games[0]
    assert "d4" in games[1]
