"""Tests for maestro/opening_reference.py.

Real vendored data (maestro/data/*.tsv) and the real curated idea dict -
these are exactly the two data sources under test, so no reason to fake
either. Lines checked below were verified against the actual TSV files
before being written into _IDEAS.
"""
from __future__ import annotations

from maestro.opening_reference import lookup_opening_idea, lookup_opening_name


# -- lookup_opening_name (vendored classification) -----------------------

def test_lookup_opening_name_finds_an_exact_match():
    info = lookup_opening_name(["e4", "c5"])

    assert info is not None
    assert info.name == "Sicilian Defense"
    assert info.eco == "B20"


def test_lookup_opening_name_falls_back_to_the_longest_known_prefix():
    # a made-up continuation past a real, well-known line
    info = lookup_opening_name(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a9"])

    assert info is not None
    assert "Sicilian" in info.name


def test_lookup_opening_name_returns_none_for_an_unrecognized_first_move():
    assert lookup_opening_name(["not-a-real-move"]) is None


def test_lookup_opening_name_returns_none_for_empty_input():
    assert lookup_opening_name([]) is None


# -- lookup_opening_idea (curated, bounded set) ---------------------------

def test_lookup_opening_idea_matches_a_curated_top_level_system():
    idea = lookup_opening_idea(["e4", "c5"])

    assert idea is not None
    assert "Sicilian" in idea


def test_lookup_opening_idea_falls_back_to_a_shorter_curated_prefix():
    # deep into a line beyond what's curated for the Sicilian specifically
    idea = lookup_opening_idea(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"])

    assert idea is not None
    assert "Sicilian" in idea


def test_lookup_opening_idea_falls_back_all_the_way_to_the_first_move():
    idea = lookup_opening_idea(["e4", "totally-uncurated-continuation"])

    assert idea is not None
    assert "center" in idea.lower()


def test_lookup_opening_idea_returns_none_outside_the_curated_set():
    assert lookup_opening_idea(["a4"]) is None  # not a curated top-level system


def test_lookup_opening_idea_returns_none_for_empty_input():
    assert lookup_opening_idea([]) is None
