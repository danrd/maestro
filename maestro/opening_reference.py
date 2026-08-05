"""Classify a game's opening move sequence against known names, and -
for a bounded set of well-known top-level systems - a short description
of the strategic idea behind it.

Two separate sources, looked up independently:

- **Classification** (ECO code + name): vendored from
  lichess-org/chess-openings (CC0 / public domain - maestro/data/*.tsv),
  ~3800 entries. Broad, accurate coverage of *names*, nothing about the
  underlying idea.
- **Ideas**: no open dataset we found actually explains the strategic
  idea behind an opening in prose, only classifies it - so this part is
  hand-curated, deliberately bounded to well-known top-level systems
  (established chess theory, not sourced from anywhere specific).
  Everything outside that bounded set gets no idea text - tracked as a
  plain opening_signature (see opening_signature.py) with statistics
  only, filled in by hand later as cases accumulate, not auto-generated.

Both lookups walk from the longest matching move-prefix down to the
shortest, so a specific line still resolves to its parent system's name
or idea if nothing more specific is known.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

DATA_DIR = Path(__file__).parent / "data"
_MOVE_NUMBER_RE = re.compile(r"^\d+\.+$")


@dataclass
class OpeningInfo:
    eco: str
    name: str


def _parse_pgn_moves(pgn: str) -> Tuple[str, ...]:
    """'1. e4 e6 2. d4 d5' -> ('e4', 'e6', 'd4', 'd5')."""
    return tuple(token for token in pgn.split() if not _MOVE_NUMBER_RE.match(token))


def _load_classification() -> Dict[Tuple[str, ...], OpeningInfo]:
    index: Dict[Tuple[str, ...], OpeningInfo] = {}
    for letter in "abcde":
        path = DATA_DIR / f"{letter}.tsv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                moves = _parse_pgn_moves(row["pgn"])
                index[moves] = OpeningInfo(eco=row["eco"], name=row["name"])
    return index


_classification_index: Optional[Dict[Tuple[str, ...], OpeningInfo]] = None


def _get_classification_index() -> Dict[Tuple[str, ...], OpeningInfo]:
    global _classification_index
    if _classification_index is None:
        _classification_index = _load_classification()
    return _classification_index


def lookup_opening_name(moves: Sequence[str]) -> Optional[OpeningInfo]:
    """Longest-prefix match of `moves` (SAN, in order, both colors)
    against the vendored classification data."""
    index = _get_classification_index()
    moves = tuple(moves)
    for length in range(len(moves), 0, -1):
        info = index.get(moves[:length])
        if info is not None:
            return info
    return None


# Hand-curated strategic ideas for a bounded set of well-known top-level
# systems - see this module's docstring for why. Keys are SAN move
# prefixes (both colors); lookup_opening_idea walks from the longest
# match down to the first move.
_IDEAS: Dict[Tuple[str, ...], str] = {
    ("e4",):
        "Stakes an immediate claim on the center and opens lines for the queen and "
        "light-squared bishop - leads to open, often tactical positions.",
    ("e4", "e5"):
        "Symmetric reply claiming equal central space - the classical open games "
        "(Italian, Ruy Lopez, Scotch, ...) grow from here.",
    ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        "Ruy Lopez (Spanish) - White pressures the knight defending e5 and builds "
        "long-term central and kingside pressure rather than going for immediate tactics.",
    ("e4", "e5", "Nf3", "Nc6", "Bc4"):
        "Italian Game - White develops quickly and aims the bishop at f7, favoring "
        "rapid development over immediate material or structural gains.",
    ("e4", "e5", "Nf3", "Nc6", "d4"):
        "Scotch Game - White opens the center immediately, trading a pair of pawns "
        "for faster piece activity and open lines.",
    ("e4", "c5"):
        "Sicilian Defense - Black immediately unbalances the position, fighting for "
        "d4 from the flank and accepting an asymmetric, sharp game in exchange for "
        "genuine winning chances rather than early equality.",
    ("e4", "e6"):
        "French Defense - Black concedes some central space temporarily for a solid "
        "pawn structure, planning ...c5 or ...f6 breaks once developed.",
    ("e4", "c6"):
        "Caro-Kann Defense - similar solidity to the French, but keeps the "
        "light-squared bishop active before the center locks up.",
    ("e4", "d5"):
        "Scandinavian Defense - Black challenges the center immediately, accepting "
        "an early queen excursion (2.exd5 Qxd5) in exchange for simple, fast development.",
    ("e4", "Nf6"):
        "Alekhine's Defense - Black lures White's pawns forward, planning to "
        "undermine the resulting overextended center later.",
    ("d4",):
        "Fights for the center more slowly than 1.e4, prioritizing a solid pawn "
        "structure and flexible piece development over immediate tactics.",
    ("d4", "d5"):
        "Classical Queen's Pawn symmetric reply - leads to Queen's Gambit-type "
        "structures and central pawn tension.",
    ("d4", "d5", "c4"):
        "Queen's Gambit - White offers the c-pawn to lure Black's d-pawn away from "
        "the center, or to gain a tempo reclaiming it later.",
    ("d4", "Nf6"):
        "Indian Defenses - Black delays committing the central pawns, often "
        "fianchettoing a bishop to pressure the center from the flank instead of "
        "occupying it directly.",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7"):
        "King's Indian Defense - Black concedes the center temporarily, then "
        "counterattacks it later with ...e5 or ...c5 once castled and developed.",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"):
        "Nimzo-Indian Defense - Black pressures c3 and is willing to trade the "
        "bishop for a knight to saddle White with doubled pawns and a static weakness.",
    ("d4", "Nf6", "c4", "g6", "Nc3", "d5"):
        "Grünfeld Defense - Black lets White build a big pawn center, then strikes "
        "at it immediately with pieces rather than matching it with pawns.",
    ("c4",):
        "English Opening - a flexible flank opening, often transposing into other "
        "systems; delays committing the central pawns.",
    ("Nf3",):
        "Réti Opening - a hypermodern approach, aiming to control the center with "
        "pieces from the flanks rather than occupying it with pawns immediately.",
}


def lookup_opening_idea(moves: Sequence[str]) -> Optional[str]:
    """Longest-prefix match of `moves` against the curated idea set.
    Falls back through progressively shorter prefixes down to the very
    first move; if even that isn't curated, returns None rather than
    guessing at one."""
    moves = tuple(moves)
    for length in range(len(moves), 0, -1):
        idea = _IDEAS.get(moves[:length])
        if idea is not None:
            return idea
    return None
