"""Combine opening_signature.py's move-prefix grouping with
opening_reference.py's name/idea lookups and each group's own mistake
statistics - the opening-specific counterpart to player_profile.py's
phase-of-game aggregation. Pure derivation, no engine or LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from maestro.game_report import GameReport
from maestro.opening_reference import lookup_opening_idea, lookup_opening_name
from maestro.opening_signature import assign_opening_signatures, group_by_signature


@dataclass
class OpeningGroupSummary:
    signature: Tuple[str, ...]
    name: Optional[str]
    idea: Optional[str]
    game_count: int
    mistake_count: int
    avg_loss_cp: float


def build_opening_groups(reports_by_hash: Dict[str, GameReport],
                          moves_by_hash: Dict[str, Sequence[str]],
                          min_games: int = 5) -> List[OpeningGroupSummary]:
    """`reports_by_hash` and `moves_by_hash` must share the same keys
    (game hashes): reports supply the mistake statistics, moves (see
    opening_signature.extract_move_prefix) supply what the games are
    grouped by. Groups that never reached `min_games` support (signature
    None - see opening_signature.py) are excluded, same "not enough
    data" rule used throughout this pipeline.
    """
    signatures = assign_opening_signatures(moves_by_hash, min_games=min_games)
    groups = group_by_signature(signatures)

    summaries = []
    for signature, game_hashes in groups.items():
        if signature is None:
            continue
        reports = [reports_by_hash[h] for h in game_hashes if h in reports_by_hash]
        if len(reports) < min_games:
            continue

        mistakes = [
            mistake for report in reports for mistake in report.mistakes
            if report.player_color is not None and mistake.color == report.player_color
        ]
        info = lookup_opening_name(signature)

        summaries.append(OpeningGroupSummary(
            signature=signature, name=info.name if info else None,
            idea=lookup_opening_idea(signature), game_count=len(reports),
            mistake_count=len(mistakes),
            avg_loss_cp=(sum(m.loss_cp for m in mistakes) / len(mistakes)) if mistakes else 0.0,
        ))

    return summaries
