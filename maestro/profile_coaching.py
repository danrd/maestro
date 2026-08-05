"""Turn a PlayerProfile + opening group summaries into cross-game
coaching feedback via an LLM - the aggregate counterpart to
coaching.py's single-game feedback. Same split as everywhere else in
this pipeline: Python already computed every number (player_profile.py,
opening_profile.py); the model turns that into readable, prioritized
advice, not another table of numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from llm_kit.prompt_builder import PromptBuilder, PromptingConfig
from maestro.opening_profile import OpeningGroupSummary
from maestro.player_profile import PlayerProfile

PROMPTS_DIR = str(Path(__file__).parent / "prompts")
PROFILE_COACHING_BLOCKS = [
    "profile_role_instruction", "phase_stats", "opening_notes", "profile_output_format",
]


class ProfileCoachingConfig(BaseModel):
    """Config for generating feedback across many games' reports."""
    model_config = ConfigDict(extra="forbid")

    blocks_dir: str = PROMPTS_DIR
    token_limit: int = Field(default=8000, ge=1)
    language: Optional[str] = None
    # Both lists are already sorted worst-first by the formatters below;
    # these just bound how many of each actually reach the prompt.
    max_phase_buckets: int = Field(default=6, ge=1)
    max_openings: int = Field(default=5, ge=1)


def _color_name(color: bool) -> str:
    return "White" if color else "Black"


def _format_phase_stats(profile: PlayerProfile, max_buckets: int) -> str:
    if not profile.phase_stats:
        return "Not enough games analyzed yet to report phase-of-game patterns."

    ranked = sorted(profile.phase_stats, key=lambda stat: stat.avg_loss_cp, reverse=True)
    lines = []
    for stat in ranked[:max_buckets]:
        line = (
            f"- Moves {stat.phase} as {_color_name(stat.color)}: {stat.mistake_count} mistakes "
            f"across {stat.games_with_data} games, average loss {stat.avg_loss_cp / 100:.2f} pawns"
        )
        if stat.unforced_count or stat.forced_count:
            line += f" ({stat.unforced_count} unforced, {stat.forced_count} forced/narrow)"
        if stat.avg_move_time_seconds is not None:
            line += f", average {stat.avg_move_time_seconds:.0f}s spent on these moves"
        lines.append(line)
    return "\n".join(lines)


def _format_opening_notes(groups: List[OpeningGroupSummary], max_openings: int) -> str:
    if not groups:
        return "Not enough games analyzed yet to report opening-specific patterns."

    ranked = sorted(groups, key=lambda group: group.avg_loss_cp, reverse=True)
    lines = []
    for group in ranked[:max_openings]:
        label = group.name or " ".join(group.signature)
        line = (
            f"- {label}: {group.mistake_count} mistakes across {group.game_count} games, "
            f"average loss {group.avg_loss_cp / 100:.2f} pawns"
        )
        if group.idea:
            line += f". Reference idea: {group.idea}"
        lines.append(line)
    return "\n".join(lines)


def build_profile_coaching_prompt(profile: PlayerProfile, opening_groups: List[OpeningGroupSummary],
                                   tokenizer, config: Optional[ProfileCoachingConfig] = None) -> Optional[str]:
    """Build (but don't generate) the cross-game coaching prompt.
    Returns None if it doesn't fit token_limit (same convention as
    PromptBuilder.build)."""
    config = config or ProfileCoachingConfig()
    prompting_config = PromptingConfig(
        blocks_dir=config.blocks_dir, blocks=PROFILE_COACHING_BLOCKS, token_limit=config.token_limit,
    )
    builder = PromptBuilder(prompting_config, tokenizer)
    context = {
        "total_games": profile.total_games,
        "phase_stats_text": _format_phase_stats(profile, config.max_phase_buckets),
        "opening_notes_text": _format_opening_notes(opening_groups, config.max_openings),
        "language": config.language,
    }
    return builder.build(task="player-profile", context=context)


def generate_profile_coaching_feedback(profile: PlayerProfile, opening_groups: List[OpeningGroupSummary],
                                        tokenizer, runner,
                                        config: Optional[ProfileCoachingConfig] = None) -> Optional[str]:
    """Build the prompt and generate feedback via `runner`. Returns
    None without calling the runner if the prompt didn't fit
    token_limit."""
    prompt = build_profile_coaching_prompt(profile, opening_groups, tokenizer, config)
    if prompt is None:
        return None
    return runner.generate(prompt)
