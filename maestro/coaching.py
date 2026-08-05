"""Turn a GameReport into personalized coaching feedback via an LLM.

chess_analysis.py/game_report.py already answer WHAT went wrong -
loss_cp, and safe_alternatives distinguishing an unforced error (many
safe moves existed) from missing the one move that actually mattered.
This module turns that structured data into readable prose that uses
that distinction, rather than another dump of numbers.

No resolvers needed here (unlike lector's knowledge_base retrieval) -
everything a mistake block needs is already computed and handed to the
prompt as plain context, so this is ordinary Jinja2 templating.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from llm_kit.prompt_builder import PromptBuilder, PromptingConfig
from maestro.game_report import GameReport, Mistake

PROMPTS_DIR = str(Path(__file__).parent / "prompts")
COACHING_BLOCKS = ["role_instruction", "game_context", "mistakes", "output_format"]


class CoachingConfig(BaseModel):
    """Config for generating feedback on one game's report."""
    model_config = ConfigDict(extra="forbid")

    blocks_dir: str = PROMPTS_DIR
    token_limit: int = Field(default=8000, ge=1)
    language: Optional[str] = None
    # report.mistakes is already sorted worst-first (see game_report.py) -
    # only the top max_mistakes are actually discussed, both to bound the
    # prompt's size and because a long tail of minor mistakes isn't
    # useful to walk through one by one.
    max_mistakes: int = Field(default=5, ge=1)


def _color_name(color: Optional[bool]) -> str:
    if color is None:
        return "unknown"
    return "White" if color else "Black"


def _format_mistake(mistake: Mistake, index: int) -> str:
    text = (
        f"{index}. Move {mistake.move_number} ({_color_name(mistake.color)}): "
        f"played {mistake.played_move}"
    )
    if mistake.best_move:
        text += f", engine's best was {mistake.best_move}"
    text += f" - lost {mistake.loss_cp / 100:.1f} pawns of evaluation."

    if mistake.safe_alternatives is not None:
        if mistake.safe_alternatives == 0:
            text += " This looks like a narrow, forced moment - few or no other safe moves existed."
        else:
            text += (
                f" At least {mistake.safe_alternatives} other safe move(s) were available here - "
                "this looks like an unforced error."
            )
    if mistake.move_time_seconds is not None:
        text += f" Time spent on this move: {mistake.move_time_seconds:.0f}s."
    return text


def _format_mistakes(report: GameReport, max_mistakes: int) -> str:
    if not report.mistakes:
        return "No mistakes above the flagged threshold were found in this game."

    shown = report.mistakes[:max_mistakes]
    lines = [_format_mistake(mistake, i + 1) for i, mistake in enumerate(shown)]
    remaining = len(report.mistakes) - len(shown)
    if remaining > 0:
        lines.append(f"... and {remaining} more mistake(s), not detailed here.")
    return "\n".join(lines)


def build_coaching_prompt(report: GameReport, tokenizer,
                           config: Optional[CoachingConfig] = None) -> Optional[str]:
    """Build (but don't generate) the coaching prompt for `report`.
    Returns None if the prompt doesn't fit token_limit (same convention
    as PromptBuilder.build)."""
    config = config or CoachingConfig()
    prompting_config = PromptingConfig(
        blocks_dir=config.blocks_dir, blocks=COACHING_BLOCKS, token_limit=config.token_limit,
    )
    builder = PromptBuilder(prompting_config, tokenizer)
    context = {
        "opening": report.opening or "unknown",
        "player_color": _color_name(report.player_color),
        "total_moves": report.total_moves,
        "mistakes_text": _format_mistakes(report, config.max_mistakes),
        "language": config.language,
    }
    return builder.build(task=report.game_id, context=context)


def generate_coaching_feedback(report: GameReport, tokenizer, runner,
                                config: Optional[CoachingConfig] = None) -> Optional[str]:
    """Build the prompt and generate feedback via `runner`
    (runner.generate(prompt) -> str - any llm_kit Runner). Returns None
    without calling the runner if the prompt itself didn't fit
    token_limit."""
    prompt = build_coaching_prompt(report, tokenizer, config)
    if prompt is None:
        return None
    return runner.generate(prompt)
