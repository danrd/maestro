"""Minimal Gradio front-end for maestro: paste PGN, get engine analysis
plus (if an LLM backend is configured) coaching feedback for the pasted
games - and, once enough games are pasted, a cross-game pattern summary
too.

No LLM is required to use this: without OPENROUTER_API_KEY set, it
still runs the full engine analysis and shows the structured statistics
(mistakes, phase/opening patterns) - LLM prose is an enhancement on top
when a key is available, not a hard requirement to try the tool at all.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

import chess.engine
import gradio as gr

from maestro.analysis_pool import split_pgn_games
from maestro.game_report import GameReport
from maestro.game_store import compute_game_hash, open_store
from maestro.opening_profile import OpeningGroupSummary, build_opening_groups
from maestro.opening_signature import extract_move_prefix
from maestro.pipeline import analyze_and_cache_games, analyze_and_coach_games, analyze_and_coach_profile
from maestro.player_profile import PlayerProfile, build_player_profile

DB_PATH = os.environ.get("MAESTRO_DB_PATH", "maestro.db")
# Lower than the library default (5) - a demo session realistically pastes a handful of games,
# not dozens, so the cross-game section would otherwise almost never show anything.
DEFAULT_PROFILE_MIN_GAMES = 3


def find_stockfish() -> Optional[str]:
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in (shutil.which("stockfish"), "/usr/games/stockfish",
                      "/usr/local/bin/stockfish", "/usr/bin/stockfish"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


class ApproxTokenizer:
    """Whitespace-split token counter - PromptBuilder only needs
    .tokenize(text) to return something with a len() for token-budget
    accounting, not real subword tokenization."""

    def tokenize(self, text: str) -> List[str]:
        return text.split()


def build_runner():
    """None if no LLM backend is configured - callers must handle that
    by skipping the LLM step entirely, not treating it as an error."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from llm_kit.llm_runtime import GenerationConfig, OpenRouterRunner

    models = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001"]
    generation_kwargs = GenerationConfig(max_tokens=1024).to_chat_completions(seed=42)
    return OpenRouterRunner(models=models, generation_kwargs=generation_kwargs, api_key=api_key)


def format_game_report(report: GameReport) -> str:
    lines = [f"### {report.game_id}", f"Opening: {report.opening or 'unknown'}"]
    if not report.mistakes:
        lines.append("No mistakes above the flagged threshold.")
        return "\n\n".join(lines)

    lines.append(f"{len(report.mistakes)} mistake(s), worst first:")
    for mistake in report.mistakes:
        color = "White" if mistake.color else "Black"
        line = f"- Move {mistake.move_number} ({color}): {mistake.played_move}"
        if mistake.best_move:
            line += f" (best: {mistake.best_move})"
        line += f" - lost {mistake.loss_cp / 100:.2f} pawns"
        if mistake.safe_alternatives is not None:
            line += f", {mistake.safe_alternatives} safe alternative(s)"
        lines.append(line)
    return "\n".join(lines)


def format_profile(profile: PlayerProfile, groups: List[OpeningGroupSummary]) -> str:
    lines = [f"## Cross-game patterns ({profile.total_games} games)"]
    if not profile.phase_stats and not groups:
        lines.append("Not enough games yet for cross-game patterns - paste more from the same player.")
        return "\n\n".join(lines)

    if profile.phase_stats:
        lines.append("### By phase of game")
        for stat in sorted(profile.phase_stats, key=lambda s: s.avg_loss_cp, reverse=True):
            color = "White" if stat.color else "Black"
            lines.append(
                f"- Moves {stat.phase} as {color}: {stat.mistake_count} mistakes across "
                f"{stat.games_with_data} games, avg loss {stat.avg_loss_cp / 100:.2f} pawns"
            )

    if groups:
        lines.append("### By opening")
        for group in sorted(groups, key=lambda g: g.avg_loss_cp, reverse=True):
            label = group.name or " ".join(group.signature)
            line = (f"- {label}: {group.mistake_count} mistakes across {group.game_count} games, "
                    f"avg loss {group.avg_loss_cp / 100:.2f} pawns")
            if group.idea:
                line += f" — idea: {group.idea}"
            lines.append(line)

    return "\n".join(lines)


def analyze(username: str, pgn_text: str, depth: int) -> str:
    stockfish_path = find_stockfish()
    if stockfish_path is None:
        return "No Stockfish binary found. Install it (e.g. `apt install stockfish`) or set STOCKFISH_PATH."

    games = split_pgn_games(pgn_text)
    if not games:
        return "Couldn't find any games in the pasted PGN."

    player_name = username.strip() or None
    limit = chess.engine.Limit(depth=int(depth))
    conn = open_store(DB_PATH)
    runner = build_runner()
    tokenizer = ApproxTokenizer()
    enough_for_profile = len(games) >= DEFAULT_PROFILE_MIN_GAMES

    sections: List[str] = []

    if runner is not None:
        coached = analyze_and_coach_games(conn, games, stockfish_path, limit, tokenizer, runner,
                                           player_name=player_name)
        for coached_game in coached:
            sections.append(format_game_report(coached_game.report))
            if coached_game.feedback:
                sections.append(f"**Coaching:** {coached_game.feedback}")

        if enough_for_profile:
            profile_result = analyze_and_coach_profile(
                conn, games, stockfish_path, limit, tokenizer, runner,
                player_name=player_name, profile_min_games=DEFAULT_PROFILE_MIN_GAMES,
            )
            sections.append(format_profile(profile_result.profile, profile_result.opening_groups))
            if profile_result.feedback:
                sections.append(f"**Overall coaching:** {profile_result.feedback}")
    else:
        reports = analyze_and_cache_games(conn, games, stockfish_path, limit, player_name=player_name)
        sections.extend(format_game_report(report) for report in reports)
        sections.append("_(No OPENROUTER_API_KEY set - showing raw analysis only, no LLM commentary.)_")

        if enough_for_profile:
            game_hashes = [compute_game_hash(game) for game in games]
            reports_by_hash = dict(zip(game_hashes, reports))
            moves_by_hash = {h: extract_move_prefix(g) for h, g in zip(game_hashes, games)}
            profile = build_player_profile(reports, min_games=DEFAULT_PROFILE_MIN_GAMES)
            groups = build_opening_groups(reports_by_hash, moves_by_hash, min_games=DEFAULT_PROFILE_MIN_GAMES)
            sections.append(format_profile(profile, groups))

    return "\n\n---\n\n".join(sections)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="maestro") as demo:
        gr.Markdown("# maestro - chess game analysis and coaching")
        with gr.Row():
            username = gr.Textbox(label="Your username (matched against White/Black in the PGN)")
            depth = gr.Slider(minimum=4, maximum=20, value=12, step=1, label="Engine depth")
        pgn_input = gr.Textbox(label="Paste PGN (one or more games)", lines=15)
        analyze_button = gr.Button("Analyze", variant="primary")
        output = gr.Markdown()
        analyze_button.click(analyze, inputs=[username, pgn_input, depth], outputs=output)
    return demo


if __name__ == "__main__":
    build_app().launch()
