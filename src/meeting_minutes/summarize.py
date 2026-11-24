"""Turn a labelled transcript into meeting minutes using the Claude Agent SDK.

The Agent SDK reuses the Claude Code login on this machine, so this runs on the
user's existing subscription rather than a metered API key.

Two options do the important work here:

    tools=[]            no Bash/Read/Edit/WebSearch. This is a pure text
                        transformation; giving it tools only adds ways to go
                        wrong and latency while it decides not to use them.
    setting_sources=[]  ignore CLAUDE.md and settings.json entirely. Without
                        this, whatever project directory the command happens to
                        run in would silently alter the summary's style.

Together they make the output a function of (prompt, transcript) and nothing
else — which is what makes prompt iteration meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLINotFoundError,
    ResultMessage,
    TextBlock,
    query,
)

log = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "minutes_th.md"


@dataclass
class Minutes:
    markdown: str
    cost_usd: float | None = None
    duration_ms: int | None = None


def load_prompt(path: Path | None = None) -> str:
    prompt_path = path or DEFAULT_PROMPT_PATH
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_user_message(
    transcript_text: str,
    *,
    title: str | None,
    audio_duration: str | None,
    speakers: list[str] | None,
    meeting_date: str | None,
) -> str:
    """Assemble the turn Claude actually reads.

    Metadata goes above the transcript so the model has the framing before it
    starts reading a potentially very long body of text.
    """
    header = ["## ข้อมูลการประชุม"]
    header.append(f"- หัวข้อ: {title}" if title else "- หัวข้อ: [ยังไม่ระบุ]")
    header.append(f"- วันที่: {meeting_date}" if meeting_date else "- วันที่: [ยังไม่ระบุ]")
    if audio_duration:
        header.append(f"- ความยาวไฟล์เสียง: {audio_duration}")
    if speakers:
        header.append(f"- ผู้พูดที่ระบบตรวจพบ: {', '.join(speakers)}")
    else:
        header.append("- ผู้พูด: ไม่ได้แยกผู้พูด (transcript ไม่มีป้ายกำกับ)")

    return (
        "\n".join(header)
        + "\n\n## Transcript\n\n"
        + transcript_text
        + "\n\n---\n\nสร้างรายงานการประชุมจาก transcript ข้างต้นตามรูปแบบที่กำหนด"
    )


async def summarize(
    transcript_text: str,
    *,
    title: str | None = None,
    audio_duration: str | None = None,
    speakers: list[str] | None = None,
    meeting_date: str | None = None,
    prompt_path: Path | None = None,
    model: str | None = None,
) -> Minutes:
    """Run one summarization turn and return the minutes as Markdown."""
    system_prompt = load_prompt(prompt_path)
    user_message = build_user_message(
        transcript_text,
        title=title,
        audio_duration=audio_duration,
        speakers=speakers,
        meeting_date=meeting_date,
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=[],           # text in, text out — no tool surface at all
        setting_sources=[],  # ignore CLAUDE.md / settings.json (see module docstring)
        max_turns=1,
        model=model,
    )

    parts: list[str] = []
    result: ResultMessage | None = None

    log.info("Summarizing transcript (%d characters)...", len(transcript_text))
    try:
        async for message in query(prompt=user_message, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
            elif isinstance(message, ResultMessage):
                result = message
    except CLINotFoundError as exc:
        raise RuntimeError(
            "The Claude Code CLI could not be started. Make sure you are logged "
            "in (`claude` in a terminal), then retry."
        ) from exc

    if result is not None and result.is_error:
        detail = "; ".join(result.errors or []) or result.subtype
        raise RuntimeError(f"Summarization failed: {detail}")

    # `result.result` is the CLI's own final-text field; the streamed assistant
    # blocks are the fallback for versions that leave it unset.
    markdown = (result.result if result and result.result else "".join(parts)).strip()
    if not markdown:
        raise RuntimeError("Summarization returned no text.")

    return Minutes(
        markdown=markdown,
        cost_usd=result.total_cost_usd if result else None,
        duration_ms=result.duration_ms if result else None,
    )
