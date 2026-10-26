import re
from dataclasses import dataclass
from pathlib import Path

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from optichat.backend.config.constants import (
    HAS_INFEASIBILITY_DIAGNOSIS,
    INFEASIBILITY_DIAGNOSIS,
    MODEL_DESCRIPTION,
    MODEL_VERSIONS,
    MODELS_DICTIONARY,
    TEMPORARY_STATES,
)
from optichat.backend.initialization import InitializedChat
from optichat.backend.tools.illustrator_tool import save_synthetic_paper

COMPONENT_REFERENCE_HEADING = "## Component Reference (for expert agent)"
MAX_TITLE_LENGTH = 80

_APP_NAME = "illustrator"
_USER_ID = "local"
# "Title: ..." / "Language: ...", tolerating Markdown decoration such as **Title:**
_HEADER_LINE = re.compile(
    r"^[\s*#>_-]*(title|language)[\s*_]*:[\s*_]*(.*?)[\s*_]*$", re.IGNORECASE
)
_HEADER_SEARCH_LINES = 10


class DescriptionError(Exception):
    """The illustrator agent did not produce a usable description."""


@dataclass(frozen=True)
class ModelDescription:
    title: str
    """Chat title, in the chat's language."""
    language: str
    """Natural language of the model's documentation and code (e.g. "Spanish")."""
    document: str
    """Markdown description: narrative (Part 1) and component reference (Part 2)."""

    @property
    def narrative(self) -> str:
        """Part 1, written for the user; Part 2 is a reference for the expert agent."""
        return self.document.split(COMPONENT_REFERENCE_HEADING, 1)[0].strip()


async def describe_model(
    agent: BaseAgent, chat: InitializedChat, paper: Path | None
) -> ModelDescription:
    """Run the illustrator on the chat's primary model and save its description.

    Must run inside the chat's workspace and solver context. Raises DescriptionError
    if the agent output has no title, language or document.
    """
    sessions = InMemorySessionService()
    session = await sessions.create_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        state={
            **chat.state,
            **TEMPORARY_STATES,
            HAS_INFEASIBILITY_DIAGNOSIS: chat.diagnosis is not None,
            INFEASIBILITY_DIAGNOSIS: chat.diagnosis or {},
        },
    )
    runner = Runner(app_name=_APP_NAME, agent=agent, session_service=sessions)
    error = None
    async for event in runner.run_async(
        user_id=_USER_ID, session_id=session.id, new_message=_request(chat.state, paper)
    ):
        error = event.error_message or error
    session = await sessions.get_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=session.id
    )
    output = session.state.get(MODEL_DESCRIPTION) or ""
    if error and not output:
        raise DescriptionError(error)

    description = parse_illustrator_output(output)
    save_synthetic_paper(description.document, chat.state[MODEL_VERSIONS][0])
    return description


def _request(state: dict, paper: Path | None) -> types.Content:
    model_name = state[MODEL_VERSIONS][0]
    text = f"Generate the description of the model `{model_name}`."
    source_path = state[MODELS_DICTIONARY][model_name].get("source_file_path")
    if source_path:
        source = Path(source_path).read_text(encoding="utf-8", errors="replace")
        text += f"\n\n## Model source code\n\n```python\n{source}\n```"

    parts = []
    if paper is not None and paper.suffix.lower() == ".pdf":
        text += "\n\nThe model paper provided by the user is attached."
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="application/pdf",
                    data=paper.read_bytes(),
                    display_name=paper.name,
                )
            )
        )
    elif paper is not None:
        paper_text = paper.read_text(encoding="utf-8", errors="replace")
        text += f"\n\n## Model paper provided by the user ({paper.name})"
        text += f"\n\n{paper_text}"
    return types.Content(role="user", parts=[types.Part(text=text), *parts])


def parse_illustrator_output(output: str) -> ModelDescription:
    """Split the illustrator output into its Title/Language header and the document."""
    lines = output.strip().splitlines()
    header: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines[:_HEADER_SEARCH_LINES]):
        stripped = line.strip()
        match = _HEADER_LINE.match(stripped)
        if match:
            header[match.group(1).lower()] = match.group(2)
            body_start = i + 1
        elif stripped and not stripped.startswith("```"):
            break

    body = lines[body_start:]
    # Blank lines or a code fence around the header are not part of the document
    while body and (not body[0].strip() or body[0].strip().startswith("```")):
        body.pop(0)
    document = "\n".join(body).strip()

    title = _clean(header.get("title", ""))[:MAX_TITLE_LENGTH].rstrip()
    language = _clean(header.get("language", ""))
    fields = {"title": title, "language": language, "document": document}
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise DescriptionError(
            f"The model description has no {', '.join(missing)} "
            f"(illustrator output: {output[:200]!r})."
        )
    return ModelDescription(title=title, language=language, document=document)


def _clean(value: str) -> str:
    """Collapse whitespace and drop surrounding quotes and periods."""
    return re.sub(r"\s+", " ", value).strip(" \"'`“”«».")
