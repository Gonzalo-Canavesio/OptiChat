import asyncio
import logging
import os
import re
import shutil
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session
from google.genai import types
from loguru import logger
from pydantic import BaseModel

from optichat.backend.agents.illustrator.agent import create_illustrator_agent
from optichat.backend.agents.illustrator.description import (
    DescriptionError,
    describe_model,
)
from optichat.backend.agents.root.agent import create_root_agent
from optichat.backend.agents.root.prompt import (
    build_initial_summary_request,
    is_initial_summary_request,
)
from optichat.backend.config.constants import (
    CHAT_CREATED_AT,
    CHAT_LANGUAGE,
    CHAT_MODELING_LANGUAGE,
    CHAT_NAME,
)
from optichat.backend.config.solver import (
    SolverNotAvailableError,
    get_solver_gamspy_name,
    make_solver,
    normalize_solver,
    use_solver,
)
from optichat.backend.config.workspace import use_workspace
from optichat.backend.initialization import ModelLoadError, initialize_chat
from optichat.backend.tools.timing_tracker import use_tracker
from optichat.llm.catalog import accepts_pdf
from optichat.llm.credentials import get_llm_credential_by_label
from optichat.llm.models import ProfileConfig
from optichat.llm.profiles import get_llm_profile_by_label

# Disable OpenTelemetry to avoid context management issues (same as agent.py)
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

APP_NAME = "backend"
USER_ID = "local"

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("OPTICHAT_DATA_DIR", ROOT / ".optichat"))
CHATS_DIR = DATA_DIR / "chats"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DIR.mkdir(parents=True, exist_ok=True)

sessions = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{DATA_DIR / 'chats.db'}")

# Solver, workspace and timing tracker are scoped to each run, but the tools still share
# a few process-wide objects (e.g. the python REPL namespace), so runs are serialized.
_run_lock = asyncio.Lock()
_agents: dict[str, Any] = {}

app = FastAPI(title="OptiChat API")


class MessageRequest(BaseModel):
    message: str
    profile_label: str
    solver: str


# ----------------------------------------------------------------------- agents


def _get_profile(label: str) -> ProfileConfig:
    profile = get_llm_profile_by_label(label)
    if profile is None or not profile.enabled:
        raise HTTPException(422, f"LLM profile '{label}' not found or incomplete.")
    return profile


def _get_root_agent(profile: ProfileConfig, solver: str):
    """One agent tree per (profile settings, solver); rebuilt if the profile changes."""
    key = f"root|{profile.model_dump_json()}|{solver}"
    if key not in _agents:
        _agents[key] = create_root_agent(profile, workflow="default")
    return _agents[key]


def _get_illustrator_agent(profile: ProfileConfig):
    key = f"illustrator|{profile.model_dump_json()}"
    if key not in _agents:
        _agents[key] = create_illustrator_agent(profile)
    return _agents[key]


def _prepare_run(profile_label: str, solver_label: str, modeling_language: str | None):
    """Profile and solver for a run, failing early if the solver cannot be used.

    A chat being created has no modeling language yet (it is detected from the model
    file), so the solver is not checked; solving the model reports the problem.
    """
    try:
        solver = normalize_solver(solver_label)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    profile = _get_profile(profile_label)
    try:
        with use_solver(solver):
            # fail early with a clear message if the solver cannot be used
            if modeling_language == "GAMSPy":
                get_solver_gamspy_name()
            elif modeling_language is not None:
                make_solver()
    except SolverNotAvailableError as e:
        raise HTTPException(422, str(e)) from e
    return profile, solver


@contextmanager
def _chat_context(chat_id: str, solver: str) -> Generator[None]:
    """The chat's workspace, solver and timing tracker, for initialization and runs."""
    with (
        use_workspace(CHATS_DIR / chat_id / "artifacts"),
        use_solver(solver),
        use_tracker(),
    ):
        yield


async def _run_agent(chat_id: str, content: types.Content, profile, solver) -> None:
    runner = Runner(
        app_name=APP_NAME,
        agent=_get_root_agent(profile, solver),
        session_service=sessions,
    )
    try:
        with _chat_context(chat_id, solver):
            async for _ in runner.run_async(
                user_id=USER_ID, session_id=chat_id, new_message=content
            ):
                pass
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


# --------------------------------------------------------------------- chat data


def _loaded_models(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Name, solve status and objective of the models this chat was created with."""
    dictionary = state.get("MODELS_DICTIONARY") or {}
    loaded = []
    for name in state.get("MODEL_VERSIONS") or []:
        objective = (dictionary.get(name) or {}).get("obj") or {}
        loaded.append(
            {
                "name": name,
                "status": objective.get("sol_status", "unknown"),
                "objective": objective.get("value"),
                "sense": objective.get("sense"),
            }
        )
    return loaded


def _chat_summary(session: Session) -> dict[str, Any]:
    state = session.state
    return {
        "models": _loaded_models(state),
        "id": session.id,
        "name": state[CHAT_NAME],
        "modeling_language": state[CHAT_MODELING_LANGUAGE],
        "created_at": state[CHAT_CREATED_AT],
    }


def _chat_messages(session: Session) -> list[dict[str, Any]]:
    """Rebuild the visible conversation from the stored ADK events."""
    messages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_invocation = None
    for event in session.events:
        parts = event.content.parts if event.content and event.content.parts else []
        if event.author == "user":
            text = "".join(p.text or "" for p in parts).strip()
            if text and not is_initial_summary_request(text):
                messages.append({"role": "user", "content": text})
            current = None
            continue
        for part in parts:
            if not part.text:
                continue
            if current is None or event.invocation_id != current_invocation:
                current = {"role": "assistant", "content": "", "thought": ""}
                messages.append(current)
                current_invocation = event.invocation_id
            field = "thought" if part.thought else "content"
            current[field] += ("\n\n" if current[field] else "") + part.text.strip()
    for message in messages:
        if message["role"] == "assistant" and not message["thought"]:
            message.pop("thought")
    return [m for m in messages if m["content"] or m.get("thought")]


async def _get_session(chat_id: str) -> Session:
    session = await sessions.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=chat_id
    )
    if session is None:
        raise HTTPException(404, f"Chat '{chat_id}' not found.")
    return session


async def _chat_detail(chat_id: str) -> dict[str, Any]:
    session = await _get_session(chat_id)
    return {**_chat_summary(session), "messages": _chat_messages(session)}


# --------------------------------------------------------------------- creation


def _save_upload(upload: UploadFile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (Path(upload.filename or "upload").name)
    with path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path


def _build_config(model: Path) -> dict[str, Any]:
    return {
        "model_name": model.stem,
        "models": {
            "local_resources": [str(model)],
            "is_solved": False,
            "is_lp": False,
        },
        "models_code": {"local_resources": [str(model)]},
    }


@app.get("/chats")
async def list_chats() -> list[dict[str, Any]]:
    response = await sessions.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    ordered = sorted(response.sessions, key=lambda s: s.last_update_time, reverse=True)
    return [_chat_summary(s) for s in ordered]


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str) -> dict[str, Any]:
    return await _chat_detail(chat_id)


@app.post("/chats")
async def create_chat(
    profile_label: str = Form(...),
    solver: str = Form(...),
    chat_name: str = Form(""),
    model_file: UploadFile = File(...),
    data_files: list[UploadFile] | None = File(None),
    model_paper: UploadFile | None = File(None),
) -> dict[str, Any]:
    model_name = Path(model_file.filename or "").name
    if not model_name.endswith(".py"):
        raise HTTPException(422, "The model file must be a .py file.")
    # The model and its data files share a directory, so names must not collide.
    data_names = [Path(f.filename or "").name for f in data_files or []]
    if model_name in data_names or len(set(data_names)) != len(data_names):
        raise HTTPException(
            422, "The model and data files must all have different file names."
        )

    async with _run_lock:
        profile, solver_key = _prepare_run(profile_label, solver, None)
        if model_paper:
            _check_paper_format(model_paper, profile)

        chat_id = uuid.uuid4().hex[:12]
        chat_dir = CHATS_DIR / chat_id
        try:
            # The model runs from this directory and reads its data files from it
            model = _save_upload(model_file, chat_dir / "model")
            for data_file in data_files or []:
                _save_upload(data_file, chat_dir / "model")
            paper = (
                _save_upload(model_paper, chat_dir / "paper") if model_paper else None
            )
            with _chat_context(chat_id, solver_key):
                chat = initialize_chat(_build_config(model))
                description = await describe_model(
                    _get_illustrator_agent(profile), chat, paper
                )
        except Exception as e:
            shutil.rmtree(chat_dir, ignore_errors=True)
            raise _creation_error(e) from e

        await sessions.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=chat_id,
            state={
                **chat.state,
                CHAT_NAME: re.sub(r"\s+", " ", chat_name).strip() or description.title,
                CHAT_LANGUAGE: description.language,
                CHAT_CREATED_AT: datetime.now().isoformat(timespec="seconds"),
            },
        )
        summary_request = build_initial_summary_request(
            description.narrative, description.language
        )
        try:
            await _run_agent(
                chat_id,
                types.Content(role="user", parts=[types.Part(text=summary_request)]),
                profile,
                solver_key,
            )
        except Exception:
            await sessions.delete_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=chat_id
            )
            shutil.rmtree(chat_dir, ignore_errors=True)
            raise

    return await _chat_detail(chat_id)


def _check_paper_format(paper: UploadFile, profile: ProfileConfig) -> None:
    """A PDF paper is sent as is to the illustrator, so its model must accept PDFs."""
    if Path(paper.filename or "").suffix.lower() != ".pdf":
        return
    label = profile.illustrator_agent.credential_label
    credential = get_llm_credential_by_label(label)
    if credential is None:
        raise HTTPException(
            422,
            f"The connection '{label}' of the profile '{profile.label}' is not "
            "configured.",
        )
    model = profile.illustrator_agent.llm_model
    if not accepts_pdf(credential, model):
        raise HTTPException(
            422,
            f"The illustrator model of the profile '{profile.label}' ({model}) does "
            "not accept PDF files. Upload the paper as a .md or .txt file, or use a "
            "profile whose illustrator model accepts PDFs.",
        )


def _creation_error(error: Exception) -> HTTPException:
    if isinstance(error, ModelLoadError):
        return HTTPException(
            422,
            f"The model could not be loaded ({error}). Check that the .py file "
            "defines a Pyomo or GAMSPy model and that every data file it reads was "
            "uploaded.",
        )
    if isinstance(error, DescriptionError):
        return HTTPException(
            502, f"The model description could not be generated: {error}"
        )
    logger.opt(exception=error).error("Chat creation failed")
    return HTTPException(500, f"{type(error).__name__}: {error}")


@app.post("/chats/{chat_id}/messages")
async def send_message(chat_id: str, request: MessageRequest) -> dict[str, Any]:
    session = await _get_session(chat_id)
    modeling_language = session.state[CHAT_MODELING_LANGUAGE]
    async with _run_lock:
        profile, solver = _prepare_run(
            request.profile_label, request.solver, modeling_language
        )
        content = types.Content(role="user", parts=[types.Part(text=request.message)])
        await _run_agent(chat_id, content, profile, solver)
    return await _chat_detail(chat_id)
