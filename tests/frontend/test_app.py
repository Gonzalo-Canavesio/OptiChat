"""The app must switch between chat and settings, and stay blocked until set up."""

from functools import partial
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.proto.Block_pb2 import Block
from streamlit.testing.v1 import AppTest

from optichat.frontend import api
from optichat.llm import credentials, profiles
from optichat.llm.models import AgentConfig, LLMCredential, ProfileConfig
from optichat.llm.store import save_items

APP = str(Path(__file__).parents[2] / "optichat" / "frontend" / "app.py")

CREDENTIAL = LLMCredential(label="Mine", provider="OpenAI", api_key="sk-good")
AGENT = AgentConfig(credential_label="Mine", llm_model="gpt-4o")
PROFILE = ProfileConfig(
    label="Profile",
    root_agent=AGENT,
    expert_agent=AGENT,
    illustrator_agent=AGENT,
    generator_agent=AGENT,
)

FIRST_CONNECTION = "Add your first connection"
FIRST_PROFILE = "Configure your first LLM profile"


@pytest.fixture
def fresh_install(tmp_path, monkeypatch):
    """No saved credentials or profiles, and a backend with no chats."""
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "credentials.yaml")
    monkeypatch.setattr(profiles, "PROFILES_FILE", tmp_path / "profiles.yaml")
    monkeypatch.setattr(
        api,
        "_client",
        httpx.Client(
            base_url=api.API_BASE_URL,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])),
        ),
    )


def _save_credential() -> None:
    save_items(credentials.CREDENTIALS_FILE, "credentials", [CREDENTIAL.model_dump()])


def _save_profile() -> None:
    save_items(profiles.PROFILES_FILE, "profiles", [PROFILE.model_dump()])


@pytest.fixture
def configured(fresh_install):
    """A saved connection and profile, so the app skips the first-run modals."""
    _save_credential()
    _save_profile()


def _run(view: str | None = None) -> AppTest:
    app = AppTest.from_file(APP)
    if view is not None:
        app.session_state["current_view"] = view
    return app.run()


def _dialogs(app: AppTest) -> list[Block.Dialog]:
    return [dialog.proto.dialog for dialog in app.get("dialog")]


def test_a_new_session_starts_on_the_chat_view(configured):
    app = _run()

    assert not app.exception
    assert app.session_state["current_view"] == "chat"
    assert app.sidebar.children
    assert app.session_state["main_view_id"] != "settings"


def test_the_settings_view_replaces_the_chat_and_its_sidebar(configured):
    app = _run("settings")

    assert not app.exception
    assert app.session_state["main_view_id"] == "settings"
    assert not app.sidebar.children


def test_leaving_the_settings_brings_the_chat_and_its_sidebar_back(configured):
    app = _run("settings")

    app.session_state["current_view"] = "chat"
    app = app.run()

    assert not app.exception
    assert app.session_state["main_view_id"] != "settings"
    assert app.sidebar.children


@pytest.mark.parametrize("view", ["chat", "settings"])
def test_without_connections_a_connection_is_required_first(fresh_install, view):
    app = _run(view)

    assert not app.exception
    (dialog,) = _dialogs(app)
    assert dialog.title == FIRST_CONNECTION
    assert dialog.width == Block.Dialog.DialogWidth.SMALL
    assert not dialog.dismissible


def test_profiles_left_without_connections_still_ask_for_a_connection(
    fresh_install,
):
    _save_profile()

    app = _run()

    assert not app.exception
    assert [d.title for d in _dialogs(app)] == [FIRST_CONNECTION]


@pytest.mark.parametrize("view", ["chat", "settings"])
def test_a_connection_without_profiles_requires_a_profile(fresh_install, view):
    _save_credential()

    app = _run(view)

    assert not app.exception
    (dialog,) = _dialogs(app)
    assert dialog.title == FIRST_PROFILE
    assert dialog.width == Block.Dialog.DialogWidth.LARGE
    assert not dialog.dismissible


@pytest.mark.parametrize("view", ["chat", "settings"])
def test_a_configured_app_shows_no_first_run_modal(configured, view):
    app = _run(view)

    assert not app.exception
    assert _dialogs(app) == []


def test_page_feedback_is_shown_once(configured):
    app = AppTest.from_file(APP)
    app.session_state["feedback"] = {
        "page": [partial(st.success, "Profile saved")],
        "sidebar": [],
        "chat": [],
    }

    app = app.run()
    assert [s.value for s in app.success] == ["Profile saved"]

    app = app.run()
    assert not app.success
