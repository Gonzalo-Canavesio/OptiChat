from functools import partial

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from optichat.frontend.state import SessionStateManager, queue_feedback

DEFAULTS = {
    "current_view": "chat",
    "current_chat": None,
    "feedback": {"page": [], "sidebar": [], "chat": []},
    "pending_prompt": None,
    "pending_chat": None,
    "main_view_id": None,
    "main_view_slot": 0,
}


@pytest.fixture
def session(monkeypatch) -> dict:
    """A fresh session state per test; bare mode would share one across them."""
    fake: dict = {}
    monkeypatch.setattr(st, "session_state", fake)
    return fake


@pytest.fixture
def manager(session) -> SessionStateManager:
    return SessionStateManager()


def test_initialize_sets_every_default(session, manager):
    manager.initialize()

    assert session == DEFAULTS


def test_initialize_keeps_values_already_set(session, manager):
    already_set = {
        "current_view": "settings",
        "current_chat": {"id": "67", "name": "Chat Example"},
        "feedback": {"page": [partial(print, "saved")], "sidebar": [], "chat": []},
        "pending_prompt": "Solve it",
        "pending_chat": {"chat_name": "Diet"},
        "main_view_id": "chat_67",
        "main_view_slot": 1,
    }
    session.update(already_set)

    manager.initialize()

    assert session == already_set


@pytest.mark.parametrize(
    "attribute",
    [
        "current_view",
        "current_chat",
        "pending_prompt",
        "pending_chat",
        "main_view_id",
        "main_view_slot",
        "is_waiting_response",
    ],
)
def test_reading_the_state_before_initialize_fails(manager, attribute):
    with pytest.raises(KeyError):
        getattr(manager, attribute)


def test_queue_feedback_before_initialize_fails():
    with pytest.raises(KeyError):
        queue_feedback(lambda: None, "page")


def test_profile_and_solver_are_none_without_their_widgets(manager):
    assert manager.selected_profile is None
    assert manager.selected_solver is None


@pytest.mark.parametrize(
    ("attribute", "key", "value"),
    [
        ("current_view", "current_view", "settings"),
        ("current_chat", "current_chat", {"id": "1", "name": "Diet"}),
        ("pending_prompt", "pending_prompt", "Solve it"),
        ("pending_chat", "pending_chat", {"chat_name": "Diet"}),
        ("main_view_id", "main_view_id", "chat_1"),
        ("main_view_slot", "main_view_slot", 1),
        ("selected_profile", "profile_selectbox", object()),
        ("selected_solver", "solver_selectbox", "HiGHS"),
    ],
)
def test_properties_read_and_write_their_session_key(
    session, manager, attribute, key, value
):
    setattr(manager, attribute, value)

    assert session[key] is value
    assert getattr(manager, attribute) is value


@pytest.mark.parametrize(
    ("pending_prompt", "pending_chat", "waiting"),
    [
        (None, None, False),
        ("Solve it", None, True),
        (None, {"chat_name": "Diet"}, True),
        ("Solve it", {"chat_name": "Diet"}, True),
        ("", None, True),
    ],
)
def test_is_waiting_response_while_a_prompt_or_chat_is_pending(
    manager, pending_prompt, pending_chat, waiting
):
    manager.pending_prompt = pending_prompt
    manager.pending_chat = pending_chat

    assert manager.is_waiting_response is waiting


def test_queue_feedback_keeps_the_order_without_running_it(session, manager):
    manager.initialize()
    calls: list[str] = []
    saved = partial(calls.append, "saved")
    deleted = partial(calls.append, "deleted")

    queue_feedback(saved, "page")
    queue_feedback(deleted, "page")

    assert session["feedback"]["page"] == [saved, deleted]
    assert calls == []


def test_queue_feedback_goes_to_its_slot(session, manager):
    manager.initialize()
    opened = partial(print, "opened")
    sent = partial(print, "sent")

    queue_feedback(opened, "sidebar")
    queue_feedback(sent, "chat")

    assert session["feedback"] == {"page": [], "sidebar": [opened], "chat": [sent]}


def _feedback_app():
    from functools import partial

    import streamlit as st

    from optichat.frontend.state import queue_feedback, show_feedback, state

    state.initialize()
    if st.session_state.get("queue", True):
        queue_feedback(
            partial(st.success, "Profile saved", icon=":material/check_circle:"), "page"
        )
        queue_feedback(partial(st.warning, "Connection unused"), "page")
        queue_feedback(partial(st.warning, "Profile unused"), "page")
        queue_feedback(partial(st.error, "Couldn't open this chat"), "sidebar")
        queue_feedback(partial(st.error, "Failed to send message"), "chat")
        st.session_state["queue"] = False
    show_feedback("page")
    with st.sidebar:
        show_feedback("sidebar")


def test_show_feedback_runs_each_slot_where_it_is_shown_once():
    app = AppTest.from_function(_feedback_app).run()

    assert not app.exception
    (success,) = app.main.success
    assert success.value == "Profile saved"
    assert success.icon == ":material/check_circle:"
    assert [w.value for w in app.main.warning] == [
        "Connection unused",
        "Profile unused",
    ]
    assert not app.main.error
    assert [e.value for e in app.sidebar.error] == ["Couldn't open this chat"]
    assert app.session_state["feedback"]["page"] == []
    assert app.session_state["feedback"]["sidebar"] == []
    assert len(app.session_state["feedback"]["chat"]) == 1

    app = app.run()

    assert not app.exception
    assert not app.success
    assert not app.warning
    assert not app.error


def _late_feedback_app():
    from functools import partial

    import streamlit as st

    from optichat.frontend.state import queue_feedback, show_feedback, state

    state.initialize()
    show_feedback("chat")
    if st.session_state.get("queue", True):
        queue_feedback(partial(st.error, "Failed to send message"), "chat")
        st.session_state["queue"] = False


def test_show_feedback_leaves_feedback_queued_after_it_for_the_next_run():
    app = AppTest.from_function(_late_feedback_app).run()

    assert not app.exception
    assert not app.error
    assert len(app.session_state["feedback"]["chat"]) == 1

    app = app.run()

    assert not app.exception
    assert [e.value for e in app.error] == ["Failed to send message"]
    assert app.session_state["feedback"]["chat"] == []


def _configure_page_app():
    from optichat.frontend.state import configure_page

    configure_page()


def test_configure_page_initializes_the_state():
    app = AppTest.from_function(_configure_page_app).run()

    assert not app.exception
    for key, value in DEFAULTS.items():
        assert app.session_state[key] == value


def _main_view_app():
    import streamlit as st

    from optichat.frontend.state import main_view_container, state

    state.initialize()
    view_id = st.session_state.get("view", "chat_1")
    with main_view_container(view_id):
        st.markdown(f"Showing {view_id}")


def _show(app: AppTest, view_id: str) -> AppTest:
    app.session_state["view"] = view_id
    return app.run()


def test_main_view_container_renders_the_view():
    app = AppTest.from_function(_main_view_app).run()

    assert not app.exception
    assert [m.value for m in app.markdown] == ["Showing chat_1"]
    assert app.session_state["main_view_id"] == "chat_1"
    assert app.session_state["main_view_slot"] == 1


def test_main_view_container_keeps_the_slot_while_the_view_is_the_same():
    app = AppTest.from_function(_main_view_app).run()
    slot = app.session_state["main_view_slot"]

    app = _show(app, "chat_1")

    assert not app.exception
    assert app.session_state["main_view_slot"] == slot


def test_main_view_container_switches_slot_for_a_new_view():
    app = AppTest.from_function(_main_view_app).run()
    first_slot = app.session_state["main_view_slot"]

    app = _show(app, "chat_2")

    assert not app.exception
    assert app.session_state["main_view_slot"] == 1 - first_slot
    assert app.session_state["main_view_id"] == "chat_2"
    assert [m.value for m in app.markdown] == ["Showing chat_2"]

    app = _show(app, "settings")

    assert app.session_state["main_view_slot"] == first_slot
    assert [m.value for m in app.markdown] == ["Showing settings"]
