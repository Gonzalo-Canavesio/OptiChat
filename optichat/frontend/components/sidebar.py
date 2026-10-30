from functools import partial

import streamlit as st

from optichat.frontend.api import BackendError, fetch_chat, fetch_chats
from optichat.frontend.components.chat import render_create_chat_dialog
from optichat.frontend.state import (
    AVAILABLE_SOLVERS,
    queue_feedback,
    show_feedback,
    state,
)
from optichat.llm.profiles import get_llm_profiles


def _open_chat(chat_id: str) -> None:
    try:
        state.current_chat = fetch_chat(chat_id)
    except BackendError as e:
        queue_feedback(partial(st.error, f"Couldn't open this chat: {e}"), "sidebar")


def render_sidebar() -> None:
    locked = state.is_waiting_response

    st.selectbox(
        "Select LLM Profile",
        options=[p for p in get_llm_profiles() if p.enabled],
        format_func=lambda p: p.label,
        index=None,
        key="profile_selectbox",
        help="Choose the LLM profile for your optimization problem.",
        disabled=locked,
    )

    st.button(
        "Manage Connections & Profiles",
        width="stretch",
        disabled=locked,
        on_click=lambda: setattr(state, "current_view", "settings"),
    )

    st.selectbox(
        "Select Solver",
        options=AVAILABLE_SOLVERS,
        index=None,
        key="solver_selectbox",
        help="Select the optimization solver used to solve the model.",
        disabled=locked,
    )

    st.divider()

    if st.button(
        "New Chat",
        width="stretch",
        icon=":material/add:",
        type="primary",
        disabled=locked,
    ):
        if not state.selected_profile:
            st.error("Please select an LLM profile in the sidebar first.")
        if not state.selected_solver:
            st.error("Please select a solver in the sidebar first.")
        if state.selected_profile and state.selected_solver:
            render_create_chat_dialog()

    try:
        chats = fetch_chats()
    except BackendError as e:
        st.error(f"Couldn't load your chats: {e}")
        chats = []

    show_feedback("sidebar")

    for chat in chats:
        st.button(
            chat["name"],
            width="stretch",
            key=f"chat_btn_{chat['id']}",
            type="secondary"
            if state.current_chat and chat["id"] == state.current_chat["id"]
            else "tertiary",
            disabled=locked,
            on_click=_open_chat,
            args=(chat["id"],),
        )
