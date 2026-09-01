import streamlit as st

from frontend.api.client import fetch_chats
from frontend.components.chat import render_create_chat_dialog
from frontend.state import AVAILABLE_SOLVERS, state
from llms import get_llm_profiles


def render_sidebar() -> None:
    st.selectbox(
        "Select LLM Profile",
        options=[p for p in get_llm_profiles() if p.enabled],
        format_func=lambda p: p.label,
        index=None,
        key="profile_selectbox",
        help="Choose the LLM profile for your optimization problem.",
    )

    if st.button("Manage Providers & Profiles", width="stretch"):
        state.current_view = "settings"
        st.rerun()

    st.selectbox(
        "Select Solver",
        options=AVAILABLE_SOLVERS,
        index=None,
        key="solver_selectbox",
        help="Select the optimization solver used to solve the model.",
    )

    st.divider()

    if st.button("💬 Start New Chat", width="stretch"):
        render_create_chat_dialog()

    chats = fetch_chats()
    for chat in chats:
        if st.button(
            chat["name"],
            width="stretch",
            key=f"chat_btn_{chat['id']}",
        ):
            state.current_chat = chat
            state.chat_messages = state.chat_histories.get(chat["id"], [])
            st.rerun()
