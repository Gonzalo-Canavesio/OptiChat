from functools import partial

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from optichat.frontend.api import BackendError, create_chat, send_message
from optichat.frontend.state import (
    ChatInfo,
    ChatMessage,
    PendingChat,
    main_view_container,
    queue_feedback,
    show_feedback,
    state,
)
from optichat.llm.models import ProfileConfig


def _queue_chat_creation(
    model_file: UploadedFile | None,
    data_files: list[UploadedFile] | None,
    model_paper: UploadedFile | None,
    profile: ProfileConfig,
    solver: str,
    chat_name: str,
) -> bool:
    if model_file is None:
        st.error("Please upload a .py model file.")
        return False

    state.pending_chat = PendingChat(
        model_file=model_file,
        data_files=data_files,
        model_paper=model_paper,
        profile_label=profile.label,
        solver=solver,
        chat_name=chat_name.strip(),
    )
    return True


def _create_pending_chat(pending: PendingChat) -> None:
    st.subheader(pending["chat_name"] or "New chat")

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "Loading the model and describing it, this may take a while..."
            ):
                state.current_chat = create_chat(**pending)
    except BackendError as e:
        queue_feedback(partial(st.error, f"Failed to create chat: {e}"), "chat")
    finally:
        state.pending_chat = None
    st.rerun()


@st.dialog("Create New Chat", width="medium")
def render_create_chat_dialog() -> None:
    if state.selected_profile is None or state.selected_solver is None:
        return

    with st.form("create_chat_dialog_form", border=False):
        model_file = st.file_uploader(
            "Model file (.py) (Required)",
            type=["py"],
            help="A Pyomo or GAMSPy model. The modeling language is detected "
            "automatically.",
        )
        data_files = st.file_uploader(
            "Data files (Optional)",
            accept_multiple_files=True,
            help="Files the model reads, in any format. They are placed next to the "
            "model file, so open them with a relative path (e.g. "
            '`open("my_data.json")`).',
        )
        chat_name = st.text_input(
            "Chat name (Optional)",
            help="Leave empty to name the chat after the problem the model solves.",
        )
        model_paper = st.file_uploader(
            "Model paper (Optional)",
            type=["pdf", "md", "txt"],
            help="Upload a file describing your model. "
            "This can help the LLM understand your model better.",
        )
        submitted = st.form_submit_button("Create", type="primary", width="stretch")

    if submitted:
        if _queue_chat_creation(
            model_file=model_file,
            data_files=data_files,
            model_paper=model_paper,
            profile=state.selected_profile,
            solver=state.selected_solver,
            chat_name=chat_name,
        ):
            st.rerun()


def _render_empty_chat_placeholder() -> None:
    st.markdown("# No active conversation")
    st.write(":gray[Start a new chat to begin analyzing an optimization model.]")
    show_feedback("chat")
    if st.button("New Chat", icon=":material/add:", type="primary"):
        if not state.selected_profile:
            st.error("Please select an LLM profile in the sidebar first.")
        if not state.selected_solver:
            st.error("Please select a solver in the sidebar first.")
        if state.selected_profile and state.selected_solver:
            render_create_chat_dialog()


def _render_chat_message(message: ChatMessage) -> None:
    with st.chat_message(message["role"]):
        thought = message.get("thought")
        if thought:
            with st.expander(":gray[Thoughts...]", expanded=False):
                st.markdown(thought)
        st.markdown(message["content"])


def _render_chat_header(chat: ChatInfo) -> None:
    st.subheader(chat["name"])


def _send_pending_prompt(chat: ChatInfo, prompt: str) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)

    assert state.selected_profile is not None, "No profile selected"
    assert state.selected_solver is not None, "No solver selected"

    try:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query, this may take a while..."):
                state.current_chat = send_message(
                    chat_id=chat["id"],
                    message=prompt,
                    profile_label=state.selected_profile.label,
                    solver=state.selected_solver,
                )
    except BackendError as e:
        queue_feedback(partial(st.error, f"Failed to send message: {e}"), "chat")
    finally:
        state.pending_prompt = None
    st.rerun()


def render_active_chat(prompt: str | None) -> None:
    chat = state.current_chat
    if not chat:
        return

    _render_chat_header(chat)
    for msg in chat.get("messages", []):
        _render_chat_message(msg)

    show_feedback("chat")

    if state.pending_prompt is not None:
        _send_pending_prompt(chat, state.pending_prompt)
        return

    if prompt:
        if not state.selected_profile:
            st.error("Please select an LLM profile in the sidebar first.")
        if not state.selected_solver:
            st.error("Please select a solver in the sidebar first.")

        if state.selected_profile and state.selected_solver:
            state.pending_prompt = prompt
            st.rerun()


def _render_chat_input() -> str | None:
    # Called outside any container so it stays pinned to the bottom of the page.
    return st.chat_input(
        "Waiting for the response..."
        if state.is_waiting_response
        else "Enter your query here...",
        disabled=state.is_waiting_response,
    )


def render_chat_view() -> None:
    if state.pending_chat is not None:
        _render_chat_input()
        with main_view_container("pending_chat"):
            _create_pending_chat(state.pending_chat)
    elif state.current_chat is not None:
        prompt = _render_chat_input()
        with main_view_container(f"chat_{state.current_chat['id']}"):
            render_active_chat(prompt)
    else:
        with main_view_container("no_chat"):
            _render_empty_chat_placeholder()
