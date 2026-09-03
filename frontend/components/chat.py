from typing import Literal

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from frontend.api.client import create_chat, send_message
from frontend.state import ChatMessage, clean_session_state_for_chat_form, state
from llms import ProfileConfig

ModelingLanguage = Literal["Pyomo", "GAMSPy", "gurobipy"]
AVAILABLE_MODELING_LANGUAGES: list[ModelingLanguage] = [
    "Pyomo",
    "GAMSPy",
    "gurobipy",
]


def _handle_create_chat_submission(
    model_files: list[UploadedFile] | None,
    data_files: list[UploadedFile] | None,
    modeling_language: ModelingLanguage | None,
    profile: ProfileConfig,
    solver: str,
    chat_name: str,
) -> bool:
    if modeling_language is None:
        st.error("Please select a modeling language.")
        return False
    if model_files is None:
        st.error("Please upload a .py model file.")
        return False

    with st.spinner("Initializing model and starting chat..."):
        chat = create_chat(
            model_files=model_files,
            data_files=data_files,
            modeling_language=modeling_language,
            profile_label=profile.label,
            solver=solver,
            chat_name=chat_name.strip(),
        )

    if chat is None:
        st.error("Failed to create chat. Please check your backend connection.")
        return False

    state.current_chat = chat
    return True


@st.dialog(
    "Create New Chat", width="medium", on_dismiss=clean_session_state_for_chat_form
)
def render_create_chat_dialog() -> None:
    if state.selected_profile is None:
        st.warning("Please select an LLM profile in the sidebar first.")
        if st.button("Close", width="stretch"):
            st.rerun()
        return
    if state.selected_solver is None:
        st.warning("Please select a solver in the sidebar first.")
        if st.button("Close", width="stretch"):
            st.rerun()
        return

    selected_language = st.selectbox(
        "Modeling Language (Required)",
        options=AVAILABLE_MODELING_LANGUAGES,
        index=None,
        key="modeling_language_selectbox",
        help="Choose the framework used in your model.",
    )

    if selected_language == "gurobipy":
        st.warning(
            "**GurobiPy** ignores the custom solver selected in the sidebar "
            "and uses its native Gurobi solver."
        )

    with st.form("create_chat_dialog_form", border=False):
        model_files = st.file_uploader(
            "Model files (.py) (Required)",
            type=["py"],
            accept_multiple_files=True,
        )
        data_files = st.file_uploader(
            "Data files (Optional)",
            accept_multiple_files=True,
        )
        chat_name = st.text_input(
            "Chat name (Optional)",
            help="Leave empty for auto-generated name.",
        )
        submitted = st.form_submit_button("Create", type="primary", width="stretch")

    if submitted:
        if _handle_create_chat_submission(
            model_files=model_files,
            data_files=data_files,
            modeling_language=selected_language,
            profile=state.selected_profile,
            solver=state.selected_solver,
            chat_name=chat_name,
        ):
            clean_session_state_for_chat_form()
            st.rerun()


def _render_empty_chat_placeholder() -> None:
    st.markdown("# No active conversation")
    st.write(":gray[Start a new chat to begin analyzing an optimization model.]")
    if st.button("New Chat", icon=":material/add:", type="primary"):
        render_create_chat_dialog()


def _render_chat_message(message: ChatMessage) -> None:
    with st.chat_message(message["role"]):
        thought = message.get("thought")
        if thought:
            with st.expander(":gray[Thoughts...]", expanded=False):
                st.markdown(thought)
        st.markdown(message["content"])


def render_active_chat() -> None:
    chat = state.current_chat
    if not chat:
        return

    messages = chat.get("messages", [])

    for msg in messages:
        _render_chat_message(msg)

    if prompt := st.chat_input("Enter your query here..."):
        if not state.selected_profile:
            st.error("Please select an LLM profile in the sidebar first.")
            return
        if not state.selected_solver:
            st.error("Please select a solver in the sidebar first.")
            return

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            new_chat = send_message(
                chat_id=chat["id"],
                message=prompt,
                profile_label=state.selected_profile.label,
                solver=state.selected_solver,
            )

        if new_chat is None:
            st.error("Failed to send message. Please check your backend connection.")
            return

        state.current_chat = new_chat
        st.rerun()


def render_chat_view() -> None:
    if state.current_chat is not None:
        render_active_chat()
    else:
        _render_empty_chat_placeholder()
