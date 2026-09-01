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
    model_file: UploadedFile | None,
    data_files: list[UploadedFile] | None,
    modeling_language: ModelingLanguage | None,
    profile: ProfileConfig,
    solver: str,
    chat_name: str,
) -> bool:
    if modeling_language is None:
        st.error("Please select a modeling language.")
        return False
    if model_file is None:
        st.error("Please upload a .py model file.")
        return False

    with st.spinner("Initializing model and starting chat..."):
        chat = create_chat(
            model_file=model_file,
            data_files=data_files,
            modeling_language=modeling_language,
            profile_label=profile.label,
            solver=solver,
            chat_name=chat_name.strip(),
        )

    if chat is None:
        st.error("Failed to create chat. Please check your backend connection.")
        return False

    chat_id = chat.get("id", "")
    initial_response = chat.get("initial_response", "")

    state.current_chat = chat
    state.chat_messages = [{"role": "assistant", "content": initial_response}]
    if chat_id:
        state.chat_histories[chat_id] = state.chat_messages

    return True


@st.dialog(
    "Create New Chat", width="medium", on_dismiss=clean_session_state_for_chat_form
)
def render_create_chat_dialog() -> None:
    profile = state.selected_profile
    solver = state.selected_solver
    if profile is None:
        st.warning("Please select an LLM profile in the sidebar first.")
    if solver is None:
        st.warning("Please select a solver in the sidebar first.")
    if profile is None or solver is None:
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
        model_file: UploadedFile | None = st.file_uploader(
            "Model file (.py) (Required)",
            type=["py"],
            accept_multiple_files=False,
        )
        data_files: list[UploadedFile] | None = st.file_uploader(
            "Data files (Optional)",
            accept_multiple_files=True,
        )
        chat_name = st.text_input(
            "Chat name (Optional)",
            help="Leave empty for auto-generated name.",
        )
        submitted = st.form_submit_button(
            "Create Chat", type="primary", width="stretch"
        )

    if submitted:
        if _handle_create_chat_submission(
            model_file=model_file,
            data_files=data_files,
            modeling_language=selected_language,
            profile=profile,
            solver=solver,
            chat_name=chat_name,
        ):
            clean_session_state_for_chat_form()
            st.rerun()


def _render_empty_chat_placeholder() -> None:
    st.markdown("### No active conversation")
    st.caption("Start a new chat to begin analyzing an optimization model.")
    if st.button("💬 Start New Chat"):
        render_create_chat_dialog()


def render_active_chat() -> None:
    chat = state.current_chat
    if not chat:
        return

    chat_id = chat.get("id", "")
    chat_name = chat.get("name", "Active Chat")
    model_file = chat.get("model_file", "")

    st.title(chat_name)
    st.caption(f"Model: {model_file}")

    messages: list[ChatMessage] = state.chat_messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your query here..."):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        selected_profile = state.selected_profile
        with st.spinner("Thinking..."):
            response = send_message(
                chat_id=chat_id,
                message=prompt,
                profile_label=selected_profile.label if selected_profile else None,
            )

        if response is None:
            messages.pop()
            state.chat_messages = messages
            return

        messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        state.chat_messages = messages
        if chat_id:
            state.chat_histories[chat_id] = messages


def render_chat_view() -> None:
    if state.current_chat is not None:
        render_active_chat()
    else:
        _render_empty_chat_placeholder()
