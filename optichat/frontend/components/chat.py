from typing import Literal

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from optichat.frontend.api.client import BackendError, create_chat, send_message
from optichat.frontend.state import (
    ChatInfo,
    ChatMessage,
    LoadedModel,
    clean_session_state_for_chat_form,
    state,
)
from optichat.llm.models import ProfileConfig

ModelingLanguage = Literal["Pyomo", "GAMSPy", "gurobipy"]
AVAILABLE_MODELING_LANGUAGES: list[ModelingLanguage] = [
    "Pyomo",
    "GAMSPy",
    "gurobipy",
]


def _handle_create_chat_submission(
    model_file: UploadedFile | None,
    data_files: list[UploadedFile] | None,
    model_paper: UploadedFile | None,
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

    try:
        with st.spinner("Initializing model and starting chat..."):
            chat = create_chat(
                model_file=model_file,
                data_files=data_files,
                model_paper=model_paper,
                modeling_language=modeling_language,
                profile_label=profile.label,
                solver=solver,
                chat_name=chat_name.strip(),
            )
    except BackendError as e:
        st.error(f"Failed to create chat: {e}")
        return False

    state.current_chat = chat
    return True


@st.dialog(
    "Create New Chat", width="medium", on_dismiss=clean_session_state_for_chat_form
)
def render_create_chat_dialog() -> None:
    if state.selected_profile is None or state.selected_solver is None:
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
        model_file = st.file_uploader(
            "Model file (.py) (Required)",
            type=["py"],
        )
        data_files = st.file_uploader(
            "Data files (Optional)",
            accept_multiple_files=True,
            help="Files the model reads, in any format. They are placed next to the "
            "model file, so open them with a relative path (e.g. "
            "`open(\"my_data.json\")`).",
        )
        chat_name = st.text_input(
            "Chat name (Optional)",
            help="Leave empty for auto-generated name.",
        )
        model_paper = st.file_uploader(
            "Model paper (Optional)",
            type=["pdf", "md", "txt"],
            help="Upload a file describing your model. This can help the LLM understand your model better.",
        )
        submitted = st.form_submit_button("Create", type="primary", width="stretch")

    if submitted:
        if _handle_create_chat_submission(
            model_file=model_file,
            data_files=data_files,
            model_paper=model_paper,
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


def _describe_model(model: LoadedModel) -> str:
    status = model["status"]
    if status == "optimal" and model.get("objective") is not None:
        sense = (model.get("sense") or "").lower()
        return (
            f"- **{model['name']}** loaded and solved to optimality "
            f"({sense + ' ' if sense else ''}objective = {model['objective']:,.6g})"
        )
    return f"- **{model['name']}** loaded, but its status is *{status}*"


def _render_chat_header(chat: ChatInfo) -> None:
    st.subheader(chat["name"])


def _render_welcome(chat: ChatInfo) -> None:
    models = chat.get("models") or []
    lines = [_describe_model(m) for m in models] or ["- No model information."]
    infeasible = any(m["status"].startswith("infeasible") for m in models)
    hint = (
        "The model is infeasible: ask why, or how to restore feasibility."
        if infeasible
        else "Ask about its values, sensitivity, what-if scenarios or robustness."
    )
    with st.chat_message("assistant"):
        st.markdown("The model is ready.\n\n" + "\n".join(lines) + f"\n\n{hint}")


def render_active_chat() -> None:
    chat = state.current_chat
    if not chat:
        return

    _render_chat_header(chat)
    messages = chat.get("messages", [])
    if not messages:
        _render_welcome(chat)

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

        try:
            with st.spinner("Thinking..."):
                new_chat = send_message(
                    chat_id=chat["id"],
                    message=prompt,
                    profile_label=state.selected_profile.label,
                    solver=state.selected_solver,
                )
        except BackendError as e:
            st.error(f"Failed to send message: {e}")
            return

        state.current_chat = new_chat
        st.rerun()


def render_chat_view() -> None:
    if state.current_chat is not None:
        render_active_chat()
    else:
        _render_empty_chat_placeholder()
