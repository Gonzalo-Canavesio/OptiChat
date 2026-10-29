from typing import Any, Literal, NotRequired, TypedDict

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from optichat.llm.models import ProfileConfig

ViewType = Literal["chat", "settings"]

PyomoSolver = Literal["GLPK", "CBC", "HiGHS", "Gurobi", "CPLEX", "SCIP"]
AVAILABLE_SOLVERS: list[PyomoSolver] = [
    "GLPK",
    "CBC",
    "HiGHS",
    "Gurobi",
    "CPLEX",
    "SCIP",
]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    thought: NotRequired[str | None]


class ChatInfo(TypedDict):
    id: str
    name: str
    messages: NotRequired[list[ChatMessage]]


class PendingChat(TypedDict):
    model_file: UploadedFile
    data_files: list[UploadedFile] | None
    model_paper: UploadedFile | None
    profile_label: str
    solver: str
    chat_name: str


class SessionStateManager:
    def initialize(self) -> None:
        if "current_view" not in st.session_state:
            st.session_state["current_view"] = "chat"
        if "current_chat" not in st.session_state:
            st.session_state["current_chat"] = None
        if "feedback" not in st.session_state:
            st.session_state["feedback"] = []
        if "pending_prompt" not in st.session_state:
            st.session_state["pending_prompt"] = None
        if "pending_chat" not in st.session_state:
            st.session_state["pending_chat"] = None
        if "send_error" not in st.session_state:
            st.session_state["send_error"] = None

    @property
    def current_view(self) -> ViewType:
        return st.session_state.get("current_view", "chat")

    @current_view.setter
    def current_view(self, value: ViewType) -> None:
        st.session_state["current_view"] = value

    @property
    def current_chat(self) -> ChatInfo | None:
        return st.session_state.get("current_chat", None)

    @current_chat.setter
    def current_chat(self, value: ChatInfo | None) -> None:
        st.session_state["current_chat"] = value

    @property
    def pending_prompt(self) -> str | None:
        return st.session_state.get("pending_prompt", None)

    @pending_prompt.setter
    def pending_prompt(self, value: str | None) -> None:
        st.session_state["pending_prompt"] = value

    @property
    def pending_chat(self) -> PendingChat | None:
        return st.session_state.get("pending_chat", None)

    @pending_chat.setter
    def pending_chat(self, value: PendingChat | None) -> None:
        st.session_state["pending_chat"] = value

    @property
    def is_waiting_response(self) -> bool:
        return self.pending_prompt is not None or self.pending_chat is not None

    @property
    def send_error(self) -> str | None:
        return st.session_state.get("send_error", None)

    @send_error.setter
    def send_error(self, value: str | None) -> None:
        st.session_state["send_error"] = value

    @property
    def selected_profile(self) -> ProfileConfig | None:
        return st.session_state.get("profile_selectbox", None)

    @selected_profile.setter
    def selected_profile(self, value: ProfileConfig | None) -> None:
        st.session_state["profile_selectbox"] = value

    @property
    def selected_solver(self) -> PyomoSolver:
        return st.session_state.get("solver_selectbox", "glpk")

    @selected_solver.setter
    def selected_solver(self, value: PyomoSolver) -> None:
        st.session_state["solver_selectbox"] = value


state = SessionStateManager()


def configure_page() -> None:
    st.set_page_config(
        page_title="OptiChat",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    state.initialize()


def queue_feedback(*args: Any, **kwargs: Any) -> None:
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = []
    st.session_state["feedback"].append((args, kwargs))


def run_toasts() -> None:
    feedback_queue: list[tuple[tuple[Any, ...], dict[str, Any]]] = st.session_state.get(
        "feedback", []
    )
    for args, kwargs in feedback_queue:
        st.success(*args, **kwargs)
    feedback_queue.clear()


def clean_session_state_for_profile_form() -> None:
    keys_to_remove = [
        "simple_credential_selectbox",
        "simple_model_selectbox",
        "root_agent_credential_selectbox",
        "root_agent_model_selectbox",
        "root_agent_temperature_input",
        "root_agent_max_tokens_input",
        "expert_agent_credential_selectbox",
        "expert_agent_model_selectbox",
        "expert_agent_temperature_input",
        "expert_agent_max_tokens_input",
        "illustrator_agent_credential_selectbox",
        "illustrator_agent_model_selectbox",
        "illustrator_agent_temperature_input",
        "illustrator_agent_max_tokens_input",
        "generator_agent_credential_selectbox",
        "generator_agent_model_selectbox",
        "generator_agent_temperature_input",
        "generator_agent_max_tokens_input",
        "profile_mode",
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)
