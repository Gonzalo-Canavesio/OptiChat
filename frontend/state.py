from typing import Any, Literal, NotRequired, TypedDict

import streamlit as st

from llms import ProfileConfig

ViewType = Literal["chat", "settings"]
RoleType = Literal["user", "assistant"]

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
    role: RoleType
    content: str


class ChatInfo(TypedDict):
    id: str
    name: str
    model_file: NotRequired[str]
    initial_response: NotRequired[str]


class SessionStateManager:
    def initialize(self) -> None:
        if "current_view" not in st.session_state:
            st.session_state["current_view"] = "chat"
        if "current_chat" not in st.session_state:
            st.session_state["current_chat"] = None
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        if "chat_histories" not in st.session_state:
            st.session_state["chat_histories"] = {}
        if "feedback" not in st.session_state:
            st.session_state["feedback"] = []

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
    def chat_messages(self) -> list[ChatMessage]:
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        return st.session_state["chat_messages"]

    @chat_messages.setter
    def chat_messages(self, value: list[ChatMessage]) -> None:
        st.session_state["chat_messages"] = value

    @property
    def chat_histories(self) -> dict[str, list[ChatMessage]]:
        if "chat_histories" not in st.session_state:
            st.session_state["chat_histories"] = {}
        return st.session_state["chat_histories"]

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
        "simple_provider_selectbox",
        "simple_model_selectbox",
        "root_agent_provider_selectbox",
        "root_agent_model_selectbox",
        "root_agent_temperature_input",
        "root_agent_max_tokens_input",
        "expert_agent_provider_selectbox",
        "expert_agent_model_selectbox",
        "expert_agent_temperature_input",
        "expert_agent_max_tokens_input",
        "illustrator_agent_provider_selectbox",
        "illustrator_agent_model_selectbox",
        "illustrator_agent_temperature_input",
        "illustrator_agent_max_tokens_input",
        "generator_agent_provider_selectbox",
        "generator_agent_model_selectbox",
        "generator_agent_temperature_input",
        "generator_agent_max_tokens_input",
        "profile_mode",
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)


def clean_session_state_for_chat_form() -> None:
    keys_to_remove = [
        "modeling_language_selectbox",
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)
