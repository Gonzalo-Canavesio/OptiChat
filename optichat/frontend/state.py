from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict, get_args

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from optichat.llm.models import ProfileConfig

ViewType = Literal["chat", "settings"]

PyomoSolver = Literal["GLPK", "CBC", "HiGHS", "Gurobi", "CPLEX", "SCIP"]
AVAILABLE_SOLVERS = get_args(PyomoSolver)

Feedback = Callable[[], object]
FeedbackSlot = Literal["page", "sidebar", "chat"]


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
            st.session_state["feedback"] = {slot: [] for slot in get_args(FeedbackSlot)}
        if "pending_prompt" not in st.session_state:
            st.session_state["pending_prompt"] = None
        if "pending_chat" not in st.session_state:
            st.session_state["pending_chat"] = None
        if "main_view_id" not in st.session_state:
            st.session_state["main_view_id"] = None
        if "main_view_slot" not in st.session_state:
            st.session_state["main_view_slot"] = 0

    @property
    def current_view(self) -> ViewType:
        return st.session_state["current_view"]

    @current_view.setter
    def current_view(self, value: ViewType) -> None:
        st.session_state["current_view"] = value

    @property
    def current_chat(self) -> ChatInfo | None:
        return st.session_state["current_chat"]

    @current_chat.setter
    def current_chat(self, value: ChatInfo | None) -> None:
        st.session_state["current_chat"] = value

    @property
    def pending_prompt(self) -> str | None:
        return st.session_state["pending_prompt"]

    @pending_prompt.setter
    def pending_prompt(self, value: str | None) -> None:
        st.session_state["pending_prompt"] = value

    @property
    def pending_chat(self) -> PendingChat | None:
        return st.session_state["pending_chat"]

    @pending_chat.setter
    def pending_chat(self, value: PendingChat | None) -> None:
        st.session_state["pending_chat"] = value

    @property
    def is_waiting_response(self) -> bool:
        return self.pending_prompt is not None or self.pending_chat is not None

    @property
    def main_view_id(self) -> str | None:
        return st.session_state["main_view_id"]

    @main_view_id.setter
    def main_view_id(self, value: str | None) -> None:
        st.session_state["main_view_id"] = value

    @property
    def main_view_slot(self) -> int:
        return st.session_state["main_view_slot"]

    @main_view_slot.setter
    def main_view_slot(self, value: int) -> None:
        st.session_state["main_view_slot"] = value

    @property
    def selected_profile(self) -> ProfileConfig | None:
        return st.session_state.get("profile_selectbox")

    @selected_profile.setter
    def selected_profile(self, value: ProfileConfig | None) -> None:
        st.session_state["profile_selectbox"] = value

    @property
    def selected_solver(self) -> PyomoSolver | None:
        return st.session_state.get("solver_selectbox")

    @selected_solver.setter
    def selected_solver(self, value: PyomoSolver | None) -> None:
        st.session_state["solver_selectbox"] = value


state = SessionStateManager()


def configure_page() -> None:
    st.set_page_config(
        page_title="OptiChat",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    state.initialize()


def queue_feedback(feedback: Feedback, slot: FeedbackSlot) -> None:
    st.session_state["feedback"][slot].append(feedback)


def show_feedback(slot: FeedbackSlot) -> None:
    feedback_queue = st.session_state["feedback"][slot]
    for feedback in feedback_queue:
        feedback()
    feedback_queue.clear()


def main_view_container(view_id: str) -> DeltaGenerator:
    # A container that replaces another keeps the old children, and the ones the new
    # view does not overwrite are only removed when the run ends, so a shorter view
    # (e.g. a chat with fewer messages) would briefly show the end of the previous
    # one under it. A new view takes the other of two slots, and the slot it leaves
    # gets an empty element, which drops the previous view at once.
    if state.main_view_id != view_id:
        state.main_view_id = view_id
        state.main_view_slot = 1 - state.main_view_slot
    slots = [
        st.container() if slot == state.main_view_slot else st.empty()
        for slot in range(2)
    ]
    return slots[state.main_view_slot]
