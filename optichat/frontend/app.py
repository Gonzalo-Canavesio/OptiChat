import streamlit as st

from optichat.frontend.components.chat import render_chat_view
from optichat.frontend.components.settings.credential_form import (
    render_llm_credential_form,
)
from optichat.frontend.components.settings.profile_form import render_llm_profile_form
from optichat.frontend.components.settings.settings_view import render_settings_view
from optichat.frontend.components.sidebar import render_sidebar
from optichat.frontend.state import configure_page, run_toasts, state
from optichat.llm.credentials import get_llm_credentials
from optichat.llm.profiles import get_llm_profiles


def run_interface() -> None:
    configure_page()
    run_toasts()

    if state.current_view == "settings":
        render_settings_view()
    else:
        with st.sidebar:
            render_sidebar()
        render_chat_view()

    if not get_llm_credentials():
        st.dialog(title="Add your first connection", dismissible=False)(
            render_llm_credential_form
        )()
    elif not get_llm_profiles():
        st.dialog(
            title="Configure your first LLM profile",
            width="large",
            dismissible=False,
        )(render_llm_profile_form)()


if __name__ == "__main__":
    run_interface()
