import streamlit as st

from frontend.components.chat import render_chat_view
from frontend.components.settings.profile_form import render_llm_profile_form
from frontend.components.settings.provider_form import render_llm_provider_form
from frontend.components.settings.settings_view import render_settings_view
from frontend.components.sidebar import render_sidebar
from frontend.state import configure_page, run_toasts, state
from llms import get_configured_llm_providers, get_llm_profiles


def run_interface() -> None:
    configure_page()
    run_toasts()

    if state.current_view == "settings":
        render_settings_view()
    else:
        with st.sidebar:
            render_sidebar()
        render_chat_view()

    if not get_configured_llm_providers():
        st.dialog(title="Configure your first LLM provider", dismissible=False)(
            render_llm_provider_form
        )()
    elif not get_llm_profiles():
        st.dialog(
            title="Configure your first LLM profile",
            width="large",
            dismissible=False,
        )(render_llm_profile_form)()


if __name__ == "__main__":
    run_interface()
