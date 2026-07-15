import streamlit as st

from llms import (
    LLMProviderConfig,
    add_llm_provider,
    get_available_types_llm_providers,
    get_configured_llm_providers,
)


def render_llm_provider_form():
    st.header("Enter your LLM provider credentials")
    st.selectbox(
        "LLM Provider (Required)",
        options=get_available_types_llm_providers(),
        index=None,
        key="provider_type_selectbox",
        help="Choose the LLM provider for your optimization problem.",
    )
    st.text_input(
        "Custom Label (Optional)",
        key="provider_label_input",
        help="You can provide a custom label for your LLM provider if you wish.",
    )
    st.text_input(
        "API Key (Required)",
        key="api_key_input",
        help="Enter your API key for the selected LLM provider.",
    )

    def on_save_provider():
        provider = LLMProviderConfig(
            label=st.session_state.provider_label_input,
            type=st.session_state.provider_type_selectbox,
            api_key=st.session_state.api_key_input,
        )
        add_llm_provider(provider)

    st.button(
        "Save API Key",
        on_click=on_save_provider,
        disabled=not (
            st.session_state.provider_type_selectbox and st.session_state.api_key_input
        ),
        use_container_width=True,
        help="Click to save your API key for the selected LLM provider.",
    )


def configure_page():
    st.set_page_config(
        page_title="OptiChat",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def run_interface():
    configure_page()
    if not get_configured_llm_providers():
        render_llm_provider_form()
    else:
        st.title("OptiChat: Optimization Chat Interface")


if __name__ == "__main__":
    run_interface()
