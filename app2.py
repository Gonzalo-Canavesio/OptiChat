import streamlit as st

from llms import (
    AgentConfig,
    LLMProviderConfig,
    ProfileConfig,
    add_llm_profile,
    add_llm_provider,
    get_available_llm_models,
    get_available_types_llm_providers,
    get_configured_llm_providers,
    get_llm_profiles,
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


def render_llm_profile_form(key_prefix="", dialog_mode=False):
    st.header("Configure your LLM profile")

    label_key = f"{key_prefix}profile_label_input"
    st.text_input(
        "Profile Label (Required)",
        key=label_key,
        help="Provide a label for your LLM profile configuration.",
    )

    agents = ["Root Agent", "Expert Agent", "Illustrator Agent", "Generator Agent"]
    agent_key_prefixes = [
        "root_agent",
        "expert_agent",
        "illustrator_agent",
        "generator_agent",
    ]

    col1, col2 = st.columns(2, gap="large")

    models_selected = {}

    for index, (agent, akp) in enumerate(zip(agents, agent_key_prefixes)):
        current_column = col1 if index < 2 else col2

        with current_column:
            st.subheader(f'"{agent}" Configuration')
            model_key = f"{key_prefix}llm_model_{akp}"
            temp_key = f"{key_prefix}temperature_{akp}"
            max_key = f"{key_prefix}max_tokens_{akp}"

            model_sel = st.selectbox(
                f"Select LLM model for {agent} (Required)",
                options=get_available_llm_models(),
                index=None,
                key=model_key,
                help=f"Choose the LLM model for the {agent}.",
            )
            st.number_input(
                f"Set temperature for {agent} (Optional)",
                min_value=0.0,
                max_value=1.0,
                value=None,
                step=0.1,
                key=temp_key,
                help=(
                    f"Set the temperature for the {agent}. "
                    "Lower values make the model more deterministic."
                ),
            )
            st.number_input(
                f"Set max tokens for {agent} (Optional)",
                min_value=1,
                value=None,
                step=1,
                key=max_key,
                help=f"Set the maximum number of tokens for the {agent}.",
            )
            models_selected[akp] = model_sel

    all_models_selected = all(models_selected.values())

    def on_save_profile():
        profile = ProfileConfig(
            label=st.session_state[label_key] or "Default Profile",
            root_agent=AgentConfig(
                llm_model=st.session_state[f"{key_prefix}llm_model_root_agent"],
                temperature=st.session_state[f"{key_prefix}temperature_root_agent"],
                max_tokens=st.session_state[f"{key_prefix}max_tokens_root_agent"],
            ),
            expert_agent=AgentConfig(
                llm_model=st.session_state[f"{key_prefix}llm_model_expert_agent"],
                temperature=st.session_state[f"{key_prefix}temperature_expert_agent"],
                max_tokens=st.session_state[f"{key_prefix}max_tokens_expert_agent"],
            ),
            illustrator_agent=AgentConfig(
                llm_model=st.session_state[f"{key_prefix}llm_model_illustrator_agent"],
                temperature=st.session_state[
                    f"{key_prefix}temperature_illustrator_agent"
                ],
                max_tokens=st.session_state[
                    f"{key_prefix}max_tokens_illustrator_agent"
                ],
            ),
            generator_agent=AgentConfig(
                llm_model=st.session_state[f"{key_prefix}llm_model_generator_agent"],
                temperature=st.session_state[
                    f"{key_prefix}temperature_generator_agent"
                ],
                max_tokens=st.session_state[f"{key_prefix}max_tokens_generator_agent"],
            ),
        )
        add_llm_profile(profile)
        if dialog_mode:
            st.session_state.selected_profile_label = (
                st.session_state[label_key] or "Default Profile"
            )
            st.rerun()

    st.space()
    st.button(
        "Save LLM Profile",
        on_click=on_save_profile,
        disabled=not (st.session_state[label_key] and all_models_selected),
        use_container_width=True,
        help="Click to save your LLM profile configuration.",
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
    elif not get_llm_profiles():
        render_llm_profile_form()
    else:
        st.title("OptiChat: Optimization Chat Interface")


if __name__ == "__main__":
    run_interface()
