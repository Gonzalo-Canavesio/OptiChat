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
    st.selectbox(
        "LLM Provider (Required)",
        options=get_available_types_llm_providers(),
        index=None,
        key="provider_type_selectbox",
        help="Choose the LLM provider for your optimization problem.",
    )
    st.text_input(
        "API Key (Required)",
        key="api_key_input",
        help="Enter your API key for the selected LLM provider.",
    )
    st.text_input(
        "Custom Label (Optional)",
        key="provider_label_input",
        help="You can provide a custom label for your LLM provider if you wish.",
    )

    def on_save_provider():
        provider = LLMProviderConfig(
            label=st.session_state.provider_label_input,
            type=st.session_state.provider_type_selectbox,
            api_key=st.session_state.api_key_input,
        )
        add_llm_provider(provider)

    st.button(
        "Save LLM Provider",
        on_click=on_save_provider,
        disabled=not (
            st.session_state.provider_type_selectbox and st.session_state.api_key_input
        ),
        use_container_width=True,
        help="Click to save your API key for the selected LLM provider.",
    )


def render_llm_profile_form():
    st.text_input(
        "Profile Label (Required)",
        key="profile_label_input",
        help="Provide a label for your LLM profile configuration.",
    )
    mode = st.segmented_control(
        "Select Mode",
        ["Simple", "Advanced"],
        default="Simple",
        key="profile_mode",
        required=True,
        width="stretch",
    )
    if mode == "Simple":
        selected_provider = st.selectbox(
            "LLM Provider (Required)",
            options=get_configured_llm_providers(),
            format_func=lambda p: f"{p.label} ({p.type})",
            index=None,
            key="simple_provider_selectbox",
            help="Choose the LLM provider for your optimization problem.",
        )
        selected_model = st.selectbox(
            "Select LLM model for the agents (Required)",
            options=get_available_llm_models(selected_provider)
            if selected_provider
            else [],
            key="simple_model_selectbox",
            index=None,
            help="Choose the LLM model for the agents.",
        )
        required_ok = st.session_state.get(
            "simple_provider_selectbox"
        ) and st.session_state.get("simple_model_selectbox")

        def on_save_profile():
            profile = ProfileConfig(
                label=st.session_state.profile_label_input,
                root_agent=AgentConfig(
                    provider_label=st.session_state.simple_provider_selectbox.label,
                    llm_model=st.session_state.simple_model_selectbox,
                ),
                expert_agent=AgentConfig(
                    provider_label=st.session_state.simple_provider_selectbox.label,
                    llm_model=st.session_state.simple_model_selectbox,
                ),
                illustrator_agent=AgentConfig(
                    provider_label=st.session_state.simple_provider_selectbox.label,
                    llm_model=st.session_state.simple_model_selectbox,
                ),
                generator_agent=AgentConfig(
                    provider_label=st.session_state.simple_provider_selectbox.label,
                    llm_model=st.session_state.simple_model_selectbox,
                ),
            )
            add_llm_profile(profile)

    elif mode == "Advanced":
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
                model_key = f"llm_model_{akp}"
                temp_key = f"temperature_{akp}"
                max_key = f"max_tokens_{akp}"

                selected_provider = st.selectbox(
                    f"LLM Provider for {agent} (Required)",
                    options=get_configured_llm_providers(),
                    format_func=lambda p: f"{p.label} ({p.type})",
                    index=None,
                    key=f"provider_selectbox_{akp}",
                    help=f"Choose the LLM provider for the {agent}.",
                )

                selected_model = st.selectbox(
                    f"Select LLM model for {agent} (Required)",
                    options=get_available_llm_models(selected_provider)
                    if selected_provider
                    else [],
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
                models_selected[akp] = selected_model

        required_ok = all(
            st.session_state.get(f"provider_selectbox_{akp}")
            and st.session_state.get(f"llm_model_{akp}")
            for akp in agent_key_prefixes
        )

        def on_save_profile():
            profile = ProfileConfig(
                label=st.session_state.profile_label_input,
                root_agent=AgentConfig(
                    provider_label=st.session_state.provider_selectbox_root_agent.label,
                    llm_model=st.session_state.llm_model_root_agent,
                    temperature=st.session_state.temperature_root_agent,
                    max_tokens=st.session_state.max_tokens_root_agent,
                ),
                expert_agent=AgentConfig(
                    provider_label=st.session_state.provider_selectbox_expert_agent.label,
                    llm_model=st.session_state.llm_model_expert_agent,
                    temperature=st.session_state.temperature_expert_agent,
                    max_tokens=st.session_state.max_tokens_expert_agent,
                ),
                illustrator_agent=AgentConfig(
                    provider_label=st.session_state.provider_selectbox_illustrator_agent.label,
                    llm_model=st.session_state.llm_model_illustrator_agent,
                    temperature=st.session_state.temperature_illustrator_agent,
                    max_tokens=st.session_state.max_tokens_illustrator_agent,
                ),
                generator_agent=AgentConfig(
                    provider_label=st.session_state.provider_selectbox_generator_agent.label,
                    llm_model=st.session_state.llm_model_generator_agent,
                    temperature=st.session_state.temperature_generator_agent,
                    max_tokens=st.session_state.max_tokens_generator_agent,
                ),
            )
            add_llm_profile(profile)
    else:
        raise ValueError(
            "Invalid mode selected. Please choose either 'Simple' or 'Advanced'."
        )

    st.space()
    st.button(
        "Save LLM Profile",
        disabled=not (st.session_state.get("profile_label_input") and required_ok),
        on_click=on_save_profile,
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
