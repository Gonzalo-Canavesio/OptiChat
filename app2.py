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
    remove_llm_provider,
    update_llm_profile,
    update_llm_provider,
)


def _feedback_queue():
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = []
    return st.session_state["feedback"]


def queue_feedback(*args, **kwargs):
    _feedback_queue().append((args, kwargs))


def _clean_old_session_state_keys(keys: list[str]):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def render_llm_provider_form(existing_provider: LLMProviderConfig | None = None):
    if existing_provider:
        st.info(
            "You can change the label for the existing provider, but the provider type "
            "and API key cannot be changed. To change the provider type or API key, "
            "please delete this provider and add a new one.",
        )
    else:
        st.info(
            "Please provide the API key for your LLM provider. You can also provide a "
            "custom label for your provider, but it is optional. If you don't provide "
            "a custom label, the label will be generated automatically.",
        )
    with st.form("llm_provider_form", border=False):
        provider_type = st.selectbox(
            "LLM Provider (Required)",
            options=get_available_types_llm_providers(),
            index=get_available_types_llm_providers().index(existing_provider.type)
            if existing_provider
            else None,
            help="Choose the LLM provider for your optimization problem.",
            disabled=existing_provider is not None,
        )
        api_key = st.text_input(
            "API Key (Required)",
            value=f"{existing_provider.api_key[:5]}...{existing_provider.api_key[-5:]}"
            if existing_provider
            else "",
            help="Enter your API key for the selected LLM provider.",
            disabled=existing_provider is not None,
        )
        label = st.text_input(
            "Custom Label (Optional)",
            value=existing_provider.label if existing_provider else "",
            help="You can provide a custom label for your LLM provider if you wish.",
        )

        if st.form_submit_button("Save", width="stretch"):
            if not provider_type or not api_key:
                st.error(
                    "Please provide both the LLM provider type and API key before saving."
                )
                return
            if existing_provider:
                provider = LLMProviderConfig(
                    label=label,
                    type=existing_provider.type,
                    api_key=existing_provider.api_key,
                )
                update_llm_provider(existing_provider, provider)
                queue_feedback(
                    f"LLM provider **{provider.label}** (before **{existing_provider.label}**) updated successfully!",
                    icon=":material/check_circle:",
                )
            else:
                provider = LLMProviderConfig(
                    label=label,
                    type=provider_type,
                    api_key=api_key,
                )
                add_llm_provider(provider)
                queue_feedback(
                    f"LLM provider **{provider.label}** added successfully!",
                    icon=":material/check_circle:",
                )
            st.rerun()


def render_llm_profile_form(existing_profile: ProfileConfig | None = None):
    configured_providers = get_configured_llm_providers()

    def get_provider_index(provider_label):
        if not provider_label:
            return None
        for i, p in enumerate(configured_providers):
            if p.label == provider_label:
                return i
        return None

    def get_model_index(provider, model_name):
        if not provider or not model_name:
            return None
        models = get_available_llm_models(provider)
        try:
            return models.index(model_name)
        except ValueError:
            return None

    default_mode = "Simple"
    if existing_profile:
        agents = [
            existing_profile.root_agent,
            existing_profile.expert_agent,
            existing_profile.illustrator_agent,
            existing_profile.generator_agent,
        ]
        first_agent = agents[0]

        # If all agents share the same provider and model, and have no temp/max_tokens, it's Simple mode
        is_simple = all(
            a.provider_label == first_agent.provider_label
            and a.llm_model == first_agent.llm_model
            and getattr(a, "temperature", None) is None
            and getattr(a, "max_tokens", None) is None
            for a in agents
        )
        default_mode = "Simple" if is_simple else "Advanced"
    st.text_input(
        "Profile Label (Required)",
        key="profile_label_input",
        help="Provide a label for your LLM profile configuration.",
    )
    mode = st.segmented_control(
        "Select Mode",
        ["Simple", "Advanced"],
        default=default_mode,
        key="profile_mode",
        required=True,
        width="stretch",
    )
    if mode == "Simple":
        default_provider_idx = None
        if existing_profile and default_mode == "Simple":
            default_provider_idx = get_provider_index(
                existing_profile.root_agent.provider_label
            )
        selected_provider = st.selectbox(
            "LLM Provider (Required)",
            options=get_configured_llm_providers(),
            format_func=lambda p: f"{p.label} ({p.type})",
            index=default_provider_idx,
            key="simple_provider_selectbox",
            help="Choose the LLM provider for your optimization problem.",
        )
        default_model_idx = None
        if existing_profile and default_mode == "Simple" and selected_provider:
            if selected_provider.label == existing_profile.root_agent.provider_label:
                default_model_idx = get_model_index(
                    selected_provider, existing_profile.root_agent.llm_model
                )
        selected_model = st.selectbox(
            "Select LLM model for the agents (Required)",
            options=get_available_llm_models(selected_provider)
            if selected_provider
            else [],
            key="simple_model_selectbox",
            index=default_model_idx,
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
            if existing_profile:
                # Assuming you have an update function available
                update_llm_profile(existing_profile, profile)
            else:
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
            existing_agent = (
                getattr(existing_profile, akp) if existing_profile else None
            )

            with current_column:
                st.subheader(f'"{agent}" Configuration')
                model_key = f"llm_model_{akp}"
                temp_key = f"temperature_{akp}"
                max_key = f"max_tokens_{akp}"
                default_provider_idx = (
                    get_provider_index(existing_agent.provider_label)
                    if existing_agent
                    else None
                )

                selected_provider = st.selectbox(
                    f"LLM Provider for {agent} (Required)",
                    options=get_configured_llm_providers(),
                    format_func=lambda p: f"{p.label} ({p.type})",
                    index=default_provider_idx,
                    key=f"provider_selectbox_{akp}",
                    help=f"Choose the LLM provider for the {agent}.",
                )

                default_model_idx = None
                if existing_agent and selected_provider:
                    if selected_provider.label == existing_agent.provider_label:
                        default_model_idx = get_model_index(
                            selected_provider, existing_agent.llm_model
                        )

                selected_model = st.selectbox(
                    f"Select LLM model for {agent} (Required)",
                    options=get_available_llm_models(selected_provider)
                    if selected_provider
                    else [],
                    index=default_model_idx,
                    key=model_key,
                    help=f"Choose the LLM model for the {agent}.",
                )
                st.number_input(
                    f"Set temperature for {agent} (Optional)",
                    min_value=0.0,
                    max_value=1.0,
                    value=getattr(existing_agent, "temperature", None)
                    if existing_agent
                    else None,
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
                    value=getattr(existing_agent, "max_tokens", None)
                    if existing_agent
                    else None,
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
            if existing_profile:
                update_llm_profile(existing_profile, profile)
            else:
                add_llm_profile(profile)
    else:
        raise ValueError(
            "Invalid mode selected. Please choose either 'Simple' or 'Advanced'."
        )

    st.space()
    if st.button(
        "Save LLM Profile",
        disabled=not (st.session_state.get("profile_label_input") and required_ok),
        width="stretch",
        help="Click to save your LLM profile configuration.",
    ):
        on_save_profile()
        st.rerun()


def render_sidebar():
    st.selectbox(
        "Select LLM Profile",
        options=get_llm_profiles(),
        format_func=lambda p: p.label,
        index=None,
        key="profile_selectbox",
        help="Choose the LLM profile for your optimization problem.",
    )
    if st.button("⚙️ Manage Providers & Profiles", width="stretch"):
        st.session_state.current_view = "settings"
        st.rerun()
    st.divider()
    if st.button("💬 Start New Chat", width="stretch"):
        pass
    example_chats = [
        "Example Chat 1",
        "Example Chat 2",
        "Example Chat 3",
    ]
    for example in example_chats:
        if st.button(example, width="stretch"):
            pass


@st.dialog("Confirm Deletion")
def render_delete_confirmation(provider):
    st.warning(
        f"Are you sure you want to delete **{provider.label}**? This action cannot be undone.",
        icon="⚠️",
    )

    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.rerun()

    with col_confirm:
        if st.button("Yes, Delete", type="primary", width="stretch"):
            remove_llm_provider(provider)
            st.rerun()


def render_settings_view():
    if st.button(
        "", type="tertiary", icon=":material/arrow_back:", help="Back to Chat"
    ):
        st.session_state.current_view = "chat"
        st.rerun()
    st.title("Settings")
    mode = st.segmented_control(
        "Settings Mode",
        ["LLM Providers", "LLM Profiles"],
        default="LLM Providers",
        key="settings_mode",
        required=True,
        width="stretch",
        label_visibility="collapsed",
    )

    if mode == "LLM Providers":
        st.text(
            "Manage your LLM providers below. You can add, edit, or delete providers."
        )
        if st.button(
            "",
            type="primary",
            help="Add a new LLM provider.",
            width="stretch",
            icon=":material/add:",
        ):
            st.dialog(title="Add New LLM Provider", width="medium")(
                render_llm_provider_form
            )()

        providers = get_configured_llm_providers()

        for provider in providers:
            with st.container(border=True):
                col_label, col_edit, col_delete = st.columns(
                    [6, 1, 1], vertical_alignment="center"
                )

                with col_label:
                    st.markdown(f"**{provider.label}** ({provider.type})")

                with col_edit:
                    if st.button(
                        "",
                        key=f"edit_btn_{provider.label}",
                        help="Edit this LLM provider's configuration.",
                        width="stretch",
                        icon=":material/edit:",
                    ):
                        st.dialog(
                            title=f"Edit LLM Provider: {provider.label}", width="medium"
                        )(render_llm_provider_form)(provider)

                with col_delete:
                    if st.button(
                        "",
                        key=f"delete_btn_{provider.label}",
                        help="Delete this LLM provider's configuration.",
                        width="stretch",
                        icon=":material/delete:",
                    ):
                        render_delete_confirmation(provider)


def render_chat_view():
    st.title("Chat")


def configure_page():
    st.set_page_config(
        page_title="OptiChat",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"


def run_toasts():
    queue = _feedback_queue()
    for args, kwargs in queue:
        st.success(*args, **kwargs)
    queue.clear()


def run_interface():
    configure_page()
    run_toasts()
    if st.session_state.current_view == "settings":
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
            title="Configure your first LLM profile", width="large", dismissible=False
        )(render_llm_profile_form)()


if __name__ == "__main__":
    run_interface()
