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
    get_llm_profiles_using_provider,
    remove_llm_profile,
    remove_llm_provider,
    update_llm_profile,
    update_llm_provider,
    valid_key_for_model,
)


def _feedback_queue():
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = []
    return st.session_state["feedback"]


def queue_feedback(*args, **kwargs):
    _feedback_queue().append((args, kwargs))


def _display_form_instructions(is_editing: bool):
    if is_editing:
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


def _render_form_fields(existing_provider: LLMProviderConfig | None):
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
    return provider_type, api_key, label


def _provider_from_form(existing_provider, provider_type, api_key, label):
    if existing_provider:
        return LLMProviderConfig(
            label=label,
            type=existing_provider.type,
            api_key=existing_provider.api_key,
        )
    else:
        return LLMProviderConfig(
            label=label,
            type=provider_type,
            api_key=api_key,
        )


def _handle_form_submission_llm_provider(
    existing_provider: LLMProviderConfig | None, provider_type, api_key, label
):
    if not provider_type or not api_key:
        st.error("Please provide both the LLM provider type and API key before saving.")
        return

    provider = _provider_from_form(existing_provider, provider_type, api_key, label)

    if existing_provider:
        update_llm_provider(existing_provider, provider)
        queue_feedback(
            f"LLM provider **{provider.label}** "
            f"(before **{existing_provider.label}**) updated successfully!",
            icon=":material/check_circle:",
        )
    else:
        add_llm_provider(provider)
        queue_feedback(
            f"LLM provider **{provider.label}** added successfully!",
            icon=":material/check_circle:",
        )


def render_llm_provider_form(existing_provider: LLMProviderConfig | None = None):
    is_editing = existing_provider is not None
    _display_form_instructions(is_editing)

    with st.form("llm_provider_form", border=False):
        provider_type, api_key, label = _render_form_fields(existing_provider)

        if st.form_submit_button("Save", width="stretch"):
            _handle_form_submission_llm_provider(
                existing_provider, provider_type, api_key, label
            )
            st.rerun()


def _get_provider_index(configured_providers, provider_label):
    if not provider_label:
        return None
    for i, p in enumerate(configured_providers):
        if p.label == provider_label:
            return i
    return None


def _get_model_index(provider, model_name):
    if not provider or not model_name:
        return None
    models = get_available_llm_models(provider)
    try:
        return models.index(model_name)
    except ValueError:
        return None


def _get_default_mode(existing_profile: ProfileConfig | None):
    if existing_profile:
        agents = [
            existing_profile.root_agent,
            existing_profile.expert_agent,
            existing_profile.illustrator_agent,
            existing_profile.generator_agent,
        ]

        is_simple = all(
            a.provider_label == agents[0].provider_label
            and a.llm_model == agents[0].llm_model
            and getattr(a, "temperature", None) is None
            and getattr(a, "max_tokens", None) is None
            for a in agents
        )
        return "Simple" if is_simple else "Advanced"
    return "Simple"


def _render_simple_mode_fields(existing_profile: ProfileConfig | None):
    is_default_simple_mode = _get_default_mode(existing_profile) == "Simple"
    default_provider_idx = None
    if existing_profile and is_default_simple_mode:
        default_provider_idx = _get_provider_index(
            get_configured_llm_providers(),
            existing_profile.root_agent.provider_label,
        )

    selected_provider = st.selectbox(
        "LLM Provider (Required)",
        options=get_configured_llm_providers(),
        format_func=lambda p: f"{p.label} ({p.type})",
        index=default_provider_idx,
        key="simple_provider_selectbox",
        help="Choose the LLM provider for your optimization problem.",
    )

    assert isinstance(selected_provider, LLMProviderConfig) or selected_provider is None

    default_model_idx = None
    if existing_profile and is_default_simple_mode and selected_provider:
        if selected_provider.label == existing_profile.root_agent.provider_label:
            default_model_idx = _get_model_index(
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

    return selected_provider, selected_model


def _render_advanced_mode_agent_fields(agent_label, existing_agent):
    default_provider_idx = (
        _get_provider_index(
            get_configured_llm_providers(), existing_agent.provider_label
        )
        if existing_agent
        else None
    )

    selected_provider = st.selectbox(
        f"LLM Provider for {agent_label} (Required)",
        options=get_configured_llm_providers(),
        format_func=lambda p: f"{p.label} ({p.type})",
        index=default_provider_idx,
        help=f"Choose the LLM provider for the {agent_label}.",
        key=f"{agent_label}_provider_selectbox",
    )

    assert isinstance(selected_provider, LLMProviderConfig) or selected_provider is None
    default_model_idx = None
    if existing_agent and selected_provider:
        if selected_provider.label == existing_agent.provider_label:
            default_model_idx = _get_model_index(
                selected_provider, existing_agent.llm_model
            )

    selected_model = st.selectbox(
        f"Select LLM model for {agent_label} (Required)",
        options=get_available_llm_models(selected_provider)
        if selected_provider
        else [],
        index=default_model_idx,
        help=f"Choose the LLM model for the {agent_label}.",
        key=f"{agent_label}_model_selectbox",
    )
    selected_temperature = st.number_input(
        f"Set temperature for {agent_label} (Optional)",
        min_value=0.0,
        max_value=1.0,
        value=getattr(existing_agent, "temperature", None) if existing_agent else None,
        step=0.1,
        help=(
            f"Set the temperature for the {agent_label}. "
            "Lower values make the model more deterministic."
        ),
        key=f"{agent_label}_temperature_input",
    )
    selected_max_tokens = st.number_input(
        f"Set max tokens for {agent_label} (Optional)",
        min_value=1,
        value=getattr(existing_agent, "max_tokens", None) if existing_agent else None,
        step=1,
        help=f"Set the maximum number of tokens for the {agent_label}.",
        key=f"{agent_label}_max_tokens_input",
    )

    return selected_provider, selected_model, selected_temperature, selected_max_tokens


def _render_advanced_mode_fields(existing_profile: ProfileConfig | None):
    agent_labels = [
        "Root Agent",
        "Expert Agent",
        "Illustrator Agent",
        "Generator Agent",
    ]
    agent_keys = [
        "root_agent",
        "expert_agent",
        "illustrator_agent",
        "generator_agent",
    ]
    col1, col2 = st.columns(2, gap="large")

    selected = {}

    for index, (agent_label) in enumerate(agent_labels):
        current_column = col1 if index < 2 else col2
        existing_agent = (
            getattr(existing_profile, agent_keys[index]) if existing_profile else None
        )

        with current_column:
            st.subheader(f'"{agent_label}" configuration')
            (
                selected_provider,
                selected_model,
                selected_temperature,
                selected_max_tokens,
            ) = _render_advanced_mode_agent_fields(agent_label, existing_agent)

            selected[agent_label] = {
                "provider": selected_provider,
                "model": selected_model,
                "temperature": selected_temperature,
                "max_tokens": selected_max_tokens,
            }

    return selected


def _handle_form_submission_llm_profile(
    mode: str,
    selected_provider,
    selected_model,
    selected,
    existing_profile: ProfileConfig | None,
    profile_label: str,
):
    if not profile_label:
        st.error("Please provide a label for your LLM profile before saving.")
        return False
    if mode == "Simple":
        if not selected_provider or not selected_model:
            st.error("Please provide both the LLM provider and model before saving.")
            return False
        if not valid_key_for_model(selected_provider, selected_model):
            st.error(
                "The selected LLM provider and model combination is invalid. "
                "Ensure that the API key is correct and that the model is available."
            )
            return False
        profile = ProfileConfig(
            label=profile_label,
            root_agent=AgentConfig(
                provider_label=selected_provider.label,
                llm_model=selected_model,
            ),
            expert_agent=AgentConfig(
                provider_label=selected_provider.label,
                llm_model=selected_model,
            ),
            illustrator_agent=AgentConfig(
                provider_label=selected_provider.label,
                llm_model=selected_model,
            ),
            generator_agent=AgentConfig(
                provider_label=selected_provider.label,
                llm_model=selected_model,
            ),
        )
    elif mode == "Advanced":
        if not selected:
            st.error("Please provide configurations for all agents before saving.")
            return False
        for agent_label, agent_config in selected.items():
            if not agent_config["provider"] or not agent_config["model"]:
                st.error(
                    f"Please provide both the LLM provider and model for "
                    f"{agent_label} before saving."
                )
                return False
            if not valid_key_for_model(agent_config["provider"], agent_config["model"]):
                st.error(
                    f"The selected LLM provider and model combination for "
                    f"{agent_label} is invalid. Ensure that the API key is correct "
                    "and that the model is available."
                )
                return False
        profile = ProfileConfig(
            label=profile_label,
            root_agent=AgentConfig(
                provider_label=selected["Root Agent"]["provider"].label,
                llm_model=selected["Root Agent"]["model"],
                temperature=selected["Root Agent"]["temperature"],
                max_tokens=selected["Root Agent"]["max_tokens"],
            ),
            expert_agent=AgentConfig(
                provider_label=selected["Expert Agent"]["provider"].label,
                llm_model=selected["Expert Agent"]["model"],
                temperature=selected["Expert Agent"]["temperature"],
                max_tokens=selected["Expert Agent"]["max_tokens"],
            ),
            illustrator_agent=AgentConfig(
                provider_label=selected["Illustrator Agent"]["provider"].label,
                llm_model=selected["Illustrator Agent"]["model"],
                temperature=selected["Illustrator Agent"]["temperature"],
                max_tokens=selected["Illustrator Agent"]["max_tokens"],
            ),
            generator_agent=AgentConfig(
                provider_label=selected["Generator Agent"]["provider"].label,
                llm_model=selected["Generator Agent"]["model"],
                temperature=selected["Generator Agent"]["temperature"],
                max_tokens=selected["Generator Agent"]["max_tokens"],
            ),
        )
    if existing_profile:
        update_llm_profile(existing_profile, profile)  # pyright: ignore[reportPossiblyUnboundVariable]
        queue_feedback(
            f"LLM profile **{profile.label}** "  # pyright: ignore[reportPossiblyUnboundVariable]
            f"(before **{existing_profile.label}**) updated successfully!",
            icon=":material/check_circle:",
        )
    else:
        add_llm_profile(profile)  # pyright: ignore[reportPossiblyUnboundVariable]
        queue_feedback(
            f"LLM profile **{profile.label}** added successfully!",  # pyright: ignore[reportPossiblyUnboundVariable]
            icon=":material/check_circle:",
        )
    return True


def _clean_session_state_for_profile_form():
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


def render_llm_profile_form(existing_profile: ProfileConfig | None = None):
    mode = st.segmented_control(
        "Select Mode",
        ["Simple", "Advanced"],
        default=_get_default_mode(existing_profile),
        key="profile_mode",
        required=True,
        width="stretch",
    )
    selected_provider, selected_model, selected = None, None, None
    if mode == "Simple":
        selected_provider, selected_model = _render_simple_mode_fields(existing_profile)
    elif mode == "Advanced":
        selected = _render_advanced_mode_fields(existing_profile)
    with st.form("llm_profile_form", border=False):
        profile_label = st.text_input(
            "Profile Label (Required)",
            help="Provide a label for your LLM profile configuration.",
            value=existing_profile.label if existing_profile else "",
        )
        if st.form_submit_button("Save", width="stretch"):
            if _handle_form_submission_llm_profile(
                mode,
                selected_provider,
                selected_model,
                selected,
                existing_profile,
                profile_label,
            ):
                _clean_session_state_for_profile_form()
                st.rerun()


def render_sidebar():
    st.selectbox(
        "Select LLM Profile",
        options=[p for p in get_llm_profiles() if p.enabled],
        format_func=lambda p: p.label,
        index=None,
        key="profile_selectbox",
        help="Choose the LLM profile for your optimization problem.",
    )
    if st.button("Manage Providers & Profiles", width="stretch"):
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


@st.dialog("Confirm Deletion LLM Provider")
def render_delete_confirmation_llm_provider(provider):
    st.warning(
        f"Are you sure you want to delete **{provider.label}**? "
        "This action cannot be undone."
    )
    profiles_using_provider = get_llm_profiles_using_provider(provider)

    if profiles_using_provider:
        st.markdown(
            "The following LLM profiles are using this provider and "
            " will be affected by this deletion:"
        )
        for profile in profiles_using_provider:
            st.markdown(f"- **{profile.label}**")
    else:
        st.markdown("No LLM profiles are currently using this provider.")
    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.rerun()

    with col_confirm:
        if st.button("Delete", type="primary", width="stretch"):
            remove_llm_provider(provider)
            queue_feedback(
                f"LLM provider **{provider.label}** deleted successfully!",
                icon=":material/check_circle:",
            )
            st.rerun()


@st.dialog("Confirm Deletion LLM Profile")
def render_delete_confirmation_llm_profile(profile):
    st.warning(
        f"Are you sure you want to delete **{profile.label}**? "
        "This action cannot be undone."
    )
    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.rerun()

    with col_confirm:
        if st.button("Delete", type="primary", width="stretch"):
            remove_llm_profile(profile)
            queue_feedback(
                f"LLM profile **{profile.label}** deleted successfully!",
                icon=":material/check_circle:",
            )
            st.rerun()


def render_settings_view():
    if st.button(
        "", type="tertiary", icon=":material/arrow_back:", help="Back to Chat"
    ):
        st.session_state.current_view = "chat"
        st.rerun()
    st.title("Manage Providers & Profiles")
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
                        render_delete_confirmation_llm_provider(provider)

    if mode == "LLM Profiles":
        st.text(
            "Manage your LLM profiles below. You can add, edit, or delete profiles."
        )
        if st.button(
            "",
            type="primary",
            help="Add a new LLM profile.",
            width="stretch",
            icon=":material/add:",
        ):
            st.dialog(
                title="Add New LLM Profile",
                width="large",
                on_dismiss=_clean_session_state_for_profile_form,
            )(render_llm_profile_form)()

        profiles = get_llm_profiles()

        for profile in profiles:
            with st.container(border=True):
                col_label, col_edit, col_delete = st.columns(
                    [6, 1, 1], vertical_alignment="center"
                )

                with col_label:
                    st.markdown(
                        f"**{profile.label}**"
                        f"{' (Incomplete)' if not profile.enabled else ''}"
                    )

                with col_edit:
                    if st.button(
                        "",
                        key=f"edit_profile_btn_{profile.label}",
                        help="Edit this LLM profile's configuration.",
                        width="stretch",
                        icon=":material/edit:",
                    ):
                        st.dialog(
                            title=f"Edit LLM Profile: {profile.label}",
                            width="large",
                            on_dismiss=_clean_session_state_for_profile_form,
                        )(render_llm_profile_form)(profile)

                with col_delete:
                    if st.button(
                        "",
                        key=f"delete_profile_btn_{profile.label}",
                        help="Delete this LLM profile's configuration.",
                        width="stretch",
                        icon=":material/delete:",
                    ):
                        render_delete_confirmation_llm_profile(profile)


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
