from typing import Any

import streamlit as st

from frontend.state import clean_session_state_for_profile_form, queue_feedback
from llms import (
    AgentConfig,
    LLMProviderConfig,
    ProfileConfig,
    add_llm_profile,
    get_available_llm_models,
    get_configured_llm_providers,
    remove_llm_profile,
    update_llm_profile,
    valid_key_for_model,
)


def _get_provider_index(
    configured_providers: list[LLMProviderConfig], provider_label: str | None
) -> int | None:
    if not provider_label:
        return None
    for i, p in enumerate(configured_providers):
        if p.label == provider_label:
            return i
    return None


def _get_model_index(
    provider: LLMProviderConfig | None, model_name: str | None
) -> int | None:
    if not provider or not model_name:
        return None
    models = get_available_llm_models(provider)
    try:
        return models.index(model_name)
    except ValueError:
        return None


def _infer_profile_mode(existing_profile: ProfileConfig | None) -> str:
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


def _render_simple_mode_fields(
    existing_profile: ProfileConfig | None,
) -> tuple[LLMProviderConfig | None, str | None]:
    is_simple_profile = _infer_profile_mode(existing_profile) == "Simple"

    if existing_profile and is_simple_profile:
        initial_provider_idx = _get_provider_index(
            get_configured_llm_providers(),
            existing_profile.root_agent.provider_label,
        )
    else:
        initial_provider_idx = None

    selected_provider = st.selectbox(
        "LLM Provider (Required)",
        options=get_configured_llm_providers(),
        format_func=lambda p: f"{p.label} ({p.type})",
        index=initial_provider_idx,
        key="simple_provider_selectbox",
        help="Choose the LLM provider for your optimization problem.",
    )

    assert isinstance(selected_provider, LLMProviderConfig) or selected_provider is None

    has_matching_provider = bool(
        existing_profile
        and selected_provider
        and selected_provider.label == existing_profile.root_agent.provider_label
    )
    if has_matching_provider and is_simple_profile and existing_profile:
        initial_model_idx = _get_model_index(
            selected_provider, existing_profile.root_agent.llm_model
        )
    else:
        initial_model_idx = None

    selected_model = st.selectbox(
        "Select LLM model for the agents (Required)",
        options=get_available_llm_models(selected_provider)
        if selected_provider
        else [],
        key="simple_model_selectbox",
        index=initial_model_idx,
        help="Choose the LLM model for the agents.",
    )

    return selected_provider, selected_model


def _render_advanced_mode_agent_fields(
    agent_label: str, existing_agent: AgentConfig | None
) -> tuple[LLMProviderConfig | None, str | None, float | None, int | None]:
    initial_provider_idx = (
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
        index=initial_provider_idx,
        help=f"Choose the LLM provider for the {agent_label}.",
        key=f"{agent_label}_provider_selectbox",
    )

    assert isinstance(selected_provider, LLMProviderConfig) or selected_provider is None

    has_matching_provider = bool(
        existing_agent
        and selected_provider
        and selected_provider.label == existing_agent.provider_label
    )
    if has_matching_provider and existing_agent:
        initial_model_idx = _get_model_index(
            selected_provider, existing_agent.llm_model
        )
    else:
        initial_model_idx = None

    selected_model = st.selectbox(
        f"Select LLM model for {agent_label} (Required)",
        options=get_available_llm_models(selected_provider)
        if selected_provider
        else [],
        index=initial_model_idx,
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


def _render_advanced_mode_fields(
    existing_profile: ProfileConfig | None,
) -> dict[str, dict[str, Any]]:
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
    selected_dict: dict[str, dict[str, Any]] = {}

    for index, agent_label in enumerate(agent_labels):
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

            selected_dict[agent_label] = {
                "provider": selected_provider,
                "model": selected_model,
                "temperature": selected_temperature,
                "max_tokens": selected_max_tokens,
            }

    return selected_dict


def _build_simple_profile(
    profile_label: str,
    selected_provider: LLMProviderConfig | None,
    selected_model: str | None,
) -> ProfileConfig | None:
    if not selected_provider or not selected_model:
        st.error("Please provide both the LLM provider and model before saving.")
        return None

    if not valid_key_for_model(selected_provider, selected_model):
        st.error(
            "The selected LLM provider and model combination is invalid. "
            "Ensure that the API key is correct and that the model is available."
        )
        return None

    agent_config = AgentConfig(
        provider_label=selected_provider.label,
        llm_model=selected_model,
    )
    return ProfileConfig(
        label=profile_label,
        root_agent=agent_config,
        expert_agent=agent_config,
        illustrator_agent=agent_config,
        generator_agent=agent_config,
    )


def _build_advanced_profile(
    profile_label: str,
    selected_dict: dict[str, dict[str, Any]] | None,
) -> ProfileConfig | None:
    if not selected_dict:
        st.error("Please provide configurations for all agents before saving.")
        return None

    for agent_label, agent_config in selected_dict.items():
        provider = agent_config.get("provider")
        model = agent_config.get("model")

        if not provider or not model:
            st.error(
                f"Please provide both the LLM provider and model for {agent_label} "
                "before saving."
            )
            return None

        if not valid_key_for_model(provider, model):
            st.error(
                f"The selected LLM provider and model combination for {agent_label} is "
                "invalid. Ensure that the API key is correct and that the model is "
                "available."
            )
            return None

    def _make_agent(name: str) -> AgentConfig:
        cfg = selected_dict[name]
        return AgentConfig(
            provider_label=cfg["provider"].label,
            llm_model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )

    return ProfileConfig(
        label=profile_label,
        root_agent=_make_agent("Root Agent"),
        expert_agent=_make_agent("Expert Agent"),
        illustrator_agent=_make_agent("Illustrator Agent"),
        generator_agent=_make_agent("Generator Agent"),
    )


def _save_profile_and_notify(
    existing_profile: ProfileConfig | None,
    new_profile: ProfileConfig,
) -> None:
    if existing_profile:
        update_llm_profile(existing_profile, new_profile)
        queue_feedback(
            f"LLM profile **{new_profile.label}** (before **{existing_profile.label}**)"
            " updated successfully!",
            icon=":material/check_circle:",
        )
    else:
        add_llm_profile(new_profile)
        queue_feedback(
            f"LLM profile **{new_profile.label}** added successfully!",
            icon=":material/check_circle:",
        )


def _handle_form_submission_llm_profile(
    mode: str,
    selected_provider: LLMProviderConfig | None,
    selected_model: str | None,
    selected_dict: dict[str, dict[str, Any]] | None,
    existing_profile: ProfileConfig | None,
    profile_label: str,
) -> bool:
    if not profile_label:
        st.error("Please provide a label for your LLM profile before saving.")
        return False

    profile = None
    if mode == "Simple":
        profile = _build_simple_profile(
            profile_label, selected_provider, selected_model
        )
    elif mode == "Advanced":
        profile = _build_advanced_profile(profile_label, selected_dict)

    if not profile:
        return False

    _save_profile_and_notify(existing_profile, profile)
    return True


def render_llm_profile_form(
    existing_profile: ProfileConfig | None = None,
) -> None:
    mode = st.segmented_control(
        "Select Mode",
        ["Simple", "Advanced"],
        default=_infer_profile_mode(existing_profile),
        key="profile_mode",
        required=True,
        width="stretch",
    )
    selected_provider, selected_model, selected_dict = None, None, None
    if mode == "Simple":
        selected_provider, selected_model = _render_simple_mode_fields(existing_profile)
    elif mode == "Advanced":
        selected_dict = _render_advanced_mode_fields(existing_profile)

    with st.form("llm_profile_form", border=False):
        profile_label = st.text_input(
            "Profile Label (Required)",
            help="Provide a label for your LLM profile configuration.",
            value=existing_profile.label if existing_profile else "",
        )
        if st.form_submit_button("Save", type="primary", width="stretch"):
            if _handle_form_submission_llm_profile(
                mode,
                selected_provider,
                selected_model,
                selected_dict,
                existing_profile,
                profile_label,
            ):
                clean_session_state_for_profile_form()
                st.rerun()


@st.dialog("Confirm Deletion LLM Profile")
def render_delete_confirmation_llm_profile(profile: ProfileConfig) -> None:
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
