import streamlit as st

from frontend.state import queue_feedback
from llms import (
    LLMProviderConfig,
    add_llm_provider,
    get_available_types_llm_providers,
    get_llm_profiles_using_provider,
    remove_llm_provider,
    update_llm_provider,
)


def _display_form_instructions(is_update: bool) -> None:
    if is_update:
        st.info(
            "You can change the label for the existing provider, but the provider type "
            "and API key cannot be changed. To change the provider type or API key, "
            "please delete this provider and add a new one."
        )
    else:
        st.info(
            "Please provide the API key for your LLM provider. You can also provide a "
            "custom label for your provider, but it is optional. If you don't provide "
            "a custom label, the label will be generated automatically."
        )


def _render_form_fields(
    existing_provider: LLMProviderConfig | None,
) -> tuple[str | None, str, str]:
    available_types = get_available_types_llm_providers()

    if existing_provider and existing_provider.type in available_types:
        selected_provider_idx = available_types.index(existing_provider.type)
    else:
        selected_provider_idx = None

    provider_type = st.selectbox(
        "LLM Provider (Required)",
        options=available_types,
        index=selected_provider_idx,
        help="Choose the LLM provider for your optimization problem.",
        disabled=bool(existing_provider),
    )

    if existing_provider and len(existing_provider.api_key) >= 10:
        api_key_value = (
            f"{existing_provider.api_key[:5]}...{existing_provider.api_key[-5:]}"
        )
    elif existing_provider:
        api_key_value = existing_provider.api_key
    else:
        api_key_value = ""

    api_key = st.text_input(
        "API Key (Required)",
        value=api_key_value,
        help="Enter your API key for the selected LLM provider.",
        disabled=bool(existing_provider),
    )

    label_value = (existing_provider.label or "") if existing_provider else ""

    label = st.text_input(
        "Custom Label (Optional)",
        value=label_value,
        help="You can provide a custom label for your LLM provider if you wish.",
    )

    return provider_type, api_key, label


def _build_provider_config(
    existing_provider: LLMProviderConfig | None,
    provider_type: str,
    api_key: str,
    label: str,
) -> LLMProviderConfig:
    if existing_provider:
        return LLMProviderConfig(
            label=label,
            type=existing_provider.type,
            api_key=existing_provider.api_key,
        )
    return LLMProviderConfig(
        label=label,
        type=provider_type,
        api_key=api_key,
    )


def _handle_form_submission_llm_provider(
    existing_provider: LLMProviderConfig | None,
    provider_type: str | None,
    api_key: str,
    label: str,
) -> bool:
    if not provider_type or not api_key:
        st.error("Please provide both the LLM provider type and API key before saving.")
        return False

    provider = _build_provider_config(existing_provider, provider_type, api_key, label)

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
    return True


def render_llm_provider_form(
    existing_provider: LLMProviderConfig | None = None,
) -> None:
    is_update = existing_provider is not None
    _display_form_instructions(is_update)

    with st.form("llm_provider_form", border=False):
        provider_type, api_key, label = _render_form_fields(existing_provider)

        if st.form_submit_button("Save", type="primary", width="stretch"):
            if _handle_form_submission_llm_provider(
                existing_provider, provider_type, api_key, label
            ):
                st.rerun()


@st.dialog("Confirm Deletion LLM Provider")
def render_delete_confirmation_llm_provider(
    provider: LLMProviderConfig,
) -> None:
    st.warning(
        f"Are you sure you want to delete **{provider.label}**? "
        "This action cannot be undone."
    )

    profiles_using_provider = get_llm_profiles_using_provider(provider)
    if profiles_using_provider:
        st.markdown(
            "The following LLM profiles are using this provider and "
            "will be affected by this deletion:"
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
