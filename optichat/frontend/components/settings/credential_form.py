import streamlit as st

from optichat.frontend.state import queue_feedback
from optichat.llm.credentials import (
    add_llm_credential,
    remove_llm_credential,
    update_llm_credential,
)
from optichat.llm.models import LLMCredential
from optichat.llm.profiles import get_llm_profiles_using_credential
from optichat.llm.providers import (
    ProviderVerificationError,
    get_available_provider_names,
)


def _display_form_instructions(is_update: bool) -> None:
    if is_update:
        st.info(
            "You can change the label of this API key, but not its provider or the "
            "key itself. To change them, please delete this API key and add a new "
            "one."
        )
    else:
        st.info(
            "Please provide the API key for your LLM provider. You can also provide a "
            "custom label for your API key, but it is optional. If you don't provide "
            "a custom label, the label will be generated automatically."
        )


def _render_form_fields(
    existing_credential: LLMCredential | None,
) -> tuple[str | None, str, str]:
    available_providers = get_available_provider_names()

    if existing_credential and existing_credential.provider in available_providers:
        selected_provider_idx = available_providers.index(existing_credential.provider)
    else:
        selected_provider_idx = None

    provider = st.selectbox(
        "LLM Provider (Required)",
        options=available_providers,
        index=selected_provider_idx,
        help="Choose the LLM provider for your optimization problem.",
        disabled=bool(existing_credential),
    )

    if existing_credential and len(existing_credential.api_key) >= 10:
        api_key_value = (
            f"{existing_credential.api_key[:5]}...{existing_credential.api_key[-5:]}"
        )
    elif existing_credential:
        api_key_value = existing_credential.api_key
    else:
        api_key_value = ""

    api_key = st.text_input(
        "API Key (Required)",
        value=api_key_value,
        help="Enter your API key for the selected LLM provider.",
        disabled=bool(existing_credential),
    )

    label_value = (existing_credential.label or "") if existing_credential else ""

    label = st.text_input(
        "Custom Label (Optional)",
        value=label_value,
        help="You can provide a custom label for your API key if you wish.",
    )

    return provider, api_key, label


def _build_credential(
    existing_credential: LLMCredential | None,
    provider: str,
    api_key: str,
    label: str,
) -> LLMCredential:
    if existing_credential:
        return LLMCredential(
            label=label,
            provider=existing_credential.provider,
            api_key=existing_credential.api_key,
        )
    return LLMCredential(
        label=label,
        provider=provider,
        api_key=api_key,
    )


def _handle_form_submission_llm_credential(
    existing_credential: LLMCredential | None,
    provider: str | None,
    api_key: str,
    label: str,
) -> bool:
    if not provider or not api_key:
        st.error("Please provide both the LLM provider and API key before saving.")
        return False

    credential = _build_credential(existing_credential, provider, api_key, label)

    if existing_credential:
        update_llm_credential(existing_credential, credential)
        queue_feedback(
            f"API key **{credential.label}** "
            f"(before **{existing_credential.label}**) updated successfully!",
            icon=":material/check_circle:",
        )
    else:
        try:
            with st.spinner("Verifying API key..."):
                add_llm_credential(credential)
        except ProviderVerificationError as e:
            st.error(f"The API key was not saved. {e}")
            return False
        queue_feedback(
            f"API key **{credential.label}** added successfully!",
            icon=":material/check_circle:",
        )
    return True


def render_llm_credential_form(
    existing_credential: LLMCredential | None = None,
) -> None:
    is_update = existing_credential is not None
    _display_form_instructions(is_update)

    with st.form("llm_credential_form", border=False):
        provider, api_key, label = _render_form_fields(existing_credential)

        if st.form_submit_button("Save", type="primary", width="stretch"):
            if _handle_form_submission_llm_credential(
                existing_credential, provider, api_key, label
            ):
                st.rerun()


@st.dialog("Confirm Deletion API Key")
def render_delete_confirmation_llm_credential(
    credential: LLMCredential,
) -> None:
    st.warning(
        f"Are you sure you want to delete **{credential.label}**? "
        "This action cannot be undone."
    )

    profiles_using_credential = get_llm_profiles_using_credential(credential)
    if profiles_using_credential:
        st.markdown(
            "The following LLM profiles are using this API key and "
            "will be affected by this deletion:"
        )
        for profile in profiles_using_credential:
            st.markdown(f"- **{profile.label}**")
    else:
        st.markdown("No LLM profiles are currently using this API key.")

    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.rerun()

    with col_confirm:
        if st.button("Delete", type="primary", width="stretch"):
            remove_llm_credential(credential)
            queue_feedback(
                f"API key **{credential.label}** deleted successfully!",
                icon=":material/check_circle:",
            )
            st.rerun()
