import streamlit as st

from optichat.frontend.state import queue_feedback
from optichat.llm.credentials import (
    add_llm_credential,
    remove_llm_credential,
    update_llm_credential,
)
from optichat.llm.models import LLMCredential
from optichat.llm.profiles import get_llm_profiles_using_credential
from optichat.llm.providers.base import Provider, ProviderVerificationError
from optichat.llm.providers.registry import get_available_provider_names, get_provider


def _display_form_instructions(is_update: bool) -> None:
    if is_update:
        st.info(
            "You can change the label of this connection, but not its provider or "
            "API key. To change them, please delete this connection and add a new "
            "one."
        )
    else:
        st.info(
            "Choose your LLM provider and give its API key if it needs one. You can "
            "also provide a custom label for the connection, but it is optional. If "
            "you don't provide a custom label, the label will be generated "
            "automatically."
        )


def _masked(api_key: str) -> str:
    return f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) >= 10 else api_key


def _render_provider_field(existing_credential: LLMCredential | None) -> str | None:
    # Outside the form, so the fields below change as soon as a provider is chosen.
    available_providers = get_available_provider_names()

    if existing_credential and existing_credential.provider in available_providers:
        selected_provider_idx = available_providers.index(existing_credential.provider)
    else:
        selected_provider_idx = None

    return st.selectbox(
        "LLM Provider (Required)",
        options=available_providers,
        index=selected_provider_idx,
        help="Choose the LLM provider for your optimization problem.",
        disabled=bool(existing_credential),
    )


def _render_api_key_field(
    spec: Provider, existing_credential: LLMCredential | None
) -> str | None:
    """Render the API key field if `spec` needs a key; return the key typed in."""
    if not spec.api_key_required:
        st.caption("No API key needed.")
        return None

    existing_key = existing_credential.api_key if existing_credential else None
    return st.text_input(
        "API Key (Required)",
        value=_masked(existing_key) if existing_key else "",
        help=f"Enter your API key for {spec.name}.",
        disabled=bool(existing_credential),
    )


def _render_label_field(existing_credential: LLMCredential | None) -> str:
    label_value = (existing_credential.label or "") if existing_credential else ""

    return st.text_input(
        "Custom Label (Optional)",
        value=label_value,
        help="You can provide a custom label for your connection if you wish.",
    )


def _handle_form_submission_llm_credential(
    existing_credential: LLMCredential | None,
    spec: Provider | None,
    api_key: str | None,
    label: str,
) -> bool:
    if existing_credential:
        credential = existing_credential.model_copy(update={"label": label})
        update_llm_credential(existing_credential, credential)
        queue_feedback(
            f"Connection **{credential.label}** "
            f"(before **{existing_credential.label}**) updated successfully!",
            icon=":material/check_circle:",
        )
        return True

    if spec is None:
        st.error("Please choose the LLM provider before saving.")
        return False
    if spec.api_key_required and not (api_key or "").strip():
        st.error(f"Please provide the API key for {spec.name} before saving.")
        return False

    credential = LLMCredential(
        label=label, provider=spec.name, api_key=api_key.strip() if api_key else None
    )
    try:
        with st.spinner(f"Connecting to {spec.name}..."):
            add_llm_credential(credential)
    except ProviderVerificationError as e:
        st.error(f"The connection was not saved. {e}")
        return False
    queue_feedback(
        f"Connection **{credential.label}** added successfully!",
        icon=":material/check_circle:",
    )
    return True


def render_llm_credential_form(
    existing_credential: LLMCredential | None = None,
) -> None:
    is_update = existing_credential is not None
    _display_form_instructions(is_update)

    provider = _render_provider_field(existing_credential)
    spec = get_provider(provider) if provider else None

    with st.form("llm_credential_form", border=False):
        api_key = _render_api_key_field(spec, existing_credential) if spec else None
        label = _render_label_field(existing_credential)

        if st.form_submit_button("Save", type="primary", width="stretch"):
            if _handle_form_submission_llm_credential(
                existing_credential, spec, api_key, label
            ):
                st.rerun()


@st.dialog("Confirm Deletion Connection")
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
            "The following LLM profiles are using this connection and "
            "will be affected by this deletion:"
        )
        for profile in profiles_using_credential:
            st.markdown(f"- **{profile.label}**")
    else:
        st.markdown("No LLM profiles are currently using this connection.")

    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.rerun()

    with col_confirm:
        if st.button("Delete", type="primary", width="stretch"):
            remove_llm_credential(credential)
            queue_feedback(
                f"Connection **{credential.label}** deleted successfully!",
                icon=":material/check_circle:",
            )
            st.rerun()
