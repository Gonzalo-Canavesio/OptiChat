import streamlit as st

from optichat.frontend.components.settings.credential_form import (
    render_delete_confirmation_llm_credential,
    render_llm_credential_form,
)
from optichat.frontend.components.settings.profile_form import (
    render_delete_confirmation_llm_profile,
    render_llm_profile_form,
)
from optichat.frontend.state import clean_session_state_for_profile_form, state
from optichat.llm.credentials import get_llm_credentials
from optichat.llm.models import LLMCredential, ProfileConfig
from optichat.llm.profiles import get_llm_profiles


def _render_credential_item(credential: LLMCredential) -> None:
    with st.container(border=True):
        col_label, col_edit, col_delete = st.columns(
            [6, 1, 1], vertical_alignment="center"
        )
        with col_label:
            st.markdown(f"**{credential.label}** ({credential.provider})")

        with col_edit:
            if st.button(
                "",
                key=f"edit_credential_btn_{credential.label}",
                help="Edit this API key's label.",
                width="stretch",
                icon=":material/edit:",
            ):
                st.dialog(
                    title=f"Edit API Key: {credential.label}",
                    width="medium",
                )(render_llm_credential_form)(credential)

        with col_delete:
            if st.button(
                "",
                key=f"delete_credential_btn_{credential.label}",
                help="Delete this API key.",
                width="stretch",
                icon=":material/delete:",
            ):
                render_delete_confirmation_llm_credential(credential)


def _render_credentials_section() -> None:
    st.text("Manage your API keys below. You can add, edit, or delete API keys.")
    if st.button(
        "New API Key",
        type="primary",
        help="Add a new API key.",
        width="stretch",
        icon=":material/add:",
    ):
        st.dialog(title="Add New API Key", width="medium")(render_llm_credential_form)()

    for credential in get_llm_credentials():
        _render_credential_item(credential)


def _render_profile_item(profile: ProfileConfig) -> None:
    with st.container(border=True):
        col_label, col_edit, col_delete = st.columns(
            [6, 1, 1], vertical_alignment="center"
        )
        with col_label:
            status_suffix = "" if profile.enabled else " (Incomplete)"
            st.markdown(f"**{profile.label}**{status_suffix}")

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
                    on_dismiss=clean_session_state_for_profile_form,
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


def _render_profiles_section() -> None:
    st.text("Manage your LLM profiles below. You can add, edit, or delete profiles.")
    if st.button(
        "New Profile",
        type="primary",
        help="Add a new LLM profile.",
        width="stretch",
        icon=":material/add:",
    ):
        st.dialog(
            title="Add New LLM Profile",
            width="large",
            on_dismiss=clean_session_state_for_profile_form,
        )(render_llm_profile_form)()

    for profile in get_llm_profiles():
        _render_profile_item(profile)


def render_settings_view() -> None:
    if st.button(
        "", type="tertiary", icon=":material/arrow_back:", help="Back to Chat"
    ):
        state.current_view = "chat"
        st.rerun()

    st.title("Manage API Keys & Profiles")
    mode = st.segmented_control(
        "Settings Mode",
        ["API Keys", "LLM Profiles"],
        default="API Keys",
        key="settings_mode",
        required=True,
        width="stretch",
        label_visibility="collapsed",
    )

    if mode == "API Keys":
        _render_credentials_section()
    elif mode == "LLM Profiles":
        _render_profiles_section()
