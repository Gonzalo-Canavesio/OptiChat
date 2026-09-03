import streamlit as st

from frontend.components.settings.profile_form import (
    render_delete_confirmation_llm_profile,
    render_llm_profile_form,
)
from frontend.components.settings.provider_form import (
    render_delete_confirmation_llm_provider,
    render_llm_provider_form,
)
from frontend.state import clean_session_state_for_profile_form, state
from llms import (
    LLMProviderConfig,
    ProfileConfig,
    get_configured_llm_providers,
    get_llm_profiles,
)


def _render_provider_item(provider: LLMProviderConfig) -> None:
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
                    title=f"Edit LLM Provider: {provider.label}",
                    width="medium",
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


def _render_providers_section() -> None:
    st.text("Manage your LLM providers below. You can add, edit, or delete providers.")
    if st.button(
        "New Provider",
        type="primary",
        help="Add a new LLM provider.",
        width="stretch",
        icon=":material/add:",
    ):
        st.dialog(title="Add New LLM Provider", width="medium")(
            render_llm_provider_form
        )()

    for provider in get_configured_llm_providers():
        _render_provider_item(provider)


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
        _render_providers_section()
    elif mode == "LLM Profiles":
        _render_profiles_section()
