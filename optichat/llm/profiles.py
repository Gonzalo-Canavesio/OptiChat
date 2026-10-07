from optichat.llm.models import LLMProviderConfig, ProfileConfig
from optichat.llm.store import (
    PROFILES_FILE,
    generate_valid_label,
    load_items,
    save_items,
)

_AGENT_FIELDS = ("root_agent", "expert_agent", "illustrator_agent", "generator_agent")


def _get_llm_profiles_from_file() -> list[ProfileConfig]:
    return [ProfileConfig(**p) for p in load_items(PROFILES_FILE, "profiles")]


def _save_configured_llm_profiles_to_file(profiles: list[ProfileConfig]):
    save_items(PROFILES_FILE, "profiles", [p.model_dump() for p in profiles])


def update_provider_label_in_profiles(old_label: str, new_label: str):
    profiles = _get_llm_profiles_from_file()
    for profile in profiles:
        for field in _AGENT_FIELDS:
            agent = getattr(profile, field)
            if agent.provider_label == old_label:
                agent.provider_label = new_label

    _save_configured_llm_profiles_to_file(profiles)


def remove_provider_from_profiles(provider_label: str):
    profiles = _get_llm_profiles_from_file()
    for profile in profiles:
        for field in _AGENT_FIELDS:
            agent = getattr(profile, field)
            if agent.provider_label == provider_label:
                agent.provider_label = ""
                agent.llm_model = ""

    _save_configured_llm_profiles_to_file(profiles)


def get_llm_profiles() -> list[ProfileConfig]:
    return _get_llm_profiles_from_file()


def _get_valid_label_for_profile(profile: ProfileConfig) -> str:
    profiles = _get_llm_profiles_from_file()
    existing_labels = {p.label for p in profiles if p.label is not None}
    return generate_valid_label(profile.label, existing_labels)


def add_llm_profile(profile: ProfileConfig):
    profile.label = _get_valid_label_for_profile(profile)

    profiles = _get_llm_profiles_from_file()
    profiles.append(profile)

    _save_configured_llm_profiles_to_file(profiles)


def update_llm_profile(existing_profile: ProfileConfig, updated_profile: ProfileConfig):
    if existing_profile.label != updated_profile.label:
        updated_profile.label = _get_valid_label_for_profile(updated_profile)

    profiles = _get_llm_profiles_from_file()
    for i, profile in enumerate(profiles):
        if profile.label == existing_profile.label:
            profiles[i] = updated_profile
            break

    _save_configured_llm_profiles_to_file(profiles)


def remove_llm_profile(profile: ProfileConfig):
    profiles = _get_llm_profiles_from_file()
    profiles = [p for p in profiles if p != profile]

    _save_configured_llm_profiles_to_file(profiles)


def get_llm_profiles_using_provider(provider: LLMProviderConfig) -> list[ProfileConfig]:
    return [
        profile
        for profile in _get_llm_profiles_from_file()
        if any(
            getattr(profile, f).provider_label == provider.label for f in _AGENT_FIELDS
        )
    ]
