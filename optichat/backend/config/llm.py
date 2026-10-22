import os
from typing import Any

from google.adk.models.lite_llm import LiteLlm

from optichat.llm.credentials import get_llm_credential_by_label
from optichat.llm.models import AgentConfig, ProfileConfig
from optichat.llm.profiles import get_llm_profile_by_label, get_llm_profiles

PROFILE_ENV_VAR = "OPTICHAT_PROFILE"


def get_litellm_params(agent_config: AgentConfig) -> dict[str, Any]:
    """Resolve an agent's credential/model settings into LiteLLM completion kwargs."""
    if not agent_config.enabled:
        raise ValueError("The agent has no API key and model configured.")

    credential = get_llm_credential_by_label(agent_config.credential_label)
    if credential is None:
        raise ValueError(
            f"The API key '{agent_config.credential_label}' is not configured."
        )

    params: dict[str, Any] = {
        "model": agent_config.llm_model,
        "api_key": credential.api_key,
    }
    if agent_config.temperature is not None:
        params["temperature"] = agent_config.temperature
    if agent_config.max_tokens is not None:
        params["max_tokens"] = agent_config.max_tokens
    return params


def build_adk_model(agent_config: AgentConfig) -> LiteLlm:
    return LiteLlm(**get_litellm_params(agent_config))


def get_default_profile() -> ProfileConfig:
    """Profile named by OPTICHAT_PROFILE, or else the first fully configured one."""
    label = os.getenv(PROFILE_ENV_VAR)
    if label:
        profile = get_llm_profile_by_label(label)
        if profile is None:
            raise ValueError(f"The LLM profile '{label}' was not found.")
    else:
        profile = next((p for p in get_llm_profiles() if p.enabled), None)
        if profile is None:
            raise ValueError(
                "No fully configured LLM profile found. "
                "Create one in the OptiChat settings."
            )

    if not profile.enabled:
        raise ValueError(f"The LLM profile '{profile.label}' is not fully configured.")
    return profile
