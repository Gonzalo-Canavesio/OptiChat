from functools import lru_cache

from litellm import litellm

from optichat.llm.models import LLMCredential
from optichat.llm.providers import ProviderVerificationError, fetch_provider_models


# Streamlit reruns the profile form on every interaction, so the list is cached.
# Failures raise and are therefore not cached.
@lru_cache(maxsize=32)
def _cached_models(provider: str, api_key: str) -> tuple[str, ...]:
    return tuple(fetch_provider_models(provider, api_key))


_AGENT_MODES = ("chat", "responses")


def _litellm_model_info(model: str) -> dict:
    """Look `model` up in LiteLLM's model map, or return {} if it isn't there.

    Some providers (e.g. Anthropic) are mapped without the `provider/` prefix our
    ids carry, so the bare name is tried too as long as its provider matches.
    """
    if model in litellm.model_cost:
        return litellm.model_cost[model]
    provider, _, bare_model = model.partition("/")
    info = litellm.model_cost.get(bare_model, {}) if bare_model else {}
    return info if info.get("litellm_provider") == provider else {}


def _can_run_agents(model: str) -> bool:
    """Whether LiteLLM knows `model` as a chat model that accepts tools."""
    info = _litellm_model_info(model)
    return info.get("mode") in _AGENT_MODES and bool(
        info.get("supports_function_calling")
    )


def fetch_agent_models(provider: str, api_key: str) -> list[str]:
    """The models `api_key` can use that the agents can run on.

    Models LiteLLM doesn't know are left out, since nothing says they accept tools.
    Raises ProviderVerificationError if the key can't be verified.
    """
    return [m for m in _cached_models(provider, api_key) if _can_run_agents(m)]


def get_available_llm_models(credential: LLMCredential) -> list[str]:
    try:
        return fetch_agent_models(credential.provider, credential.api_key)
    except ProviderVerificationError:
        return []


# OpenAI model families whose function calling only works through the Responses
# API, not /v1/chat/completions. LiteLLM's model map does not flag them yet.
_OPENAI_RESPONSES_ONLY_PREFIXES = ("gpt-6",)


def resolve_litellm_model(provider: str | None, model: str) -> str:
    """Prefix `model` with LiteLLM's Responses API route when it requires one."""
    if provider != "OpenAI" or model.startswith(("openai/", "responses/")):
        return model
    mode = litellm.model_cost.get(model, {}).get("mode")
    if mode == "responses" or model.startswith(_OPENAI_RESPONSES_ONLY_PREFIXES):
        return f"openai/responses/{model}"
    return model
