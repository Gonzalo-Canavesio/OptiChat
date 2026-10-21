import os
from unittest.mock import patch

from litellm import AuthenticationError, litellm

from optichat.llm.models import LLMCredential
from optichat.llm.providers import get_env_var_name_for_provider


def get_available_llm_models(credential: LLMCredential) -> list[str]:
    temp_env_var = {
        get_env_var_name_for_provider(credential.provider): credential.api_key
    }
    with patch.dict(os.environ, temp_env_var):
        return litellm.get_valid_models(check_provider_endpoint=True)


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


_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Does nothing.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def get_model_validation_error(credential: LLMCredential, model: str) -> str | None:
    """Return why `model` can't be used with `credential`'s key, or None if it works."""
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        # max_completion_tokens (not max_tokens) because newer models reject the
        # latter; it is at least 16 because some models refuse smaller limits and
        # reasoning models need headroom to produce any output.
        litellm.completion(
            model=resolve_litellm_model(credential.provider, model),
            messages=messages,
            api_key=credential.api_key,
            # The agents rely on function calling, so the model must accept tools.
            tools=[_PROBE_TOOL],
            max_completion_tokens=256,
        )
    except AuthenticationError:
        return "The API key was rejected by the provider."
    except Exception as e:
        return str(e).replace(credential.api_key, "***").strip()
    return None
