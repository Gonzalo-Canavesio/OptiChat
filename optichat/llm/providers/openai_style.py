from collections.abc import Callable

import httpx

from optichat.llm.providers.base import ProviderModel
from optichat.llm.providers.litellm_map import litellm_supports_tools


def bearer_auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def list_openai_style_models(
    http: httpx.Client,
    provider: str,
    url: str,
    api_key: str | None,
    auth_headers: Callable[[str], dict[str, str]],
    litellm_prefix: str = "",
) -> list[ProviderModel]:
    """List the models of an OpenAI-style `{"data": [{"id": ...}]}` listing.

    `litellm_prefix` is what LiteLLM needs to route the model; empty keeps the
    provider's bare id. LiteLLM's model map says which models accept tools.
    """
    if not api_key:
        raise ValueError(f"{provider} needs an API key to list its models.")
    payload = http.get(url, headers=auth_headers(api_key)).json()
    model_ids = [litellm_prefix + model["id"] for model in payload["data"]]
    return [
        ProviderModel(model_id, litellm_supports_tools(model_id))
        for model_id in model_ids
    ]
