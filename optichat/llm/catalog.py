from functools import lru_cache

from optichat.llm.models import LLMCredential
from optichat.llm.providers.base import ProviderModel
from optichat.llm.providers.registry import fetch_provider_models, get_provider


# Streamlit reruns the profile form on every interaction, so the list is cached.
# Failures raise and are therefore not cached.
@lru_cache(maxsize=32)
def _cached_models(provider: str, api_key: str | None) -> tuple[ProviderModel, ...]:
    return tuple(fetch_provider_models(provider, api_key))


def accepts_pdf(credential: LLMCredential, model: str) -> bool:
    """Whether `model`, used through `credential`, accepts PDF files as input."""
    return get_provider(credential.provider).accepts_pdf(model)


def fetch_agent_models(credential: LLMCredential) -> list[str]:
    """The ids of the models `credential` gives access to that the agents can run on.

    Models the provider doesn't say accept tools are left out. Raises
    ProviderVerificationError if the credential can't be verified.
    """
    models = _cached_models(credential.provider, credential.api_key)
    return [m.id for m in models if m.supports_tools]
