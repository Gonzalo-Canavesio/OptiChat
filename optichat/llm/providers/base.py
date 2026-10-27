from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import httpx


class ProviderVerificationError(Exception):
    """The provider's access could not be verified; the message is user-facing."""


class APIKeyRejectedError(ProviderVerificationError):
    pass


class ProviderUnavailableError(ProviderVerificationError):
    pass


@dataclass(frozen=True)
class ProviderModel:
    id: str
    supports_tools: bool


class Provider(ABC):
    """An LLM provider OptiChat can list models from and run the agents on.

    To add one, subclass it, set `name` and `api_key_required`, implement both
    methods with plain httpx calls on the client you're given, and list an
    instance in registry.PROVIDERS. HTTP errors, bad statuses and unexpected
    answers are turned into user-facing errors for you.
    """

    name: ClassVar[str]
    api_key_required: ClassVar[bool]

    @abstractmethod
    def list_models(
        self, http: httpx.Client, api_key: str | None
    ) -> list[ProviderModel]:
        """List all the models the key (if any) gives access to.

        Set `supports_tools` on each one; choosing which ones the agents can use
        is up to the caller. litellm_map.litellm_supports_tools can tell for the
        models LiteLLM knows.
        """

    @abstractmethod
    def accepts_pdf(self, model: str) -> bool:
        """Whether the model with this LiteLLM id accepts PDF files as input.

        litellm_map.litellm_accepts_pdf can tell for the models LiteLLM knows.
        """
