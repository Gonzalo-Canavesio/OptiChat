import httpx

from optichat.llm.providers.base import Provider, ProviderModel
from optichat.llm.providers.litellm_map import litellm_supports_tools

# The Ollama server on this computer, at the address LiteLLM also defaults to.
_BASE_URL = "http://localhost:11434"
# ollama_chat is the LiteLLM route that passes tools to Ollama natively.
_PREFIX = "ollama_chat/"


class Ollama(Provider):
    name = "Ollama (Local)"
    api_key_required = False

    def list_models(
        self, http: httpx.Client, api_key: str | None
    ) -> list[ProviderModel]:
        """List the models pulled into the local Ollama server; it needs no key."""
        payload = http.get(f"{_BASE_URL}/api/tags").json()
        names = [model["name"] for model in payload["models"]]
        return [
            ProviderModel(_PREFIX + name, self._supports_tools(http, name))
            for name in names
        ]

    def accepts_pdf(self, model: str) -> bool:
        """Never: Ollama's API takes images, not PDF files."""
        return False

    def _supports_tools(self, http: httpx.Client, name: str) -> bool:
        """Ask Ollama whether model `name` accepts tools, or else LiteLLM's model map.

        Older Ollama versions don't report the model's capabilities.
        """
        payload = http.post(f"{_BASE_URL}/api/show", json={"model": name}).json()
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, list):
            return "tools" in capabilities
        return litellm_supports_tools(_PREFIX + name)
