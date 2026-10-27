from litellm import litellm

_AGENT_MODES = ("chat", "responses")


def litellm_model_info(model: str) -> dict:
    """Look `model` up in LiteLLM's model map, or return {} if it isn't there.

    Some providers (e.g. Anthropic) are mapped without the `provider/` prefix our
    ids carry, so the bare name is tried too as long as its provider matches.
    """
    if model in litellm.model_cost:
        return litellm.model_cost[model]
    provider, _, bare_model = model.partition("/")
    info = litellm.model_cost.get(bare_model, {}) if bare_model else {}
    return info if info.get("litellm_provider") == provider else {}


def litellm_supports_tools(model: str) -> bool:
    """Whether LiteLLM knows `model` as a chat model that accepts tools."""
    info = litellm_model_info(model)
    return info.get("mode") in _AGENT_MODES and bool(
        info.get("supports_function_calling")
    )


def litellm_accepts_pdf(model: str) -> bool:
    """Whether LiteLLM knows `model` accepts PDF files as input."""
    return bool(litellm_model_info(model).get("supports_pdf_input"))
