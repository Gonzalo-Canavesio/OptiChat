from google.adk.agents import Agent

from optichat.backend.agents.illustrator.prompt import ILLUSTRATOR_PROMPT
from optichat.backend.config.constants import MODEL_DESCRIPTION
from optichat.backend.config.llm import build_adk_model
from optichat.backend.tools.callback_tool import (
    check_illustrator_agent_runtime,
    check_llm_request,
    check_llm_response,
    check_tool_response,
    check_tool_usage,
    start_illustrator_timing,
)
from optichat.backend.tools.illustrator_tool import get_model_info_for_description
from optichat.llm.models import ProfileConfig


def create_illustrator_agent(profile: ProfileConfig):
    illustrator_agent = Agent(
        name="illustrator_agent",
        model=build_adk_model(profile.illustrator_agent),
        tools=[get_model_info_for_description],
        description=(
            "Generates comprehensive descriptions of optimization models "
            "by explaining the problem context, decision variables, "
            "parameters, constraints, and objectives "
            "in accessible language for domain practitioners."
        ),
        instruction=ILLUSTRATOR_PROMPT,
        output_key=MODEL_DESCRIPTION,
        before_agent_callback=start_illustrator_timing,
        after_agent_callback=check_illustrator_agent_runtime,
        before_model_callback=check_llm_request,
        after_model_callback=check_llm_response,
        before_tool_callback=check_tool_usage,
        after_tool_callback=check_tool_response,
    )
    return illustrator_agent
