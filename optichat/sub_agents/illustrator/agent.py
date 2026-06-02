from google.adk.agents import Agent

from optichat.config.llm import illustrator_model
from optichat.sub_agents.illustrator.prompt import ILLUSTRATOR_PROMPT
from optichat.tools.callback_tool import (
    check_illustrator_agent_runtime,
    check_llm_request,
    check_llm_response,
    check_tool_response,
    check_tool_usage,
    diagnose_if_infeasible,
)
from optichat.tools.illustrator_tool import get_model_info_for_description


def create_illustrator_agent():
    illustrator_agent = Agent(
        name="illustrator_agent",
        model=illustrator_model,
        tools=[get_model_info_for_description],
        description=(
            "Generates comprehensive descriptions of optimization models "
            "by explaining the problem context, decision variables, "
            "parameters, constraints, and objectives "
            "in accessible language for domain practitioners."
        ),
        instruction=ILLUSTRATOR_PROMPT,
        output_key="MODEL_DESCRIPTION",
        before_agent_callback=diagnose_if_infeasible,
        after_agent_callback=check_illustrator_agent_runtime,
        before_model_callback=check_llm_request,
        after_model_callback=check_llm_response,
        before_tool_callback=check_tool_usage,
        after_tool_callback=check_tool_response,
    )
    return illustrator_agent
