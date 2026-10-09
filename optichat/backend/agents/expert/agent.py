from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from optichat.backend.agents.expert.prompt import get_expert_agent_prompt
from optichat.backend.agents.generator.agent import create_generator_agent
from optichat.backend.config.constants import OUTPUT_KEY_EXPERT_AGENT
from optichat.backend.config.llm import build_adk_model
from optichat.backend.tools.callback_tool import (
    check_expert_agent_runtime,
    check_is_expert_agent_used,
    check_llm_request,
    check_llm_response,
    check_tool_response,
    check_tool_usage,
)
from optichat.backend.tools.custom_tool import (
    infeasibility_diagnosis,
    robustness_analysis,
)
from optichat.backend.tools.search_tool import get_model_components
from optichat.llm.models import ProfileConfig


def create_expert_agent(profile: ProfileConfig, prompt_version=1, tools_version=1):
    expert_agent_prompt = get_expert_agent_prompt(prompt_version)
    generator_tool = AgentTool(agent=create_generator_agent(profile))
    if tools_version == 1:
        expert_agent_tools = [
            get_model_components,
            generator_tool,
            #   python_repl_func,
            #   code_rag,
            #   paper_rag,
            infeasibility_diagnosis,
            #   ldr_model_generator,
            #   ldr_expression_generator,
            robustness_analysis,
        ]  # TODO: infeasibility diagnosis, ldr_model_generator and ldr_expression_generator, robustness_analysis under testing
    else:
        raise NotImplementedError(
            f"Tools version '{tools_version}' is not implemented."
        )

    expert_agent = Agent(
        name="expert_agent",
        model=build_adk_model(profile.expert_agent),
        tools=expert_agent_tools,
        description=(
            "Optimization & operations research expert that "
            "interacts with <models>, <models_code>, <models_paper>"
        ),
        instruction=expert_agent_prompt,
        output_key=OUTPUT_KEY_EXPERT_AGENT,
        disallow_transfer_to_parent=True,
        before_agent_callback=check_is_expert_agent_used,
        after_agent_callback=check_expert_agent_runtime,
        before_model_callback=check_llm_request,
        after_model_callback=check_llm_response,
        before_tool_callback=check_tool_usage,
        after_tool_callback=check_tool_response,
    )
    return expert_agent
