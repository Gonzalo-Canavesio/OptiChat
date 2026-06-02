from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from optichat.config.constants import OUTPUT_KEY_EXPERT_AGENT
from optichat.config.llm import expert_model
from optichat.sub_agents.expert.prompt import get_expert_agent_prompt
from optichat.sub_agents.generator.agent import create_generator_agent
from optichat.tools.callback_tool import (
    check_expert_agent_runtime,
    check_is_expert_agent_used,
    check_llm_request,
    check_llm_response,
    check_tool_response,
    check_tool_usage,
)
from optichat.tools.custom_tool import infeasibility_diagnosis, robustness_analysis
from optichat.tools.search_tool import get_model_components


def create_expert_agent(prompt_version=1, tools_version=1):
    expert_agent_prompt = get_expert_agent_prompt(prompt_version)
    generator_tool = AgentTool(agent=create_generator_agent())
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
        model=expert_model,
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
