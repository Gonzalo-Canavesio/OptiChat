from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from optichat.config.constants import OUTPUT_KEY_ROOT_AGENT
from optichat.config.llm import root_model
from optichat.sub_agents.expert.agent import create_expert_agent
from optichat.sub_agents.illustrator.agent import create_illustrator_agent
from optichat.sub_agents.root.prompt import ROOT_AGENT_PROMPT
from optichat.tools.callback_tool import (
    check_llm_request,
    check_llm_response,
    check_root_agent_runtime,
    handle_illustrator_response,
    initialize_session_and_start_root_agent,
    route_to_expert,
)


def create_root_agent(workflow="default"):
    if workflow == "default":
        expert_agent = create_expert_agent(prompt_version=1, tools_version=1)
        illustrator_agent = create_illustrator_agent()

        root_agent = Agent(
            name="root_agent",
            model=root_model,
            sub_agents=[expert_agent],
            tools=[route_to_expert, AgentTool(illustrator_agent)],
            description="first point of contact for all user queries",
            instruction=ROOT_AGENT_PROMPT,
            output_key=OUTPUT_KEY_ROOT_AGENT,
            before_agent_callback=initialize_session_and_start_root_agent,
            after_agent_callback=check_root_agent_runtime,
            before_model_callback=check_llm_request,
            after_model_callback=check_llm_response,
            after_tool_callback=handle_illustrator_response,
        )

        return root_agent
    else:
        raise NotImplementedError(f"Workflow '{workflow}' is not implemented.")
