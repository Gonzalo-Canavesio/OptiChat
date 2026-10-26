from google.adk.agents import Agent

from optichat.backend.agents.expert.agent import create_expert_agent
from optichat.backend.agents.root.prompt import ROOT_AGENT_PROMPT
from optichat.backend.config.constants import OUTPUT_KEY_ROOT_AGENT
from optichat.backend.config.llm import build_adk_model
from optichat.backend.tools.callback_tool import (
    check_llm_request,
    check_llm_response,
    check_root_agent_runtime,
    route_to_expert,
    start_root_turn,
)
from optichat.llm.models import ProfileConfig


def create_root_agent(profile: ProfileConfig, workflow="default"):
    if workflow == "default":
        expert_agent = create_expert_agent(profile, prompt_version=1, tools_version=1)

        root_agent = Agent(
            name="root_agent",
            model=build_adk_model(profile.root_agent),
            sub_agents=[expert_agent],
            tools=[route_to_expert],
            description="first point of contact for all user queries",
            instruction=ROOT_AGENT_PROMPT,
            output_key=OUTPUT_KEY_ROOT_AGENT,
            before_agent_callback=start_root_turn,
            after_agent_callback=check_root_agent_runtime,
            before_model_callback=check_llm_request,
            after_model_callback=check_llm_response,
        )

        return root_agent
    else:
        raise NotImplementedError(f"Workflow '{workflow}' is not implemented.")
