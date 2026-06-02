import os

import yaml
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=env_path)

PROVIDER = os.getenv("PROVIDER")
API_KEY = os.getenv("API_KEY")

if not PROVIDER:
    raise ValueError("PROVIDER environment variable is not set.")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set.")

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config_agents.yaml")
with open(config_path, encoding="utf-8") as file:
    config = yaml.safe_load(file)

if PROVIDER not in config:
    raise ValueError(f"The provider '{PROVIDER}' was not found in config_agents.yaml")

agent_configs = config[PROVIDER]

root_model = LiteLlm(model=agent_configs["root"]["model"], api_key=API_KEY)
expert_model = LiteLlm(model=agent_configs["expert"]["model"], api_key=API_KEY)
illustrator_model = LiteLlm(
    model=agent_configs["illustrator"]["model"], api_key=API_KEY
)

coder_model_name = agent_configs["coder"]["model"]
