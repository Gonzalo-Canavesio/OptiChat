import logging
import os

from loguru import logger

from optichat.backend.agents.root.agent import create_root_agent
from optichat.backend.config.llm import get_default_profile

# Disable OpenTelemetry to avoid context management issues
os.environ["OTEL_SDK_DISABLED"] = "true"
# Suppress OpenTelemetry warnings
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

logger.add("conversation_logs.txt", rotation="10 MB")

root_agent = create_root_agent(get_default_profile(), workflow="default")
