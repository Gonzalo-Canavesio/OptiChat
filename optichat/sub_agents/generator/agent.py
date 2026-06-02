import asyncio
import json
import os
import re
import traceback as tb
from collections.abc import AsyncGenerator

import litellm
import pyomo.environ as pyo
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from langchain_experimental.utilities import PythonREPL
from loguru import logger
from pydantic import PrivateAttr

from optichat.config.constants import (
    GENERATOR_OUTPUT,
    MODEL_VERSIONS,
    MODELS_DICTIONARY,
)
from optichat.config.llm import coder_model_name
from optichat.sub_agents.generator.prompt import get_coder_prompt
from optichat.tools.callback_tool import _get_source_code_for_prompt

_PYTHON_REPL_TOOL = {
    "type": "function",
    "function": {
        "name": "python_repl",
        "description": (
            "Execute Python code in the REPL. "
            "pyomo, models_dictionary, solve_model, modify_and_solve, and all other "
            "shortcut functions are pre-injected — do NOT import them. "
            "Returns stdout output from the execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StateProxy:
    """Minimal tool_context proxy so generated code can access tool_context.state."""

    def __init__(self, state: dict):
        self.state = state


def _get_last_user_text(ctx: InvocationContext) -> str:
    """Return the most recent user-role text from session events (the expert's instruction)."""
    for event in reversed(list(ctx.session.events)):
        if event.content and event.content.role == "user":
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


def _inject_state_into_repl(repl: PythonREPL, state: dict) -> None:
    """Inject pyomo objects, shortcut functions, and session state into the REPL."""
    import optichat.tools.shortcut_functions as shortcut_functions

    # Clean stale user variables from prior calls
    if hasattr(repl, "globals") and repl.globals:
        preserved = {
            "__builtins__",
            "pyo",
            "value",
            "Constraint",
            "ConstraintList",
            "Var",
            "Param",
            "Objective",
            "ConcreteModel",
            "Set",
            "Expression",
            "minimize",
            "maximize",
            "models_dictionary",
            "MODEL_VERSIONS",
            "tool_context",
        }
        for name in dir(shortcut_functions):
            if callable(getattr(shortcut_functions, name)) and not name.startswith("_"):
                preserved.add(name)
        stale = [k for k in list(repl.globals.keys()) if k not in preserved]
        for k in stale:
            del repl.globals[k]

    repl.globals["pyo"] = pyo
    repl.globals["value"] = pyo.value
    repl.globals["Constraint"] = pyo.Constraint
    repl.globals["ConstraintList"] = pyo.ConstraintList
    repl.globals["Var"] = pyo.Var
    repl.globals["Param"] = pyo.Param
    repl.globals["Objective"] = pyo.Objective
    repl.globals["ConcreteModel"] = pyo.ConcreteModel
    repl.globals["Set"] = pyo.Set
    repl.globals["Expression"] = pyo.Expression
    repl.globals["minimize"] = pyo.minimize
    repl.globals["maximize"] = pyo.maximize

    for name in dir(shortcut_functions):
        item = getattr(shortcut_functions, name)
        if callable(item) and not name.startswith("_"):
            repl.globals[name] = item

    models_dictionary = state.get(MODELS_DICTIONARY, {}).copy()
    repl.globals[MODELS_DICTIONARY.lower()] = models_dictionary
    repl.globals["MODEL_VERSIONS"] = state.get(MODEL_VERSIONS, [])
    repl.globals["tool_context"] = _StateProxy(state)
    repl.locals = repl.globals


# ---------------------------------------------------------------------------
# CoderAgent
# ---------------------------------------------------------------------------


class CoderAgent(BaseAgent):
    """
    Single coder agent that writes and executes Pyomo code via litellm
    with python_repl registered as a native tool.

    Agentic loop:
      1. Codex receives the expert's structured instruction.
      2. Codex calls python_repl(code=...) to execute code.
      3. Stdout result is fed back; codex iterates if needed.
      4. Loop exits when codex stops calling tools.
    """

    _repl: PythonREPL = PrivateAttr(default_factory=PythonREPL)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        state = ctx.session.state

        user_instruction = _get_last_user_text(ctx)
        if not user_instruction:
            logger.warning("[CoderAgent] No user instruction found in session events")
            user_instruction = "Generate and execute Python/Pyomo code as instructed."
        logger.info(
            f"[CoderAgent] user_instruction received ({len(user_instruction)} chars):\n{user_instruction[:500]}"
        )

        # Parse the MODEL: field from the expert's grammar instruction as a hint
        _model_match = re.search(
            r"(?i)^MODEL\s*:\s*(.+)$", user_instruction, flags=re.MULTILINE
        )
        model_hint = _model_match.group(1).strip() if _model_match else None

        model_source_code = _get_source_code_for_prompt(state, model_hint=model_hint)
        system_instruction = get_coder_prompt(model_source_code=model_source_code)

        _inject_state_into_repl(self._repl, state)

        execution_log = []
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction},
        ]

        # Agentic tool-call loop
        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            response = await litellm.acompletion(
                model=coder_model_name,
                messages=messages,
                tools=[_PYTHON_REPL_TOOL],
                api_key=os.getenv("API_KEY"),
            )

            assistant_msg = response.choices[0].message
            # litellm message to dict for appending to messages history
            messages.append(assistant_msg.model_dump())

            tool_calls = assistant_msg.tool_calls
            if not tool_calls:
                break

            for tc in tool_calls:
                # Handle litellm structure
                if tc.type != "function":
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                    code = args.get("code", "")
                except json.JSONDecodeError:
                    code = ""
                    logger.error(
                        f"[CoderAgent] iter={iteration} JSON decode error on tool call arguments: {tc.function.arguments}"
                    )

                logger.info(
                    f"[CoderAgent] iter={iteration} executing {len(code)} chars…"
                )

                try:
                    repl_result = str(
                        await asyncio.wait_for(
                            asyncio.to_thread(self._repl.run, code),
                            timeout=180.0,
                        )
                    )
                except TimeoutError:
                    repl_result = "Execution timed out after 180 s."
                    logger.error(f"[CoderAgent] iter={iteration} timeout")
                except Exception as exc:
                    repl_result = (
                        f"Error: {type(exc).__name__}: {exc}\n{tb.format_exc()}"
                    )
                    logger.error(f"[CoderAgent] iter={iteration} exception: {exc}")

                execution_log.append(
                    f"--- code (iter {iteration}) ---\n{code}\n--- result ---\n{repl_result}"
                )
                logger.info(
                    f"[CoderAgent] iter={iteration} result: {repl_result[:120]}"
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": repl_result,
                    }
                )

        # Collect final text from the last response
        final_text = assistant_msg.content or ""

        combined = "\n\n".join(execution_log)
        if final_text:
            combined += "\n\n--- final response ---\n" + final_text

        logger.info(
            f"[CoderAgent] Done — {len(combined)} chars, {iteration} iteration(s)"
        )

        updated_models = self._repl.globals.get(MODELS_DICTIONARY.lower(), {})
        final_models = (
            updated_models if updated_models else state.get(MODELS_DICTIONARY, {})
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=combined)],
            ),
            actions=EventActions(
                state_delta={
                    GENERATOR_OUTPUT: combined,
                    MODELS_DICTIONARY: final_models,
                    MODEL_VERSIONS: list(final_models.keys()),
                }
            ),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_generator_agent() -> CoderAgent:
    """
    Create the generator agent.

    Returns a single CoderAgent named 'generator_agent' that:
      - Calls the LiteLLM API directly
      - Has python_repl registered as a native tool
      - Writes and executes Pyomo code in an agentic loop
      - Writes GENERATOR_OUTPUT and updated MODELS_DICTIONARY to session state
    """
    return CoderAgent(
        name="generator_agent",
        description=(
            "Pyomo code generator and executor. Receives a structured instruction "
            "from the expert agent, writes Python/Pyomo code, and executes it via "
            "python_repl in an agentic loop. Returns execution log as GENERATOR_OUTPUT."
        ),
    )
