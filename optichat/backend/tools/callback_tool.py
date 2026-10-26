import time
import json
import os
import glob
import tiktoken
from loguru import logger
from typing import Dict, Any
from typing import Optional
from copy import deepcopy
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.models import LlmResponse, LlmRequest
from optichat.backend.config import workspace
from optichat.backend.config.constants import (TEMPORARY_STATES,
                                       EXPERT_AGENT_USES, EXPERT_AGENT_MAX_USES, EXPERT_AGENT_START_TIME,
                                       MODELS_DICTIONARY, HISTORICAL_MODELS_METADATA, CHAT_MODELING_LANGUAGE,
                                       IS_MODELS_DICTIONARY_AVAILABLE, USER_QUERY,
                                       CURRENT_SELECTED_MODEL, CURRENT_SELECTED_MODEL_INFO,
                                       ANALYSIS_PROMPT_CONTENT,
                                       EXPERT_AGENT_PYTHON_REPL_FUNC_USES,
                                       LLM_CALL_COUNTER, TOOL_CALL_START_TIME, LLM_CALL_START_TIME,
                                       MODEL_COMPONENTS_CACHE, CURRENT_ANALYSIS_TYPE,
                                       OUTPUT_KEY_EXPERT_AGENT, OUTPUT_KEY_ROOT_AGENT)
from optichat.backend.tools import timing_tracker
from optichat.backend.agents.expert.prompt import get_expert_agent_prompt
from optichat.backend.agents.root.prompt import ANALYSIS_CONTENT_FEASIBLE, ANALYSIS_CONTENT_INFEASIBLE
import re


def _resolve_model_status(model_name: str, historical_metadata: dict) -> str:
    """Look up the solution_status for a single model name from historical_metadata."""
    for date_data in historical_metadata.values():
        if model_name in date_data:
            return date_data[model_name].get("solution_status", "unknown")
        for base_data in date_data.values():
            modified = base_data.get("modified_models", {})
            if model_name in modified:
                return modified[model_name].get("solution_status", "unknown")
    return "unknown"


def update_selected_model_state(model_names, historical_metadata: dict, state: dict) -> None:
    """
    Update all three model-selection state keys from a single call:
      - CURRENT_SELECTED_MODEL       : list of selected model names
      - CURRENT_SELECTED_MODEL_INFO  : display string with one line per model + status
      - ANALYSIS_PROMPT_CONTENT      : prompt block driven by the primary (first) model's status

    model_names may be a list or a single string (converted to a one-element list).
    """
    if isinstance(model_names, str):
        model_names = [model_names] if model_names else []

    if not model_names:
        state[CURRENT_SELECTED_MODEL] = []
        state[CURRENT_SELECTED_MODEL_INFO] = "(none)"
        state[ANALYSIS_PROMPT_CONTENT] = ANALYSIS_CONTENT_FEASIBLE
        return

    # Resolve status for every selected model
    statuses = {name: _resolve_model_status(name, historical_metadata) for name in model_names}

    state[CURRENT_SELECTED_MODEL] = model_names

    # Build multi-model display string
    info_lines = [f"- {name} (solution status: {statuses[name]})" for name in model_names]
    state[CURRENT_SELECTED_MODEL_INFO] = "\n".join(info_lines)

    # ANALYSIS_PROMPT_CONTENT is driven by the PRIMARY (first) model's status
    primary_status = statuses[model_names[0]]
    state[ANALYSIS_PROMPT_CONTENT] = (
        ANALYSIS_CONTENT_INFEASIBLE if primary_status.startswith("infeasible")
        else ANALYSIS_CONTENT_FEASIBLE
    )
    logger.info(f"Selected model state updated: {model_names} (primary status: {primary_status})")


def start_root_turn(callback_context: CallbackContext):
    """Root agent before_agent_callback: reset the per-message state for a new message.

    The chat itself is initialized before the root agent ever runs
    (see optichat.backend.initialization).
    """
    state = callback_context.state
    state.update(TEMPORARY_STATES)

    parts = callback_context.user_content.parts or []
    user_query = (parts[0].text or "") if parts else ""

    # Parse [Using model: ...] prefix → update all model-selection state keys and strip from query
    selected_model_match = re.match(r'^\[Using model:\s*([^\]]+)\]\s*', user_query)
    if selected_model_match:
        model_names = [m.strip() for m in selected_model_match.group(1).split(",") if m.strip()]
        update_selected_model_state(model_names, state.get(HISTORICAL_MODELS_METADATA, {}), state)
        user_query = user_query[selected_model_match.end():]
    state[USER_QUERY] = user_query

    # Clear model components cache for new query
    state[MODEL_COMPONENTS_CACHE] = {}

    timing_tracker.reset_tracker()
    timing_tracker.record_agent_start("root_agent")
    return None


def check_is_expert_agent_used(callback_context: CallbackContext):
    expert_agent_uses = callback_context.state.get(EXPERT_AGENT_USES, 0)
    expert_agent_max_uses = callback_context.state.get(EXPERT_AGENT_MAX_USES, 5)

    if expert_agent_uses >= expert_agent_max_uses:
        return types.Content(
            parts=[types.Part(text=f"[system message]: Expert agent has been used {expert_agent_uses} times. "
                                   f"The limit is {expert_agent_max_uses} times per user query. "
                                   f"Please explain the results gathered so far to the user.")],
            role="model"
        )
    else:
        callback_context.state[EXPERT_AGENT_USES] = expert_agent_uses + 1
        # Only set start time on first use
        if expert_agent_uses == 0:
            callback_context.state[EXPERT_AGENT_START_TIME] = time.time()
            # Record agent start in timing tracker
            timing_tracker.record_agent_start("expert_agent")

        # Read analysis type already set by root's after_model_callback (check_llm_response)
        analysis_type = callback_context.state.get(CURRENT_ANALYSIS_TYPE, "RETRIEVAL")
        logger.info(f"Expert agent invoked with analysis type: {analysis_type}")
        return None


def check_expert_agent_runtime(callback_context: CallbackContext):
    expert_agent_uses = callback_context.state.get(EXPERT_AGENT_USES, 0)
    if expert_agent_uses > 0:
        # Record agent end in timing tracker
        timing_tracker.record_agent_end("expert_agent")

        # Legacy timing (kept for backwards compatibility)
        start_time = callback_context.state.get(EXPERT_AGENT_START_TIME)
        if start_time:
            elapsed_time = time.time() - start_time
            logger.debug(f"*** Expert Agent Runtime: {elapsed_time:.2f} s "
                         f"({elapsed_time/60:.2f} min) ***")
    return None


def check_llm_request(callback_context: CallbackContext, llm_request: LlmRequest):
    agent_name = callback_context.agent_name

    # Start timing for LLM call
    callback_context.state[LLM_CALL_START_TIME] = time.time()

    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
    # Ensure system_instruction is Content and parts list exists
    if not isinstance(original_instruction, types.Content):
         # Handle case where it might be a string (though config expects Content)
         original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

    original_text = original_instruction.parts[0].text or ""
    show_first_n_chars = 0
    logger.info((f"[Callback] Inspecting LLM request from '{agent_name}': "
                 f"{original_text[:show_first_n_chars]}"
                 f"\n... (showing only the first {show_first_n_chars} characters)"))
                   
    # Dynamic Prompt Injection for Expert Agent
    if agent_name == "expert_agent":
        analysis_type = callback_context.state.get(CURRENT_ANALYSIS_TYPE, "RETRIEVAL")
        logger.info(f"Injecting Dynamic Prompt for {agent_name} with strategy: {analysis_type}")

        try:
            # Inject component index for ALL expert_agent queries
            model_description = _get_component_index_for_prompt(callback_context.state)

            # Inject diagnosis report for feasibility restoration queries
            diagnosis_report = None
            cached_model_name = None  # Track the model name for diagnosis status
            if analysis_type == "FEASIBILITY_RESTORATION":
                # Prefer state-tracked model selection (primary = first); fall back to query-text extraction
                selected = callback_context.state.get(CURRENT_SELECTED_MODEL, [])
                base_model_name = (selected[0] if isinstance(selected, list) and selected else selected or "")
                if base_model_name and "__" in base_model_name:
                    base_model_name = base_model_name.split("__")[0]
                if not base_model_name:
                    user_query = callback_context.state.get(USER_QUERY, "")
                    models_dictionary = callback_context.state.get(MODELS_DICTIONARY, {})
                    base_model_name = _extract_base_model_from_query(user_query, models_dictionary)

                if base_model_name:
                    logger.info(f"Identified base model for diagnosis report: {base_model_name}")
                    diagnosis_data = _load_infeasibility_diagnosis_report(base_model_name)
                    if diagnosis_data:
                        diagnosis_report = _format_diagnosis_for_prompt(diagnosis_data)
                        cached_model_name = base_model_name  # Store for prompt injection
                        logger.info("✓ Diagnosis report formatted and ready for injection")
                    else:
                        logger.warning(f"No diagnosis report found for {base_model_name}")
                else:
                    logger.warning("Could not identify base model from query for diagnosis report")

            new_prompt_text = get_expert_agent_prompt(
                prompt_version=1,
                analysis_type=analysis_type,
                diagnosis_report=diagnosis_report,
                cached_model_name=cached_model_name,
                model_description=model_description,
                modeling_language=callback_context.state.get(CHAT_MODELING_LANGUAGE, "Pyomo"),
            )

            # Manually substitute state-based placeholders that ADK would normally
            # inject into the static instruction but won't re-process on our override.
            state = callback_context.state
            new_prompt_text = new_prompt_text.replace(
                "{IS_MODELS_DICTIONARY_AVAILABLE}",
                str(state.get(IS_MODELS_DICTIONARY_AVAILABLE, False))
            )
            new_prompt_text = new_prompt_text.replace(
                "{MODELS_METADATA_FORMATTED}",
                state.get("MODELS_METADATA_FORMATTED", "No models available")
            )
            new_prompt_text = new_prompt_text.replace(
                "{USER_QUERY}",
                state.get(USER_QUERY, "")
            )
            new_prompt_text = new_prompt_text.replace(
                "{EXPERT_AGENT_PYTHON_REPL_FUNC_USES}",
                str(state.get(EXPERT_AGENT_PYTHON_REPL_FUNC_USES, 3))
            )

            llm_request.config.system_instruction = new_prompt_text
            logger.info(f"[ExpertAgent] Full system prompt ({len(new_prompt_text)} chars):\n{new_prompt_text}")
        except Exception as e:
            logger.error(f"Failed to inject dynamic prompt: {e}")

    return None


def _generate_component_index(version: str, models_dictionary: dict) -> str:
    """
    Generate a compact component families index for a model version.
    Groups indexed components by base name, shows type and doc string.
    No values or expressions — just enough to make precise GMC calls.
    """
    from collections import defaultdict

    if version not in models_dictionary:
        return ""

    model_data = models_dictionary[version]
    sol_status = model_data.get("obj", {}).get("sol_status", "unknown")
    obj_value = model_data.get("obj", {}).get("value", "unknown")

    # Group components by base name (strip index brackets)
    families = {}
    for comp_name, comp_data in model_data.items():
        if not isinstance(comp_data, dict):
            continue
        comp_type = comp_data.get("component_type", "")
        if not comp_type:
            continue
        # Extract base name: "demand[1,2]" → "demand"
        base_match = re.match(r'^([^\[]+)', comp_name)
        base_name = base_match.group(1).strip() if base_match else comp_name
        is_indexed = '[' in comp_name
        if base_name not in families:
            families[base_name] = {
                "type": comp_type,
                "doc": comp_data.get("doc", ""),
                "indexed": is_indexed
            }

    # Group by component type
    by_type = defaultdict(list)
    for name, info in sorted(families.items()):
        by_type[info["type"]].append((name, info))

    type_order = ["objective", "parameter", "variable", "constraint"]
    type_labels = {
        "objective": "Objective",
        "parameter": "Parameters",
        "variable": "Variables",
        "constraint": "Constraints"
    }

    lines = [f"MODEL COMPONENTS INDEX: {version} ({sol_status}, obj={obj_value})"]
    for t in type_order:
        if t not in by_type:
            continue
        lines.append(f"\n{type_labels[t]}:")
        for name, info in by_type[t]:
            idx_str = "[indexed]" if info["indexed"] else "[scalar]"
            doc_str = f" — {info['doc']}" if info.get("doc") else ""
            lines.append(f"  {name} {idx_str}{doc_str}")
    lines.append(
        "\nUse component names above as pattern filters in get_model_components "
        "to fetch specific current values (e.g., pattern='demand' to get all demand entries)."
    )
    return "\n".join(lines)


def _strip_paper_metadata(content: str) -> str:
    """Strip header metadata and footer disclaimer from a generated paper file.

    Removes everything up to and including the first '---' line (header block
    with timestamp / author / model name) and everything from the last '---'
    line onward (the auto-generated disclaimer footer).
    """
    lines = content.split('\n')

    # Find first '---' separator → marks end of header
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == '---':
            start_idx = i + 1
            break

    # Find last '---' separator → marks start of footer
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '---':
            end_idx = i
            break

    return '\n'.join(lines[start_idx:end_idx]).strip()


def _get_component_index_for_prompt(state: dict) -> Optional[str]:
    """
    Return the component index / model description for the model being queried.

    Priority:
    1. Cache (state key COMPONENT_INDEX_{base_model})
    2. Generated paper file at <workspace>/model_objects/generated_papers/{base_model}_description.md
    3. Fall back to _generate_component_index (programmatic summary)
    """
    # Prefer the state-tracked selection (primary = first); fall back to query-text extraction
    selected = state.get(CURRENT_SELECTED_MODEL, [])
    base_model = (selected[0] if isinstance(selected, list) and selected else selected or "")
    if base_model and "__" in base_model:
        base_model = base_model.split("__")[0]  # strip modified suffix to get base name

    models_dict = state.get(MODELS_DICTIONARY, {})
    if not base_model:
        user_query = state.get(USER_QUERY, "")
        base_model = _extract_base_model_from_query(user_query, models_dict)

    if not base_model:
        # models_dictionary may not be loaded yet (e.g. frontend model selection before init).
        # Scan the papers directory to find an available description file.
        papers_dir = str(workspace.generated_papers_dir())
        paper_files = sorted(glob.glob(os.path.join(papers_dir, "*_description.md")))
        if paper_files:
            # Pick the most recently modified paper
            paper_files.sort(key=os.path.getmtime, reverse=True)
            chosen = os.path.basename(paper_files[0])          # e.g. "recovery_description.md"
            base_model = chosen[: chosen.rfind("_description.md")]
            logger.info(f"Inferred base model from papers directory: {base_model}")
        else:
            logger.warning("Could not identify base model for component index injection")
            return None

    # 1. Check cache
    cache_key = f"COMPONENT_INDEX_{base_model}"
    if cache_key in state:
        logger.info(f"Using cached component index for {base_model}")
        return state[cache_key]

    # 2. Try generated paper file
    paper_path = os.path.join(workspace.generated_papers_dir(), f"{base_model}_description.md")
    if os.path.exists(paper_path):
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                raw = f.read()
            index_str = _strip_paper_metadata(raw)
            logger.info(f"Loaded component index from paper for {base_model} ({len(index_str)} chars)")
            state[cache_key] = index_str
            return index_str
        except Exception as e:
            logger.warning(f"Failed to read paper file for {base_model}: {e}")

    # 3. Fall back to programmatic generation
    index_str = _generate_component_index(base_model, models_dict)
    if not index_str:
        logger.warning(f"Could not generate component index for {base_model}")
        return None

    state[cache_key] = index_str
    logger.info(f"Generated component index for {base_model} ({len(index_str)} chars)")
    return index_str


def _get_source_code_for_prompt(state: dict, model_hint: str = None) -> Optional[str]:
    """
    Retrieve and format source code for the base model referenced in query.

    model_hint: optional model name (e.g. parsed from grammar instruction MODEL: field)
                used as fallback when USER_QUERY doesn't mention the model name.

    Returns formatted source code string or None if not available.
    """
    from optichat.backend.tools.source_code_utils import (
        get_source_code_for_model,
        format_source_for_prompt
    )

    # 1. Identify base model — prefer state-tracked selection (primary = first)
    selected = state.get(CURRENT_SELECTED_MODEL, [])
    base_model = (selected[0] if isinstance(selected, list) and selected else selected or "")
    if base_model and "__" in base_model:
        base_model = base_model.split("__")[0]  # strip modified suffix to get base name

    models_dict = state.get(MODELS_DICTIONARY, {})
    historical_metadata = state.get(HISTORICAL_MODELS_METADATA, {})
    if not base_model:
        user_query = state.get(USER_QUERY, "")
        base_model = _extract_base_model_from_query(user_query, models_dict)

    if not base_model and model_hint:
        # Fall back to the model name from the grammar instruction (MODEL: field)
        base_model = _extract_base_model_from_query(model_hint, models_dict)
        if base_model:
            logger.info(f"Using model hint '{model_hint}' → resolved base model: {base_model}")
    if not base_model and model_hint:
        # models_dict may be empty (e.g. called before session state is fully populated).
        # Use model_hint directly as the model name — it comes from MODEL: <exact_key>
        # in the expert's grammar instruction, so it is reliable.
        base_model = model_hint.strip()
        logger.info(f"models_dict empty; using model_hint directly as base model: {base_model}")
    if not base_model:
        logger.warning("Could not identify base model for source code injection")
        return None
    
    # 2. Check cache
    cache_key = f"SOURCE_CODE_{base_model}"
    if cache_key in state:
        logger.info(f"Using cached source code for {base_model}")
        return state[cache_key]
    
    # 3. Retrieve source code
    source_code = get_source_code_for_model(base_model, models_dict, historical_metadata)
    if not source_code:
        logger.warning(f"No source code found for {base_model}")
        return None
    
    # 4. Get pkl path
    model_info = models_dict.get(base_model, {})
    pkl_path = model_info.get("local_path_to_object", "unknown")
    
    # 5. Format for prompt
    formatted_source = format_source_for_prompt(source_code, base_model, pkl_path, historical_metadata)
    
    # 6. Cache it
    state[cache_key] = formatted_source
    
    logger.info(f"Injecting {len(formatted_source)} chars of source code for {base_model}")
    return formatted_source


def _extract_base_model_from_query(user_query: str, models_dictionary: dict) -> Optional[str]:
    """
    Extract the base model name from user query.
    Prefer base models (no __) over modified versions.
    """
    # Remove analysis tag
    query_clean = re.sub(r'^\[(WHAT_IF|WHY_NOT|FEASIBILITY_RESTORATION)\]\s*', 
                         '', user_query, flags=re.IGNORECASE)
    
    # Look for explicit model mentions (prefer longer matches to avoid partial matches)
    for version in sorted(models_dictionary.keys(), key=lambda v: len(v), reverse=True):
        if version in query_clean and "__" not in version:
            return version
    
    # Default to first base model
    for version in models_dictionary.keys():
        if "__" not in version:
            return version

    return None


def _load_infeasibility_diagnosis_report(model_name: str) -> Optional[dict]:
    """
    Load infeasibility diagnosis report from <workspace>/inf_detail/{model_name}_inf_detail.json.

    Args:
        model_name: Base model name (e.g., 'diet_inf_1', 'aircraft_inf_1')

    Returns:
        Dictionary with diagnosis data, or None if file doesn't exist
    """
    report_path = os.path.join(workspace.inf_detail_dir(), f"{model_name}_inf_detail.json")

    if not os.path.exists(report_path):
        logger.warning(f"No diagnosis report found at: {report_path}")
        return None

    try:
        with open(report_path, 'r') as f:
            diagnosis_data = json.load(f)
        logger.info(f"✓ Loaded diagnosis report for {model_name}")
        return diagnosis_data
    except Exception as e:
        logger.error(f"Failed to load diagnosis report from {report_path}: {e}")
        return None


def _format_diagnosis_for_prompt(diagnosis_data: dict) -> str:
    """
    Format the diagnosis report JSON into a readable prompt section.
    Only works with data from {model}_inf_detail.json.

    Args:
        diagnosis_data: Dictionary loaded from {model}_inf_detail.json

    Returns:
        Formatted string for prompt injection
    """
    if not diagnosis_data:
        return "(No prior diagnosis report available)"

    sections = []
    sections.append("## PRIOR INFEASIBILITY DIAGNOSIS REPORT")
    sections.append("")

    # Add model name
    if "source_model_name" in diagnosis_data:
        sections.append(f"**Model:** {diagnosis_data['source_model_name']}")

    # Add diagnosis status and resolution info
    if "diagnosis_status" in diagnosis_data:
        sections.append(f"**Diagnosis Status:** {diagnosis_data['diagnosis_status']}")
    if "resolution_round" in diagnosis_data:
        sections.append(f"**Resolution Round:** {diagnosis_data['resolution_round']}")
    if "solution_status" in diagnosis_data:
        sections.append(f"**Solution Status after relaxation:** {diagnosis_data['solution_status']}")
    if "objective_value" in diagnosis_data:
        sections.append(f"**Objective Value after relaxation:** {diagnosis_data['objective_value']}")

    # Add violated constraints
    if "violated_constraints" in diagnosis_data and diagnosis_data["violated_constraints"]:
        sections.append("\n**Violated Constraints (IIS):**")
        for constraint in diagnosis_data["violated_constraints"]:
            sections.append(f"  - {constraint}")

    # Add slack values
    if "slack_values" in diagnosis_data and diagnosis_data["slack_values"]:
        sections.append("\n**Slack Values (required relaxation per constraint):**")
        for constraint, slack in diagnosis_data["slack_values"].items():
            sections.append(f"  - {constraint}: {slack}")

    # Add full diagnosis text
    if "full_diagnosis_text" in diagnosis_data and diagnosis_data["full_diagnosis_text"]:
        sections.append("\n**Full Diagnosis:**")
        sections.append(diagnosis_data["full_diagnosis_text"])

    sections.append("")
    sections.append("**IMPORTANT:** Use this diagnosis to guide your feasibility restoration approach.")
    sections.append("You should NOT call the infeasibility_diagnosis tool again unless the user explicitly requests a fresh diagnosis.")

    return "\n".join(sections)

def check_llm_response(callback_context: CallbackContext, llm_response: LlmResponse):
    agent_name = callback_context.agent_name

    # Calculate LLM call duration
    llm_start_time = callback_context.state.get(LLM_CALL_START_TIME)
    if llm_start_time:
        llm_duration = time.time() - llm_start_time

        # Increment LLM call counter for this agent
        llm_call_counters = callback_context.state.get(LLM_CALL_COUNTER, {})
        call_number = llm_call_counters.get(agent_name, 0) + 1
        llm_call_counters[agent_name] = call_number
        callback_context.state[LLM_CALL_COUNTER] = llm_call_counters

        # Record in timing tracker
        timing_tracker.record_llm_call(agent_name, llm_duration, call_number)

    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            logger.info((f"[Callback] Inspecting LLM response from '{agent_name}': "
                         f"{original_text}"))
        elif llm_response.content.parts[0].function_call:
            fc = llm_response.content.parts[0].function_call
            logger.info((f"[Callback] Inspecting LLM function call from '{agent_name}': "
                         f"{fc.name} | args: {dict(fc.args)}"))
        else:
            logger.info("[Callback] Inspected LLM response: No text content found.")
    elif llm_response.error_message:
        logger.error((f"[Callback] Inspected LLM response: "
                      f"Contains error '{llm_response.error_message}'. "))
    else:
        logger.warning("[Callback] Inspected LLM response: Empty LlmResponse.")

    if agent_name == "root_agent" and llm_response.content and llm_response.content.parts:
        has_route_call = any(
            p.function_call and p.function_call.name == "route_to_expert"
            for p in llm_response.content.parts
        )
        if has_route_call:
            function_call_parts = [p for p in llm_response.content.parts if p.function_call]
            seen_route = False
            deduplicated_parts = []
            for p in function_call_parts:
                if p.function_call and p.function_call.name == "route_to_expert":
                    if seen_route:
                        logger.warning("[Root callback] Dropped duplicate route_to_expert call — only one allowed per response")
                        continue
                    seen_route = True
                deduplicated_parts.append(p)
            stripped_response = LlmResponse(
                content=types.Content(role="model", parts=deduplicated_parts)
            )
            return stripped_response

    return None


def route_to_expert(analysis_type: str, tool_context: ToolContext) -> str:
    """Classify the current query and immediately transfer control to expert_agent.

    Call this once for every non-[GENERAL] query instead of writing an
    ANALYSIS_TYPE marker and calling transfer_to_agent separately.

    Args:
        analysis_type: One of RETRIEVAL, SENSITIVITY, WHAT_IF, WHY_NOT,
                       FEASIBILITY_RESTORATION, ROBUSTNESS.
        tool_context: Injected by ADK — do not pass manually.

    Returns:
        Confirmation string (not shown to user; transfer fires immediately).
    """
    valid_types = {"RETRIEVAL", "SENSITIVITY", "WHAT_IF", "WHY_NOT",
                   "FEASIBILITY_RESTORATION", "ROBUSTNESS"}
    analysis_type = analysis_type.strip().upper()
    if analysis_type not in valid_types:
        return (f"Invalid analysis_type '{analysis_type}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}")
    tool_context.state[CURRENT_ANALYSIS_TYPE] = analysis_type
    logger.info(f"[route_to_expert] CURRENT_ANALYSIS_TYPE set to {analysis_type}")
    # Trigger ADK transfer to expert_agent via the function response event actions
    tool_context.actions.transfer_to_agent = "expert_agent"
    return f"Routing to expert_agent with analysis_type={analysis_type}."


def check_tool_usage(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    agent_name = tool_context.agent_name
    tool_name = tool.name

    # Start timing for tool call
    tool_context.state[TOOL_CALL_START_TIME] = time.time()

    usage_key = f"{agent_name.upper()}_{tool_name.upper()}_USES"
    if usage_key in tool_context.state:
        uses_left = tool_context.state[usage_key]
        if uses_left <= 0:
            logger.debug(f"Usage key '{usage_key}' has no remaining uses.")
            return {"result": f"\n[system message]: **WARNING** '{tool_name}' tool cannot be used anymore! "}
        else:
            tool_context.state[usage_key] -= 1
            logger.debug(f"Usage key '{usage_key}' decremented. Remaining uses: {tool_context.state[usage_key]}")
    else:
        logger.debug(f"Usage key '{usage_key}' not found in the state. Skipping tool usage check.")
    return None


def check_tool_response(tool: BaseTool,
                        args: Dict[str, Any],
                        tool_context: ToolContext,
                        tool_response: Dict):
    show_first_n_chars = None
    agent_name = tool_context.agent_name
    tool_name = tool.name

    # Calculate tool call duration
    tool_start_time = tool_context.state.get(TOOL_CALL_START_TIME)
    if tool_start_time:
        tool_duration = time.time() - tool_start_time

        # Record in timing tracker
        # For AgentTool (expert_agent, illustrator_agent), this is delegation time
        if tool_name in ["expert_agent", "illustrator_agent"]:
            timing_tracker.record_delegation(agent_name, tool_name, tool_duration)
        else:
            timing_tracker.record_tool_call(agent_name, tool_name, tool_duration)

    # AgentTool may return str instead of Dict as tool_response
    if isinstance(tool_response, dict):
        result = tool_response.get("result", "")
    else:
        result = str(tool_response)

    max_tokens_key = f"{agent_name.upper()}_{tool_name.upper()}_MAX_TOKENS"
    if max_tokens_key in tool_context.state:
        max_tokens = tool_context.state[max_tokens_key]
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(result)
            token_count = len(tokens)
            if token_count > max_tokens:
                # truncate the result to max_tokens
                truncated_tokens = tokens[:max_tokens]
                truncated_result = encoding.decode(truncated_tokens)
                truncated_result += ("... \n[system message]: **WARNING** "
                                     "Execution result was truncated due to token limit.")
                logger.warning((f"'{tool_name.upper()}' execution (truncated) result: {truncated_result[:show_first_n_chars]}"
                                "\n... (showing only the first {show_first_n_chars} characters)"))
                # return a truncated tool_response dictionary or string
                if isinstance(tool_response, dict):
                    truncated_tool_response = deepcopy(tool_response)
                    truncated_tool_response["result"] = truncated_result
                    return truncated_tool_response
                else:
                    return truncated_result
        except Exception as e:
            raise RuntimeError(f"Token counting failed: {e}.")
    else:
        logger.debug(f"max tokens key '{max_tokens_key}' not found in the state. Skipping tool response check.")
    
    # Log the result - only show truncation message if actually truncating
    if show_first_n_chars is not None:
        logger.info(f"'{tool_name.upper()}' execution result: {result[:show_first_n_chars]}"
                    f"\n... (showing only the first {show_first_n_chars} characters)")
    else:
        logger.info(f"'{tool_name.upper()}' execution result: {result}")
    
    return None  # Return None to indicate no modification to tool_response

def start_illustrator_timing(callback_context: CallbackContext):
    timing_tracker.record_agent_start("illustrator_agent")
    return None


def check_illustrator_agent_runtime(callback_context: CallbackContext):
    """
    Record illustrator agent runtime after agent completes.

    Args:
        callback_context: Callback context with state

    Returns:
        None
    """
    timing_tracker.record_agent_end("illustrator_agent")
    return None


def check_root_agent_runtime(callback_context: CallbackContext):
    """
    Record root agent runtime and print summary after agent completes.

    Args:
        callback_context: Callback context with state

    Returns:
        None
    """
    timing_tracker.record_agent_end("root_agent")

    # End query timing and print summary
    timing_tracker.get_tracker().end_query()
    timing_tracker.print_summary()

    # Optionally save to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = str(workspace.timing_dir())
    os.makedirs(output_dir, exist_ok=True)
    timing_file = os.path.join(output_dir, f"timing_{timestamp}.json")
    timing_tracker.save_to_file(timing_file)

    # Forward expert's response to root's output key when expert responded directly via transfer
    expert_output = callback_context.state.get(OUTPUT_KEY_EXPERT_AGENT, "")
    if expert_output:
        root_output = callback_context.state.get(OUTPUT_KEY_ROOT_AGENT, "")
        if not root_output:
            callback_context.state[OUTPUT_KEY_ROOT_AGENT] = expert_output
            logger.info("[Root callback] Forwarded expert output to root output key")

    return None


