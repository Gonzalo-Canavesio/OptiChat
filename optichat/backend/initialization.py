import glob
import os
import traceback
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from loguru import logger

from optichat.backend.config import workspace
from optichat.backend.config.constants import (
    CFG,
    CHAT_MODELING_LANGUAGE,
    HISTORICAL_MODELS_METADATA,
    IS_MODELS_DICTIONARY_AVAILABLE,
    MODEL_VERSIONS,
    MODELS_DICTIONARY,
    PERSISTENT_STATES,
)
from optichat.backend.tools.callback_tool import update_selected_model_state
from optichat.backend.tools.extract_tool import (
    _solve_model,
    extract_model_info,
    restore_model_object,
    save_model_object,
)
from optichat.backend.tools.gamspy_extractor import (
    extract_gamspy_model_info,
    save_gamspy_model,
    solve_gamspy_model,
)
from optichat.backend.tools.metadata_store import (
    add_model_to_metadata,
    load_metadata,
    save_metadata,
    save_model_data,
)


class ModelLoadError(Exception):
    """None of the chat's models could be loaded."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(errors.values()) or "no model was found")


@dataclass
class InitializedChat:
    state: dict[str, Any]
    """Initial session state of the chat."""
    diagnosis: dict[str, Any] | None
    """Infeasibility diagnosis of the primary model, if it is infeasible. It is only
    input for the illustrator agent, so it is not part of the session state."""


def initialize_chat(cfg: dict) -> InitializedChat:
    """Load the models in `cfg`, diagnose the primary one if it is infeasible and
    return the chat's initial state. Raises ModelLoadError if no model loads."""
    cfg = _expand_cfg(cfg)
    models_dictionary, model_versions, _, load_errors = load_models(cfg)
    if not model_versions:
        raise ModelLoadError(load_errors)

    state = deepcopy(PERSISTENT_STATES)
    state[CFG] = cfg
    state[MODELS_DICTIONARY] = models_dictionary
    # Only the models from the config are shown to the root agent, not historical ones
    state[MODEL_VERSIONS] = model_versions
    state[CHAT_MODELING_LANGUAGE] = models_dictionary[model_versions[0]][
        "modeling_language"
    ]

    diagnosis = _diagnose_if_infeasible(model_versions[0], state)

    # Read after the diagnosis, which registers the relaxed model versions it creates
    historical_metadata = load_metadata()
    state[HISTORICAL_MODELS_METADATA] = historical_metadata
    state["MODELS_METADATA_FORMATTED"] = format_models_metadata_for_prompt(
        historical_metadata
    )
    state[IS_MODELS_DICTIONARY_AVAILABLE] = True
    update_selected_model_state(model_versions, historical_metadata, state)
    return InitializedChat(state=state, diagnosis=diagnosis)


def _diagnose_if_infeasible(model_name: str, state: dict) -> dict[str, Any] | None:
    status = str(
        state[MODELS_DICTIONARY][model_name].get("obj", {}).get("sol_status", "unknown")
    )
    if status not in ("infeasible", "infeasibleOrUnbounded"):
        return None

    from optichat.backend.tools.custom_tool import infeasibility_diagnosis

    logger.info(f"Model '{model_name}' is infeasible. Running diagnosis...")
    try:
        # The diagnosis only uses the context's state, which it updates in place
        return infeasibility_diagnosis(model_name, SimpleNamespace(state=state))
    except Exception as e:
        logger.exception("Infeasibility diagnosis failed")
        return {"status": "error", "result": f"Diagnosis failed: {e}"}


def format_models_metadata_for_prompt(historical_metadata: dict) -> str:
    """
    Format model metadata into human-readable string for prompt display.

    Args:
        historical_metadata: Dictionary of model metadata (tree structure)

    Returns:
        Formatted string with one line per model showing key information

    Example output:
        - supply_chain_model_3: Base model: supply_chain_model_3 (optimal)
        - supply_chain_model_3__demand_3_1_plus10: What-if analysis: increased demand[3,0] by 10. Objective improved to 45.2. (optimal)
    """
    if not historical_metadata:
        return "    No models available"

    formatted_lines = []

    # Iterate through dates (sorted reverse)
    for date_key in sorted(historical_metadata.keys(), reverse=True):
        date_models = historical_metadata[date_key]

        # Iterate through base models
        for base_name, base_data in date_models.items():
            # Add base model
            description = base_data.get("description", "Base model")
            status = base_data.get("solution_status", "unknown")
            line = f"    - {base_name}: {description} ({status})"
            formatted_lines.append(line)

            # Add modified models
            modified_models = base_data.get("modified_models", {})
            for mod_name, mod_data in modified_models.items():
                mod_desc = mod_data.get("description", "No description")
                mod_status = mod_data.get("solution_status", "unknown")
                mod_line = f"    - {mod_name}: {mod_desc} ({mod_status})"
                formatted_lines.append(mod_line)

    return "\n".join(formatted_lines)


def _find_and_copy_source_file(version_name: str, cfg: dict) -> str | None:
    """
    Find the .py source file for this model and copy it to tmp folder.

    Strategy:
    1. Check models_code paths in config for {version}.py
    2. If found, copy to <workspace>/model_objects/{version}.py
    3. Return the copied path

    Args:
        version_name: Model version name (e.g., 'RTN_inf_1')
        cfg: Configuration dictionary

    Returns:
        Path to the copied .py file, or None if not found
    """
    import shutil

    if "models_code" not in cfg:
        logger.debug(
            f"No models_code in config, skipping source file copy for {version_name}"
        )
        return None

    # Look for source file in config paths
    code_paths = cfg["models_code"].get("local_resources", [])
    source_file = None

    for path in code_paths:
        if "*" in path:
            # Handle wildcard paths (e.g., "examples/*/*.py")
            matches = glob.glob(path.replace("*", f"{version_name}"))
            if matches:
                source_file = matches[0]
                logger.debug(f"Found source file via wildcard: {source_file}")
                break
        elif version_name in path and path.endswith(".py"):
            # Direct path match
            source_file = path
            logger.debug(f"Found source file via direct match: {source_file}")
            break

    if not source_file or not os.path.exists(source_file):
        logger.warning(f"No .py source file found for {version_name}")
        return None

    # Copy to <workspace>/model_objects
    model_dir = workspace.model_objects_dir()
    os.makedirs(model_dir, exist_ok=True)

    # Directories for infeasibility diagnosis reports and robustness analysis outputs
    os.makedirs(workspace.inf_detail_dir(), exist_ok=True)
    os.makedirs(workspace.robust_dir(), exist_ok=True)

    dest_path = os.path.join(model_dir, f"{version_name}.py")

    try:
        shutil.copy2(source_file, dest_path)
        logger.info(f"Copied source file: {source_file} -> {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy source file {source_file}: {e}")
        return None


def _find_model(module, py_file_path: str):
    """
    The model a .py file defines, and its modeling language ("Pyomo" or "GAMSPy").

    Pyomo: the module's `model` if it is a ConcreteModel, else its only ConcreteModel.
    GAMSPy: the module's only Model.
    """
    import gamspy as gp
    import pyomo.environ as pyo

    file_name = os.path.basename(py_file_path)
    values = vars(module).values()
    pyomo_models = {id(v): v for v in values if isinstance(v, pyo.ConcreteModel)}
    gamspy_models = {id(v): v for v in values if isinstance(v, gp.Model)}

    if pyomo_models and gamspy_models:
        raise ValueError(
            f"{file_name} defines both Pyomo and GAMSPy models; it must define one model."
        )
    if gamspy_models:
        if len(gamspy_models) > 1:
            raise ValueError(
                f"{file_name} must define one GAMSPy Model; found {len(gamspy_models)}."
            )
        return next(iter(gamspy_models.values())), "GAMSPy"
    candidate = getattr(module, "model", None)
    if isinstance(candidate, pyo.ConcreteModel):
        return candidate, "Pyomo"
    if len(pyomo_models) == 1:
        return next(iter(pyomo_models.values())), "Pyomo"
    if not pyomo_models:
        raise ValueError(
            f"{file_name} must define a Pyomo ConcreteModel or a GAMSPy Model; found neither."
        )
    raise ValueError(
        f"{file_name} defines {len(pyomo_models)} Pyomo ConcreteModels; "
        "name the one to analyze `model`."
    )


def _load_model_from_py(py_file_path: str):
    """
    Load the optimization model defined by a .py file.

    The file runs from its own directory (cwd and sys.path), so it reads the data
    files next to it with relative paths, in whatever format and way it chooses;
    nothing is injected into its namespace.

    Args:
        py_file_path: Path to the .py file containing the model definition

    Returns:
        Tuple of (model, version_name, modeling_language)
    """
    import importlib.util
    import sys

    base_dir = os.path.dirname(os.path.abspath(py_file_path))

    # Create module from file.
    # IMPORTANT: do NOT register in sys.modules before pickling.
    # If cloudpickle sees the module in sys.modules it stores a lightweight
    # reference ("module: loaded_model, fn: supply_rule") instead of the actual
    # bytecode. That reference breaks across server restarts because sys.modules
    # is wiped. By keeping the module out of sys.modules, cloudpickle is forced
    # to serialize the full function bytecode into the .pkl, making it truly
    # self-contained and loadable in future sessions without needing the .py file.
    spec = importlib.util.spec_from_file_location("loaded_model", py_file_path)
    loaded_module = importlib.util.module_from_spec(spec)

    modules_before = set(sys.modules)
    previous_cwd = os.getcwd()
    sys.path.insert(0, base_dir)
    os.chdir(base_dir)
    try:
        spec.loader.exec_module(loaded_module)
    finally:
        os.chdir(previous_cwd)
        sys.path.remove(base_dir)
        # Helper modules the model imported from its directory get the same treatment
        # as the model module (pickled by value), and are not reused by another chat
        # that has a different file with the same name.
        for name in set(sys.modules) - modules_before:
            module_file = getattr(sys.modules[name], "__file__", None) or ""
            if os.path.abspath(module_file).startswith(base_dir + os.sep):
                del sys.modules[name]

    model, modeling_language = _find_model(loaded_module, py_file_path)
    version = os.path.splitext(os.path.basename(py_file_path))[0]

    return model, version, modeling_language


def _describe_load_error(error: Exception, resource_path: str) -> str:
    """'<file>, line N: Type: message', pointing at the model's own line when it raised."""
    model_file = os.path.abspath(resource_path)
    frames = [
        f
        for f in traceback.extract_tb(error.__traceback__)
        if os.path.abspath(f.filename) == model_file
    ]
    where = os.path.basename(resource_path)
    if frames:
        where += f", line {frames[-1].lineno}"
    return f"{where}: {type(error).__name__}: {error}"


def load_models(cfg: dict):
    """
    Initialize models with new architecture:
    - Load lightweight metadata for all models
    - Only load current models (from config) into models_dictionary
    - Historical models loaded lazily on demand

    Returns:
        models_dictionary: Runtime cache with current models only
        current_model_versions: List of current model versions (shown to root agent)
        historical_metadata: Dict of all models metadata (for lazy loading)
        load_errors: {model file name: description of why it could not be loaded}
    """
    # STEP 1: Load existing metadata (lightweight, always loaded)
    metadata = load_metadata()
    # In new structure, metadata IS the historical metadata (tree structure)
    historical_metadata = metadata

    if historical_metadata:
        logger.info(f"Loaded metadata with {len(historical_metadata)} date entries")
    else:
        logger.info("No existing metadata found, starting fresh")

    # STEP 2: Initialize current models from config
    models_dictionary = {}  # Runtime cache for loaded models
    load_errors = {}  # model file name -> why it could not be loaded
    current_model_versions = []  # Models from config (shown to root agent)

    if "models" in cfg:
        is_solved = cfg["models"].get("is_solved", False)
        is_lp = cfg["models"].get("is_lp", False)

        for resource_path in cfg["models"].get("local_resources", []):
            try:
                # Check if this is .pkl (backward compat) or .py (new format)
                if resource_path.endswith(".pkl"):
                    # OLD PATH: Load from pickle
                    logger.info(f"Loading model from pickle: {resource_path}")
                    model, version = restore_model_object(resource_path)
                    modeling_language = "Pyomo"
                elif resource_path.endswith(".py"):
                    # The model reads its own data files from its directory
                    logger.info(f"Loading model from Python file: {resource_path}")
                    model, version, modeling_language = _load_model_from_py(
                        resource_path
                    )
                else:
                    raise ValueError(
                        f"Unsupported model file format: {resource_path}. Must be .pkl or .py"
                    )
                logger.info(f"Detected modeling language: {modeling_language}")

                if modeling_language == "GAMSPy":
                    termination_condition = solve_gamspy_model(model)
                    info = extract_gamspy_model_info(model, termination_condition)
                    info.update(
                        {
                            "local_path_to_object": save_gamspy_model(model, version),
                            "gamspy_model_name": model.name,
                        }
                    )
                else:
                    # Solve model (same for both formats)
                    model, termination_condition = _solve_model(
                        model, is_lp=is_lp, is_solved=is_solved
                    )

                    # Extract model information
                    info = extract_model_info(
                        model, termination_condition=termination_condition
                    )
                    local_path_to_object = save_model_object(model, version)
                    info.update({"local_path_to_object": local_path_to_object})
                info["modeling_language"] = modeling_language

                # NEW: Copy .py source file to tmp if it exists
                source_py_path = _find_and_copy_source_file(version, cfg)
                if source_py_path:
                    info.update({"source_file_path": source_py_path})

                # Save full model data to individual file
                save_model_data(version, info)

                # Add to runtime cache
                models_dictionary[version] = info
                current_model_versions.append(version)

                # Add/update metadata (base models get automatic description)
                metadata = add_model_to_metadata(
                    metadata=metadata,
                    version_name=version,
                    model_info=info,
                    base_model=None,
                    description=None,  # Will auto-generate "Base model: {version}"
                    source_file_path=source_py_path,  # NEW: track .py file path
                )

                # Update historical_metadata dict
                historical_metadata = metadata

                logger.info(f"✓ Successfully loaded model: {version}")

            except Exception as e:
                logger.error(f"Failed to load model from {resource_path}: {e}")
                logger.exception("Full traceback:")
                load_errors[os.path.basename(resource_path)] = _describe_load_error(
                    e, resource_path
                )
                # Continue to next model instead of failing entire initialization
                continue

        logger.info(f"Initialized {len(models_dictionary)} model(s) from config")

    # STEP 3: Save updated metadata
    if metadata:
        save_metadata(metadata)
        logger.info("Saved metadata")

    logger.info(f"Current models (shown to root agent): {current_model_versions}")

    return models_dictionary, current_model_versions, historical_metadata, load_errors


def _expand_cfg(cfg: dict):
    cfg_out = deepcopy(cfg)

    required_sections = ["models", "models_code"]
    extension_filters = {
        "models": [
            ".pkl",
            ".py",
        ],  # Support both .pkl (old) and .py (new separated format)
        "models_code": [".py"],
    }

    for section_key, section_cfg in cfg.items():
        if section_key in required_sections:
            local_resources = section_cfg.get("local_resources", [])
            allowed_extensions = extension_filters.get(section_key, None)
            expanded_resources = _expand_resources(local_resources, allowed_extensions)
            cfg_out[section_key]["local_resources"] = expanded_resources
    return cfg_out


def _expand_resources(local_resources: list[str], allowed_extensions: list[str] | None):
    """
    Expand wildcard * patterns with allowed extensions filtering
    """
    expanded_resources = []
    for resource_path in local_resources:
        if "*" in resource_path:
            matches = glob.glob(resource_path)
        else:
            matches = [resource_path]
        if not matches:
            raise FileNotFoundError(f"No files matched: {resource_path}")
        if allowed_extensions:
            filtered_matches = [
                match
                for match in matches
                if any(match.lower().endswith(ext) for ext in allowed_extensions)
            ]
        else:
            filtered_matches = matches
        expanded_resources.extend(sorted(filtered_matches))
    return expanded_resources
