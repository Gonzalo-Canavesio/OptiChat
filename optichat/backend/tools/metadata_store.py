import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Import constants
from optichat.backend.config import workspace


def load_metadata(metadata_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load lightweight metadata index for all models.

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Dictionary with structure:
        {
            "YYYY-MM-DD": {
                "base_model_name": {
                    "solution_status": "...",
                    "pickle_path": "...",
                    "data_path": "...",
                    "modified_models": {
                        "modified_model_name": {
                            "description": "...",
                            "solution_status": "...",
                            ...
                        }
                    }
                }
            }
        }
    """
    metadata_path = metadata_path or workspace.metadata_path()
    if not os.path.exists(metadata_path):
        logger.info(f"No metadata file found at {metadata_path}, starting fresh")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return {}


def save_metadata(metadata: Dict[str, Any], metadata_path: Optional[str] = None):
    """
    Save lightweight metadata index to file.

    Args:
        metadata: Complete metadata dictionary
        metadata_path: Path to save metadata.json
    """
    metadata_path = metadata_path or workspace.metadata_path()
    try:
        # Ensure directory exists
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to file with pretty formatting
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata to {metadata_path}: {e}")


def load_model_data(version: str, data_folder: Optional[str] = None) -> Dict[str, Any]:
    """
    Load full model component data for a specific version.

    Args:
        version: Model version name
        data_folder: Directory containing model data files

    Returns:
        Dictionary with complete model information (same structure as models_dictionary entry)
    """
    data_folder = data_folder or workspace.model_data_dir()
    data_path = os.path.join(data_folder, f"{version}.json")

    if not os.path.exists(data_path):
        logger.error(f"Model data file not found: {data_path}")
        raise FileNotFoundError(f"Model data file not found for version: {version}")

    try:
        with open(data_path, 'r') as f:
            model_data = json.load(f)
        logger.info(f"Loaded model data for {version} from {data_path}")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model data from {data_path}: {e}")
        raise


def save_model_data(version: str, model_info: Dict[str, Any], data_folder: Optional[str] = None):
    """
    Save full model component data to individual file.

    Args:
        version: Model version name
        model_info: Complete model information dictionary
        data_folder: Directory to save model data files
    """
    data_folder = data_folder or workspace.model_data_dir()
    try:
        # Ensure directory exists
        Path(data_folder).mkdir(parents=True, exist_ok=True)

        data_path = os.path.join(data_folder, f"{version}.json")

        # Write to file with pretty formatting
        with open(data_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Saved model data for {version} to {data_path}")
    except Exception as e:
        logger.error(f"Error saving model data to {data_folder}/{version}.json: {e}")


def extract_metadata_from_model_info(
    model_info: Dict[str, Any],
    version_name: str,
    description: Optional[str] = None,
    source_file_path: Optional[str] = None,
    repl_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract lightweight metadata from full model info.

    Args:
        model_info: Complete model information dictionary
        version_name: Name/version of the model
        description: Human-readable description (optional)

    Returns:
        Lightweight metadata dictionary
    """
    # Extract solution status
    obj_info = model_info.get("obj", {})
    solution_status = obj_info.get("sol_status", "unknown")

    # Get pickle path
    pickle_path = model_info.get("local_path_to_object", "")

    # Generate data path
    data_path = os.path.join(workspace.model_data_dir(), f"{version_name}.json")

    metadata = {
        "solution_status": solution_status,
        "pickle_path": pickle_path,
        "data_path": data_path
    }

    if description:
        metadata["description"] = description
    
    if source_file_path:
        metadata["source_file_path"] = source_file_path
    
    if repl_code:
        metadata["repl_code"] = repl_code

    return metadata


def add_model_to_metadata(
    metadata: Dict[str, Any],
    version_name: str,
    model_info: Dict[str, Any],
    base_model: Optional[str] = None,
    description: Optional[str] = None,
    source_file_path: Optional[str] = None,
    repl_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add or update a model entry in metadata with tree structure.

    Args:
        metadata: The metadata dictionary to update
        version_name: Name/version of the model
        model_info: Complete model information dictionary
        base_model: Name of base model (None for initial model)
        description: Human-readable description

    Returns:
        Updated metadata dictionary
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Initialize date entry if not exists
    if today not in metadata:
        metadata[today] = {}

    # Extract basic metadata
    model_metadata = extract_metadata_from_model_info(
        model_info=model_info,
        version_name=version_name,
        description=description,
        source_file_path=source_file_path,
        repl_code=repl_code
    )

    if base_model is None:
        # It's a base model
        if version_name not in metadata[today]:
            metadata[today][version_name] = {}
        
        # Update fields
        metadata[today][version_name].update(model_metadata)
        
        # Ensure modified_models dict exists
        if "modified_models" not in metadata[today][version_name]:
            metadata[today][version_name]["modified_models"] = {}
            
    else:
        # It's a modified model
        # Ensure base model exists in metadata (might be from a previous date or today)
        # We need to find where the base model is located
        base_model_date = None
        
        # Check today first
        if base_model in metadata[today]:
            base_model_date = today
        else:
            # Check other dates
            for date_key in metadata:
                if base_model in metadata[date_key]:
                    base_model_date = date_key
                    break
        
        # If base model not found, we treat it as a new base model for today (fallback)
        if base_model_date is None:
            logger.warning(f"Base model {base_model} not found in metadata. Creating as new base model entry.")
            # This shouldn't happen ideally if base model was saved correctly
            # But for robustness, we'll create a placeholder base model entry
            metadata[today][base_model] = {
                "solution_status": "unknown",
                "modified_models": {}
            }
            base_model_date = today

        # Add to modified_models of the base model
        if "modified_models" not in metadata[base_model_date][base_model]:
            metadata[base_model_date][base_model]["modified_models"] = {}
            
        metadata[base_model_date][base_model]["modified_models"][version_name] = model_metadata

    logger.info(f"Added metadata for model: {version_name}")
    return metadata
