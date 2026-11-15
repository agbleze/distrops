import importlib
import importlib.util
import os
import sys
from typing import List


def load_plugins(plugin_paths: List[str]) -> None:
    if not plugin_paths:
        return
    
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        
    for target in plugin_paths:
        if target.endswith(".py") or os.path.sep in target:
            _load_from_file_path(target)
        else:
            _load_from_module_name(target)
            
            
def _load_from_module_name(module_name: str):
    try:
        importlib.import_module(module_name)
        print(f"Loaded plugin module: {module_name}")
    except ImportError as e:
        raise ImportError(
            f"Failed to import plugin module '{module_name}'."
            f"Ensure it is installed or in PYTHONPATH. Original error: {e}"
        ) from e