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
        
def _load_from_file_path(file_path: str):
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Plugin file not found at: '{abs_path}'")

    module_name = f"distrops_plugin_{os.path.splitext(os.path.basename(abs_path))[0]}"
    spec = importlib.util.spec_from_file_location(name=module_name, location=abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for plugin file: '{abs_path}'")
    
    module = importlib.util.module_from_spec(spec=spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module=module)
    print(f"Loaded plugin file: '{file_path}'")
    