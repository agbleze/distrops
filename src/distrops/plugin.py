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