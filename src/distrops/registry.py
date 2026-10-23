from typing import Callable, Dict, Any


class Registry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        