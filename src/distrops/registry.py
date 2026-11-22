from typing import Callable, Dict, Any, List
import importlib



class Registry:
    def __init__(self, registry_name: str = "DEFAULT"):
        self.registry_name = registry_name
        self._registry = {}
        self._registry.setdefault(registry_name, {})
        
        
    def register_component(self, name: str, cls: Any, registry_name=None):
        """_summary_

        Args:
            name (str): name of the component -function or class to register
            cls (Any): Object to register - function or class

        Raises:
            ValueError: Raise when the name is already registered in the registry
        
        Returns:
            None: _description_
        """
        registry_name = self.registry_name if not registry_name else registry_name
        if registry_name in self.list_available_registry(): #self._registry[registry_name]:
            if name in self._registry[registry_name]:
                raise ValueError(f"{name} is already in {registry_name} registry")
        else:
            self._registry[registry_name] = {}
        self._registry[registry_name][name] = cls
        
    def register(self, name: str = None, registry_name: str = None) -> Callable:
        """@decorator hook interface

        Args:
            name (str, optional): Name of the component - function or class. Defaults to None.

        Returns:
            Callable: Function or class that is being registered in the registry.
        """
        registry_name = self.registry_name if not registry_name else registry_name
        self.name = name
        
        def decorator(cls_or_func: Any) -> Callable:
            name = self.name or cls_or_func.__name__
            self.register_component(name=name, cls=cls_or_func, registry_name=registry_name)
            return cls_or_func
        return decorator
    
    def get(self, name_or_path: str, registry_name: str = None) -> Any:
        """Get component from regisry or dynamically via a dot-separated python import path!

        Args:
            name (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            Any: _description_
        """
        registry_name = self.registry_name if not registry_name else registry_name
                
        if registry_name in self._registry and name_or_path in self._registry[registry_name]:
            return self._registry[registry_name][name_or_path]

        if "." in name_or_path:
            # module_path, attr_name = name_or_path.rsplit(".", 1)
            # module = importlib.import_module(module_path)
            # return getattr(module, attr_name)
            return resolve_dot_path(dot_path=name_or_path)
        
        available = self.list_available_registry_items(registry_name=registry_name)
        
        
        raise KeyError(
            f"{name_or_path} not found in {registry_name} registry and is not a valid import path. "
            f"Registered in {registry_name}: {available}"
        )
        
        
    def list_available_registry(self) -> List:
        return list(self._registry.keys())
    
    def list_available_registry_items(self, registry_name: None):
        registry_name = self.registry_name if not registry_name else registry_name
        registry_items = list(self._registry.get(registry_name, {}).keys())
        return registry_items
        

def resolve_dot_path(dot_path: str):
    """
    Resolves arbitrary nested dot paths like:
        - 'torch.nn.CrossEntropyLoss' (module.Class)
        - 'transformers.AutoModelForSequenceClassification.from_pretrained' (module)
    """
    parts = dot_path.split(".")
    
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_name)
            
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except ModuleNotFoundError:
            continue
    raise ImportError(f"Could not import any module from path: '{dot_path}'")
    
            
DATASET_REGISTRY = Registry("DATASET_REGISTRY")
MODEL_REGISTRY = Registry("MODEL_REGISTRY")


registry = Registry()