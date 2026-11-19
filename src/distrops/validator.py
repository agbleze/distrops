import inspect
from typing import get_type_hints
from registry import registry
 
class PipelineValidationError(Exception):
    pass

FRAMEWORK_INJECTED_PARAMS = {"self", "config", "params", "model_parameters"}


def _check_callable_signature(callable_obj: Any, available_keys: set,
                              stage_label: str,
                              ignore_params: set = None
                              ) -> None:
    """Inspects a callable's required parameters against available configuration

    Args:
        callable_obj (Any): _description_
        available_keys (set): _description_
        stage_label (str): _description_
        ignore_params (set, optional): _description_. Defaults to None.
    """
    ignore_params = ignore_params or FRAMEWORK_INJECTED_PARAMS
    sig = inspect.signature(callable_obj)
    missing_params = []
    for param_name, param in sig.parameters.items():
        if param_name in ignore_params:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is inspect.Parameter.empty and param_name not in available_keys:
            missing_params.append(param_name)
            
    if missing_params:
        raise PipelineValidationError(
            f"Contract Breach in {stage_label}: Missing required arguments {missing_params}. "
            f"Neither defined in 'params' nor produced by upstream stages."
        )


def _resolve_target_callable(callable_obj: Any) -> Any:
    """Returns __call__ if inspecting a class instance, or __init__ if inspecting a class

    Args:
        callable_obj (Any): _description_

    Returns:
        Any: _description_
    """
    if inspect.isclass(callable_obj):
        return callable_obj.__init__
    if callable(callable_obj) and not inspect.isfunction(callable_obj) and not inspect.ismethod(callable_obj):
        return callable_obj.__call__
    return callable_obj

    
def validate_pipeline_dag(global_config: dict, registry_name="workflows") -> bool:
    workflows = global_config.get("workflows", [])
    if not workflows:
        raise PipelineValidationError(f"Configuration manifest contains no 'workflows' steps")
        
    available_keys = set(global_config.keys())
    strict_validation = True
    
    for idx, step in enumerate(workflows):
        step_name = step.get("name")
        step_params = step.get("params", {})
        stage_label = f"Stage {idx + 1} ('{step_name}')"
        
        try:
            workflow_cls = registry.get(name_or_path=step_name, registry_name=registry_name)
        except (KeyError, ImportError, AttributeError) as e:
            raise PipelineValidationError(f"{stage_label} failed workflow resolution: {e}") from e
        
        stage_input_keys = available_keys.union(set(step_params.keys()))
        
        if strict_validation:
            target_sig = _resolve_target_callable(workflow_cls)
            _check_callable_signature(target_sig, stage_input_keys, stage_label)
            
        known_categories = set(registry.keys())
        for param_key, param_val in step_params.items():
            if not isinstance(param_val, str):
                continue
            is_registered_category = param_key in known_categories
            is_dot_path = "." in param_val and not param_val.startswith(("/", "./", "../"))    
            
            if is_registered_category or is_dot_path:
                try:
                    component_callable = registry.get(param_key, param_val)
                except (KeyError, ImportError, AttributeError) as e:
                    raise PipelineValidationError(
                        f"Contract Breach in {stage_label} -> '{param_key}={param_val}': {e}"
                    ) from e
                
                if strict_validation:
                    target_sig = _resolve_target_callable(component_callable) 
                    _check_callable_signature(
                        target_sig, stage_input_keys,
                        f"{stage_label} -> {param_key}='{param_val}'"
                    )   
        
        call_target = workflow_cls.__call__ if inspect.isclass(workflow_cls) else workflow_cls
        type_hints = get_type_hints(call_target)
        return_type = type_hints.get("return")
        
        if hasattr(return_type, "__annotations__"):
            produced_keys = set(return_type.__annotations__.keys())
            available_keys.update(produced_keys)
        elif return_type is dict or return_type is None:
            strict_validation = False
            
    print(f"Dry-Run DAG Validation passed: all workflows, registries and dot-paths are aligned. ")
    return True
        
        
    
