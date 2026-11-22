import inspect
from typing import get_type_hints, Any
from .registry import registry
 
class PipelineValidationError(Exception):
    pass

FRAMEWORK_INJECTED_PARAMS = {"self", "config", "params", 
                             "model_parameters"}


def _check_callable_signature(callable_obj: Any, 
                              available_keys: set,
                              stage_label: str,
                              ignore_params: set = None,
                              is_partial: bool = False
                              ) -> None:
    """Inspects a callable's required parameters against available configuration

    Args:
        callable_obj (Any): _description_
        available_keys (set): _description_
        stage_label (str): _description_
        ignore_params (set, optional): _description_. Defaults to None.
    """
    ignore_params = ignore_params or FRAMEWORK_INJECTED_PARAMS
    target_func = _resolve_target_callable(callable_obj)
    sig = inspect.signature(target_func)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    if is_partial:
        if not has_var_keyword:
            invalid_keys = [k for k in available_keys if k not in sig.parameters]
            if invalid_keys:
                raise PipelineValidationError(
                    f"Contract Breach in {stage_label}: Unexpected arguments {invalid_keys} for partial target."
                )

    missing_params = []
    for param_name, param in sig.parameters.items():
        if param_name in ignore_params:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        
        # if is_partial and param.kind in (
        #                                     inspect.Parameter.POSITIONAL_ONLY,
        #                                     inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #                                 ) and param.default is inspect.Parameter.empty:
        #     continue
        
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

def _validate_nested_targets(node: Any, registry_name: str, stage_label) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k != "_target_":
                _validate_nested_targets(node=v, registry_name=k, stage_label=stage_label)
                
        if "_target_" in node:
            target_name = node["_target_"]
            is_partial = bool(node.get("_partial_", False))
            try:
                callable_obj = registry.get(name_or_path=target_name, registry_name=registry_name)
            except (KeyError, ImportError, AttributeError) as e:
                raise PipelineValidationError(
                    f"Contract Breach in {stage_label} -> '{registry_name}._target_={target_name}': {e}"
                ) from e
                
            provided_keys = {k for k in node.keys() if k not in ("_target_", "_partial_")}
            _check_callable_signature(callable_obj=callable_obj, available_keys=provided_keys,
                                      stage_label=f"{stage_label} -> {registry_name}('{target_name}')",
                                      is_partial=is_partial
                                      )
        
    elif isinstance(node, list):
        for item in node:
            _validate_nested_targets(item, registry_name=registry_name, stage_label=stage_label)        
    
        
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
        
        _validate_nested_targets(step_params, registry_name="components", stage_label=stage_label)
        
        call_target = workflow_cls.__call__ if inspect.isclass(workflow_cls) else workflow_cls
        if strict_validation:
            #target_sig = _resolve_target_callable(workflow_cls)
            _check_callable_signature(call_target, stage_input_keys, stage_label)
            
        type_hints = get_type_hints(call_target)
        return_type = type_hints.get("return")
        
        if hasattr(return_type, "__annotations__"):
            produced_keys = set(return_type.__annotations__.keys())
            available_keys.update(produced_keys)
        elif return_type is dict or return_type is None:
            strict_validation = False
            
    print(f"Dry-Run DAG Validation passed: all workflows, registries and dot-paths are aligned. ")
    return True
        
        
    
