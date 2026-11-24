import pytest
import ray
from typing import TypedDict, Callable, Dict, Any
import textwrap
from distrops.utils import get_expected_arguments, resolve_config_targets
from distrops.engine import PipelineOrchestrator
from distrops.validator import PipelineValidationError, validate_pipeline_dag
from distrops.registry import registry
import torch
import functools
from transformers import AutoModelForSequenceClassification
from distrops.runner import (resolve_search_spaces, _sample_resolved_search_spaces,
                             train_hyperparameter_search,
                             run_single_train, main
                             )
import yaml
from ray import tune

class MockPrepOutput(TypedDict):
    processed_uri: str
    input_dim: int
    num_classes: int
    
class MockTrainOutput(TypedDict):
    final_score: float
    optimizer_lr: float


@registry.register(registry_name="raw_features", name="custom_scaled_choice")
def custom_scaled_choice(base_values: list, multiplier: int = 2):
    """Custom researcher search space prior registered under the parameter's category."""
    return tune.choice([v * multiplier for v in base_values])


@registry.register(registry_name="workflows", name="untyped_dict_prep")
def untyped_dict_prep(seed_val: int = 7) -> Dict[str, Any]:
    return {"dynamic_key": seed_val * 2}


@registry.register(registry_name="workflows", name="downstream_consumer")
def downstream_consumer(dynamic_key: int) -> dict:
    return {"consumed_val": dynamic_key + 1}

@registry.register(registry_name="workflows", name="broken_return_workflow")
def broken_return_workflow():
    return "not_a_dict"

    
@registry.register(name="adam", registry_name="optimizer")
class MockAdam:
    def __call__():
        return torch.optim.Adam



@registry.register(name="mock_prep", registry_name="workflows")
class MockPrepWorkflow:
    def __call__(self, raw_uri: str, raw_features: int = 8, num_classes: int = 3, scale: int = 4) -> MockPrepOutput:
        return {"processed_uri": f"{raw_uri}_scaled_{scale}",
                "input_dim": raw_features,
                "num_classes": num_classes
                }   
    
@registry.register(name="mock_train", registry_name="workflows")
class MockTrainWorkflow:
    def __call__(self, processed_uri: str, input_dim: int,
                 num_classes: int, loss: torch.nn.Module,
                 optimizer: Callable, lr=0.01
                 ) -> MockTrainOutput:
        
        #loss_cls = torch.nn.CrossEntropyLoss
        #opt_cls = registry.get(optimizer, registry_name="optimizer")
        
        assert isinstance(loss, torch.nn.CrossEntropyLoss)
        assert isinstance(optimizer, functools.partial)
        
        model = torch.nn.Linear(input_dim, num_classes)
        opt = optimizer(model.parameters())
        assert isinstance(opt, torch.optim.SGD)
        
        x = torch.randn(4, input_dim)
        y = torch.randint(0, num_classes, (4,))
        opt.zero_grad()
        out = model(x)
        loss_val = loss(out, y)
        loss_val.backward()
        opt.step()
        
        assert processed_uri.endswith("_scaled_4")
        
        return {"final_loss": float(loss_val.item()),
                "optimizer_lr": float(opt.param_groups[0]["lr"])
                }


def test_recursive_resolve_config_targets_nested_and_partial():
    nested_cfg = {
        "criterion": {"_target_": "torch.nn.CrossEntropyLoss",
                      "label_smoothing": 0.15
                      },
        "opt_factory": {"_target_": "torch.optim.AdamW",
                        "_partial_": True,
                        "lr": 1e-4,
                        "weight_decay": 0.05
                        },
        "plain_string": "data.csv"
    }
    resolved = resolve_config_targets(nested_cfg)
    
    assert isinstance(resolved["criterion"], torch.nn.CrossEntropyLoss)
    assert resolved["criterion"].label_smoothing == 0.15
    assert isinstance(resolved["opt_factory"], functools.partial)
    assert resolved["plain_string"] == "data.csv"
    
    dummy_model = torch.nn.Linear(4, 2)
    opt = resolved["opt_factory"](dummy_model.parameters())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-4
    
    
    
def test_dot_path_component_resolution():
    ce_cls = registry.get(name_or_path="torch.nn.CrossEntropyLoss",
                           registry_name="loss"
                           )
    assert ce_cls is torch.nn.CrossEntropyLoss
    
def test_registered_component_resolution():
    opt_cls = registry.get(name_or_path="adam", registry_name="optimizer")
    assert opt_cls is not None
    
def test_dag_validation_passes_with_dot_paths_and_typed_dict():
    valid_config = {"workflows": [
                            {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
                            {"name": "mock_train",
                            "params": {
                                        "optimizer": {"_target_": "torch.optim.SGD",
                                                     "_partial_": True,
                                                     "lr": 0.05
                                                     }, 
                                       "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                                       
                                       }
                            }
                        ]
                    }
    assert validate_pipeline_dag(valid_config) is True
    
def test_dag_validation_catches_missing_args_on_dot_path_class():
    broken_dot_path_config = {
        "workflows": [
            {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
            {"name": "mock_train", "params": {"loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                                                # "optimizer": {"_target_": "torch.optim.SGD",
                                                #               #"_partial_": True,
                                                #               },
                                                "models": {"_target_": "torch.nn.Linear",
                                                           "in_features": 10,
                                                           #"model_name": "microsoft/deberta-v3-small",
                                                           }
                                              # lr intentionally omitted
                                            }
             
            }
        ]
    }    
    
    with pytest.raises(PipelineValidationError, match="Missing required arguments \['out_features'\]"):
        validate_pipeline_dag(broken_dot_path_config)


def test_dag_validation_catches_typo_in_partial_target():
    broken_partial_config = {
        "workflows": [
            {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
            {
                "name": "mock_train",
                "params": {
                    "optimizer": {
                        "_target_": "torch.optim.SGD",
                        "_partial_": True,
                        "learning_rat_typo": 0.01,  # Misspelled parameter name!
                    },
                },
            },
        ]
    }
    with pytest.raises(PipelineValidationError, match=r"Unexpected arguments \['learning_rat_typo'\]"):
        validate_pipeline_dag(broken_partial_config)
        
        
#def test_pipeline_orchestrator_nested_targets_and_partial_end_to_end():
            
def test_dynamic_plugin_file_with_unified_registry_end_to_end(tmp_path):
    plugin_file = tmp_path / "custom_research_plugin.py"
    plugin_code = textwrap.dedent("""
                                  from typing import TypedDict, Any
                                  from distrops.registry import registry
                                  
                                  @registry.register(registry_name="loss", name="custom_focal_loss")
                                  class CustomFocalLoss:
                                      def __init__(self, focal_gamma: float, focal_alpha: float = 0.25):
                                          self.gamma = focal_gamma
                                          self.alpha = focal_alpha
                                          
                                      def compute(self, base_val: float) -> float:
                                          return base_val * self.gamma * self.alpha
                                          
                                          
                                  class PluginStageOutput(TypedDict):
                                      computed_loss: float
                                  
                                  
                                  @registry.register(registry_name="workflows", name="plugin_eval_workflow")
                                  class PluginEvalWorkflow:
                                      def __call__(self, loss: Any,) -> PluginStageOutput:
                                          #loss_cls = registry.get(registry_name="loss", name_or_path=loss)
                                          #loss_fn = loss_cls(focal_gamma=focal_gamma, focal_alpha=focal_alpha)
                                          return {"computed_loss": loss.compute(10.0)}
                                     
                                    
                                
                                  """
                                  ) 
    plugin_file.write_text(plugin_code)
    ray.init(ignore_reinit_error=True, num_cpus=2,
             runtime_env={"env_vars": {"PYTHONPATH": "src"}}
             )
    try:
        invalid_config = {
            "plugins": [str(plugin_file)],
            "workflows": [
                {"name": "plugin_eval_workflow",
                 "resources": {"cpu": 1, "gpu": 0.0},
                 "params": {"loss": {"_target_":"custom_focal_loss"}},
                 }
            ]
        }
        orchestrator_fail = PipelineOrchestrator(invalid_config) 
        with pytest.raises(PipelineValidationError, match="Missing required arguments \\['focal_gamma'\\]"):
            orchestrator_fail.run()
            
        valid_config = {
            "plugins": [str(plugin_file)],
            "workflows": [
                {"name": "plugin_eval_workflow",
                 "resources": {"cpu": 1, "gpu": 0.0},
                 "params": {"loss": {"_target_": "custom_focal_loss",
                                     "focal_gamma": 2.0,
                                     "focal_alpha": 0.5
                                    },
                            },
                 }
            ]
        }
        orchestrator_pass = PipelineOrchestrator(valid_config)
        result = orchestrator_pass.run()
        assert result == {"computed_loss": 10.0}    
    finally:
        ray.shutdown()
        
        
@pytest.fixture
def mock_registry():
    return {"mock_prep": MockPrepWorkflow, "mock_train": MockTrainWorkflow}


def test_signature_get_expected_arguments():
    def target_fn(a: int, /, b: str, c: float = 1.5, *, d: bool = True):
        return (a,b,c,d)
    
    config = {"a": 10, "b": "hello",
              "d": False,
              "unused_infra_key": "/tmp/ray",
              "gpu": 0.5
              }
    args, kwargs = get_expected_arguments(target_fn, config)
    assert args == (10,)
    assert kwargs == {"b": "hello", "c": 1.5, "d": False}
    assert target_fn(*args, **kwargs) == (10, "hello", 1.5, False)


# def test_dag_validation_passes_with_valid_typed_dict_flow():
#     valid_config = {
#         "workflows": [
#             {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
#             {"name": "mock_train", "params": {"epochs": 5}},
#         ]
#     }
#     assert validate_pipeline_dag(valid_config) is True
    
    
def test_dag_validation_fails_on_missing_upstream_key():
    broken_config = {
        "workflows": [
            {"name": "mock_train", "params": {"loss": {"_target_": "torch.nn.CrossEntropyLoss",
                                                        "label_smoothing": 0.1,
                                                        },
                                            "optimizer": {"_target_": "torch.optim.SGD",
                                                            "_partial_": True,
                                                            "lr": 0.025,
                                                            "momentum": 0.9
                                                            }
                                            }
             }
        ]
    }
    
    with pytest.raises(PipelineValidationError, match="Missing required argument"):
        validate_pipeline_dag(broken_config)
        
        
def test_dag_validation_fails_on_unregistered_workflow():
    unregistered_config = {
        "workflows": [
            {"name": "non_existent_step", "params": {}},
        ]
    }
    with pytest.raises(PipelineValidationError, match="non_existent_step not found in workflows registry"):
        validate_pipeline_dag(unregistered_config)
        
        
def test_pipeline_orchestrator_nested_targets_and_partial_end_to_end_execution():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    try:
        config = {
            "workflows": [
                {
                    "name": "mock_prep",
                    "resources": {"cpu": 1, "gpu": 0.0},
                    "params": {"raw_uri": "dataset_v1", "scale": 4, 
                               "raw_features": 6, "num_classes": 4
                               },
                },
                {
                    "name": "mock_train",
                    "resources": {"cpu": 1, "gpu": 0.0},
                    "params": {"loss": {"_target_": "torch.nn.CrossEntropyLoss",
                                        "label_smoothing": 0.1,
                                        },
                               "optimizer": {"_target_": "torch.optim.SGD",
                                             "_partial_": True,
                                             "lr": 0.025,
                                             "momentum": 0.9
                                             }
                               },
                }
            ]
        }
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()
        
        assert result["optimizer_lr"] == 0.025
        assert isinstance(result["final_loss"], float)
        assert orchestrator.pipeline_state["input_dim"] == 6
    finally:
        ray.shutdown()
        




def test_resolve_search_spaces_and_dry_run_sampling():
    """
    Verifies that '_target_' + '_search_space_: true' resolves both Ray Tune primitives
    and custom registered distributions while leaving runtime '_target_' blocks untouched.
    """
    raw_config = {
        "workflows": [
            {"name": "mock_prep",
             "resources": {"cpu": 1, "gpu": 0.0},
             "params": {
                 "raw_uri": "s3://bucket/data",
                 "raw_features": {
                     "_target_": "custom_scaled_choice",
                     "_search_space_": True,
                     "base_values": [4, 8],
                     "multiplier": 2,
                 }
             }
            },
            {
                "name": "mock_train",
                "resources": {"cpu": 1, "gpu": 0.0},
                "params": {
                    "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                    "optimizer": {
                        "_target_": "torch.optim.SGD",
                        "_partial_": True,
                        "lr": {
                            "_target_": "ray.tune.loguniform",
                            "_search_space_": True,
                            "lower": 1e-3,
                            "upper": 1e-1,
                        },
                    },
                },
                    
            },
        ]
    }
    
    assert validate_pipeline_dag(raw_config) is True
    resolved_space = resolve_search_spaces(raw_config)
    prep_sampler = resolved_space["workflows"][0]["params"]["raw_features"]
    lr_sampler = resolved_space["workflows"][1]["params"]["optimizer"]["lr"]
    
    assert isinstance(prep_sampler, tune.search.sample.Categorical)
    assert prep_sampler.categories == [8, 16]
    assert isinstance(lr_sampler, tune.search.sample.Float)
    
    # Runtime _target_ blocks must remain dicts until worker execution
    assert resolved_space["workflows"][1]["params"]["loss"] == {"_target_": "torch.nn.CrossEntropyLoss"}
    
    # Sample concrete values and re-verify DAG contracts
    sampled_cfg = _sample_resolved_search_spaces(resolved_space)
    assert sampled_cfg["workflows"][0]["params"]["raw_features"] in [8, 16]
    assert 1e-3 <= sampled_cfg["workflows"][1]["params"]["optimizer"]["lr"] <= 1e-1
    assert validate_pipeline_dag(sampled_cfg) is True
    


def test_runner_train_hyperparameter_search_end_to_end(tmp_path):
    """
    Writes a temporary YAML manifest using '_target_' + '_search_space_: true'
    with both ray.tune.loguniform and a custom plugin distribution, then executes
    train_hyperparameter_search(config_path) end-to-end.
    """
    plugin_file = tmp_path / "custom_dist_plugin.py"
    plugin_file.write_text(
        textwrap.dedent("""
                        from ray import tune
                        from distrops.registry import registry
                        from typing import TypedDict, Callable
                        import functools
                        import torch
                        
                        @registry.register(registry_name="raw_features", name="plugin_feature_choice")
                        def plugin_feature_choice(options: list):
                            return tune.choice(options)
                            
                            
                        class PluginPrepOutput(TypedDict):
                            processed_uri: str
                            input_dim: int
                            num_classes: int

                        class PluginTrainOutput(TypedDict):
                            final_loss: float
                            optimizer_lr: float

                        @registry.register(registry_name="workflows", name="plugin_mock_prep")
                        class PluginMockPrep:
                            def __call__(self, raw_uri: str, raw_features: int = 8, num_classes: int = 3) -> PluginPrepOutput:
                                return {
                                    "processed_uri": f"{raw_uri}_ok",
                                    "input_dim": raw_features,
                                    "num_classes": num_classes,
                                }

                        @registry.register(registry_name="workflows", name="plugin_mock_train")
                        class PluginMockTrain:
                            def __call__(
                                self,
                                processed_uri: str,
                                input_dim: int,
                                num_classes: int,
                                loss: torch.nn.Module,
                                optimizer: Callable,
                            ) -> PluginTrainOutput:
                                model = torch.nn.Linear(input_dim, num_classes)
                                opt = optimizer(model.parameters())
                                x = torch.randn(4, input_dim)
                                y = torch.randint(0, num_classes, (4,))
                                opt.zero_grad()
                                loss_val = loss(model(x), y)
                                loss_val.backward()
                                opt.step()
                                return {
                                    "final_loss": float(loss_val.item()),
                                    "optimizer_lr": float(opt.param_groups[0]["lr"]),
                                }
                        """)
    )
    
    config_file = tmp_path / "hpo_unified_config.yml"
    yaml_payload = {
        "plugins": [str(plugin_file)],
        "infrastructure": {
            "storage_path": str(tmp_path / "runner_ray_results"),
            "experiment_name": "runner_unified_hpo_test",
            "trial_cpu": 1,
            "trial_gpu": 0.0,
        },
        "hpo_bounds": {
            "metric": "final_loss",
            "mode": "min",
            "num_samples": 2,
            "search_alg": {
                "_target_": "ray.tune.search.optuna.OptunaSearch",
                "metric": "final_loss",
                "mode": "min",
            },
            "scheduler": {
                "_target_": "ray.tune.schedulers.ASHAScheduler",
                "time_attr": "training_iteration",
                "max_t": 1,
                "grace_period": 1,
            },
        },
        "workflows": [
            {
                "name": "plugin_mock_prep",
                "resources": {"cpu": 1, "gpu": 0.0},
                "params": {
                    "raw_uri": "s3://bucket/data",
                    "raw_features": {
                        "_target_": "plugin_feature_choice",
                        "_search_space_": True,
                        "options": [4, 8],
                    },
                },
            },
            {
                "name": "plugin_mock_train",
                "resources": {"cpu": 1, "gpu": 0.0},
                "params": {
                    "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                    "optimizer": {
                        "_target_": "torch.optim.SGD",
                        "_partial_": True,
                        "lr": {
                            "_target_": "ray.tune.loguniform",
                            "_search_space_": True,
                            "lower": 0.01,
                            "upper": 0.05,
                        },
                    },
                },
            },
        ],
    }
    
    config_file.write_text(yaml.safe_dump(yaml_payload))
    
    try:
        results, best_result = train_hyperparameter_search(config_path=str(config_file))
        assert len(results) == 2
        assert "final_loss" in best_result.metrics
        assert 0.01 <= best_result.metrics["optimizer_lr"] <= 0.05
    finally:
        ray.shutdown()
        
        

def test_untyped_dict_fallback_and_stage_override(tmp_path):
    """Verifies Dict[str, Any] return hints relax strict downstream checks and stage params override upstream keys."""
    cfg = {
        "workflows": [
            {"name": "untyped_dict_prep", "resources": {"cpu": 1, "gpu": 0.0}, "params": {"seed_val": 5}},
            {"name": "downstream_consumer", "resources": {"cpu": 1, "gpu": 0.0}, "params": {"dynamic_key": 100}},
        ]
    }
    assert validate_pipeline_dag(cfg) is True
    ray.init(ignore_reinit_error=True, num_cpus=2)
    try:
        orch = PipelineOrchestrator(cfg)
        result = orch.run()
        assert result == {"consumed_val": 101}
        assert orch.pipeline_state["stage_0_untyped_dict_prep"] == {"dynamic_key": 10}
    finally:
        ray.shutdown()    
    
    

def test_cli_train_subcommand_end_to_end(tmp_path, monkeypatch):
    """Tests `distrops train --config_path ...` CLI entrypoint end-to-end."""
    config_file = tmp_path / "cli_train.yml"
    config_file.write_text(
        yaml.safe_dump({"infrastructure": {"storage_path": str(tmp_path / "cli_out")},
                        "workflows": [
                            {"name": "untyped_dict_prep"}
                        ]})
    )
    monkeypatch.setattr("sys.argv", ["distrops", "train", "--config_path", str(config_file)])
    try:
        main()
    finally:
        ray.shutdown()
        
        
def test_workflow_non_dict_return_raises_type_error():
    """Ensures workflows returning non-dict objects trigger a clear TypeError."""
    cfg = {"workflows": [{"name": "broken_return_workflow", "resources": {"cpu": 1, "gpu": 0.0}, "params": {}}]}
    ray.init(ignore_reinit_error=True, num_cpus=2)
    try:
        orch = PipelineOrchestrator(cfg)
        with pytest.raises(TypeError, match="must return a dict"):
            orch.run()
    finally:
        ray.shutdown()
    