from .registry import registry
import torch
import ray
from typing import Any, Callable, TypedDict, Dict, Optional
import os
import math
import ray.data
import torch.nn as nn
import pyarrow as pa


class DeepLearningTrainTask:
    def __init__(self, engine, *args, **kwargs):
        self.engine = engine
        
    def __call__(self, *args, **kwargs):
        return self.execute()
    
    def execute(self):
        optimizer_func = registry.get(self.engine.config.get("optimizer_name"))
        metric_func = registry.get(self.engine.config.get("metric_name"))
        
        optimizer = optimizer_func(self.engine.model.parameters(), lr=self.engine.config.get("learning_rate", 0.001))
        epochs = self.engine.config.get("epochs", 10)
        metrics = {}
        
        train_loss = float("inf")
        train_steps = 0
        avg_eval_loss = float("inf")
        eval_acc = 0.0
        metric_result = {}
        
        for epoch in range(epochs):
            self.engine.model.train()
            
            
            for batch in self.engine.train_data_iterable:
                optimizer.zero_grad()
                outputs = self.engine.model(**batch)
                loss = outputs.loss
                
                train_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                train_ref = batch["labels"].cpu().numpy()
                metric_func.add_batch(predictions=train_preds, references=train_ref)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
            train_acc = metric_func.compute()["accuracy"]
            metrics[f"epoch_{epoch + 1}"] = {"train_loss": avg_train_loss, "train_accuracy": train_acc}
            
            self.engine.model.eval()
            with torch.no_grad():
                eval_loss = 0.0
                eval_steps = 0
                
                for batch in self.engine.test_data_iterable:
                    outputs = self.engine.model(**batch)
                    eval_loss += outputs.loss.item()
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    references = batch["labels"].cpu().numpy()
                    metric_func.add_batch(predictions=preds, references=references)
                    eval_steps += 1
                
                avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else 0.0
                eval_acc = metric_func.compute()["accuracy"]
                
            metric_result = {"eval_loss": avg_eval_loss,
                             "train_loss": avg_train_loss,
                             "eval_accuracy": eval_acc,
                             "train_accuracy": train_acc,
                             "epoch": epoch + 1
                             }
            print(f"Epoch {epoch + 1} train loss: {avg_train_loss}, train accuracy: {train_acc}")
            print(f"Epoch {epoch + 1} eval loss: {avg_eval_loss}, eval accuracy: {eval_acc}")
            ray.tune.report(metric_result)
             
        print(f"Training completed for {epochs}. Final evaluation loss: {avg_eval_loss}, final evaluation accuracy: {eval_acc}")
        return metric_result            


class GradientDescentOutput(TypedDict):
    checkpoint_point: str
    eval_loss: float
    eval_accuracy: float
    
    
@registry.register(name="gradient_descent", registry_name="tasks")
class GradientDescentTask:
    def __call__(self, 
             train_data_path: str,
             test_data_path: str,
             model: nn.Module,
             optimizer: Callable,
             loss: Optional[nn.Module] = None,
             scheduler: Optional[Callable] = None,
             checkpoint_dir: str = "./checkpoints",
             epochs: int = 3,
             batch_size: int = 16,
             drop_last: bool = False,
             local_shuffle_buffer_size=1000,
             metrics_queue: Optional[Any] = None,
             *args, **kwargs
             ) -> GradientDescentOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"metrics_queue: {metrics_queue}")
        opt = optimizer(model.parameters())
        lr_scheduler = scheduler(opt) if scheduler is not None else None
        
        train_ds = ray.data.read_parquet(train_data_path)
        test_ds = ray.data.read_parquet(test_data_path)
        
        total_samples = train_ds.count()
        steps_per_epoch = (total_samples // batch_size
                            if drop_last else math.ceil(total_samples/batch_size)
                        )
        max_steps = steps_per_epoch * epochs
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_eval_loss = float("inf")
        best_eval_acc = 0.0
        best_checkpoint = ""
        
        for epoch in range(epochs):
            model.train()
            train_stream = train_ds.iter_torch_batches(batch_size=batch_size, drop_last=drop_last, 
                                                       local_shuffle_buffer_size=local_shuffle_buffer_size
                                                       )
            
            for batch in train_stream:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()
                         }
                opt.zero_grad()
                
                if "input_ids" in batch:
                    outputs = model(**batch)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                else:
                    inputs = batch.get("inputs") or batch.get("x")
                    outputs = model(inputs)
                    logits = output.logits if hasattr(outputs, "logits") else outputs    
                
                if loss is not None:
                    targets = batch.get("labels") if "labels" in batch else batch.get("y")
                    step_loss = loss(logits, targets)
                elif hasattr(outputs, "loss") and outputs.loss is not None:
                    step_loss = outputs.loss
                else:
                    raise ValueError("Model did not return '.loss' and no custom 'loss' _target_ was provided.")
                
                step_loss.backward()
                opt.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                    
            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total_eval = 0
            
            test_stream = test_ds.iter_torch_batches(batch_size=batch_size)
            with torch.no_grad():
                for batch in test_stream:
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k,v in batch.items()
                    }
                    targets = batch.get("labels") if "labels" in batch else batch.get("y")
                    
                    if "input_ids" in batch:
                        outputs = model(**batch)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    else:
                        inputs = batch.get("inputs") or batch.get("x")
                        outputs = model(inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        
                    if loss is not None:
                        batch_loss = loss(logits, targets)
                    else:
                        batch_loss = outputs.loss
                        
                    batch_n = targets.size(0)
                    val_loss_sum += float(batch_loss.item()) * batch_n
                    preds = torch.argmax(logits, dim=-1)
                    correct += int((preds == targets).sum().item())
                    total_eval += batch_n
                    
            avg_eval_loss = val_loss_sum / max(total_eval, 1)
            eval_accuracy = correct / max(total_eval, 1)
            print(f"Epoch {epoch + 1}/{epochs} | Eval Loss: {avg_eval_loss:.4f} | Eval Acc: {eval_accuracy:.4f}",
                flush=True
                )
            
            if metrics_queue is not None:
                met = {"training_iteration": epoch +1,
                    "eval_loss": avg_eval_loss,
                    "eval_accuracy": eval_accuracy
                    }
                metrics_queue.put(met)
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_eval_acc = eval_accuracy
                best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), best_checkpoint)
        return {"checkpoint_point": best_checkpoint,
                "eval_loss": best_eval_loss,
                "eval_accuracy": best_eval_acc
                }






class TabularTrainTask:
    def __init__(self, engine, *args, **kwargs):
        self.engine = engine
        
    def __call__(self, *args, **kwargs):
        self.execute() 
        
    def execute(self):
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        self.engine.model.fit(X_train, y_train)
        eval_score = self.engine.model.score(X_test, y_test)
        train_score = self.engine.model.score(X_train, y_train)
        metrics = {"eval_loss": eval_score, "train_loss": train_score}
        
        ray.tune.report(metrics)
    