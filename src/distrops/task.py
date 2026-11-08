from .registry import registry
import torch
import ray

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


