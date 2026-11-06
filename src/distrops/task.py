from .registry import registry
import torch

class DeepLearningTrainTask:
    def __init__(self, engine, *args, **kwargs):
        self.engine = engine
        
    def __call__(self, *args, **kwargs):
        return self.engine.execute_training_pipeline(*args, **kwargs)
    
    
    def execute(self):
        optimizer_func = registry.get(self.engine.config.get("optimizer_name"))
        metric_func = registry.get(self.engine.config.get("metric_name"))
        
        optimizer = optimizer_func(self.engine.model.parameters(), lr=self.engine.config.get("learning_rate", 0.001))
        epochs = self.engine.config.get("epochs", 10)
        metrics = {}
        
        for epoch in range(epochs):
            self.engine.model.train()
            train_loss = 0.0
            train_steps = 0
            
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
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    metric_func.add_batch(predictions=preds, references=batch["labels"].cpu().numpy())
                    



        