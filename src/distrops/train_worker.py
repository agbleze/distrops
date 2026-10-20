from .data_loader import get_data
import torch
import evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import ray

def train_model(config):
    model_name = config.get("model_name")
    metric_name = config.get("metric_name", "accuracy")
    epochs = config.get("epochs")
    num_labels = config.get("num_labels")
    learning_rate = config.get("learning_rate")
    optimizer_name = config.get("optimizer")
    weight_decay = config.get("weight_decay")
    lr_scheduler_type = config.get("lr_scheduler_type", "linear")
    
    train_data_iterable, test_data_iterable, max_steps_per_epoch = get_data(config=config)
    
    metric = evaluate.load(metric_name)

    print("Getting Model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                use_safetensors=True,
                                                                num_labels=num_labels
                                                                )
    if optimizer_name == "Adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(model.parameters(), lr=learning_rate, 
                              weight_decay=weight_decay,
                              scale_parameter=False, relative_step=False
                              )
    else:
        optimizer = getattr(torch.optim, optimizer_name)
        optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    max_step = max_steps_per_epoch * epochs
    warmup_steps = int(max_step * 0.1)
    
    print(f"Using lr_scheduler_type: {lr_scheduler_type}, optimizer: {optimizer_name}, learning_rate: {learning_rate}, weight_decay: {weight_decay}")
    lr_scheduler = get_scheduler(name=lr_scheduler_type,
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=max_step
                                 )
    
    avg_eval_loss = float('inf')
    accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in train_data_iterable:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            train_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            train_ref = batch["labels"].cpu().numpy()
            metric.add_batch(predictions=train_preds, references=train_ref)
            
            loss.backward()
            optimizer.step()
            
            lr_scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / train_steps if train_loss > 0 else 0.0
        train_acc = metric.compute()["accuracy"]
        print(f"Epoch {epoch + 1} train loss: {avg_train_loss}")
        print(f"Epoch {epoch + 1} train accuracy: {train_acc}")

        model.eval()
        eval_loss = 0.0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in test_data_iterable:
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                references = batch["labels"].cpu().numpy()
                
                metric.add_batch(predictions=preds, references=references)
                eval_steps += 1

        avg_eval_loss = eval_loss / eval_steps if eval_loss > 0 else 0.0
        accuracy = metric.compute()["accuracy"]

        metric_result = {"eval_loss": avg_eval_loss,
                        "train_loss": avg_train_loss,
                        "eval_accuracy": accuracy,
                        "train_accuracy": train_acc,
                        "epoch": epoch
                        }
        print(f"Epoch {epoch + 1} evaluation loss: {avg_eval_loss}")
        print(f"Epoch {epoch + 1} evaluation accuracy: {accuracy}")
        
        ray.tune.report(metric_result)
    print(f"Training completed for {epochs} epochs. Final evaluation loss: {avg_eval_loss}, accuracy: {accuracy}")    
