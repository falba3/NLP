from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.metrics import f1_score
import numpy as np
import torch


torch.manual_seed(0) # to replicate results
dataset = load_dataset('rotten_tomatoes')

# Customized Stopping Callback
class CustomEarlyStoppingCallback(TrainerCallback):
    """
    Stopping Callback that will stop training based on f1 metric
    default patience: 3 epochs
    default delta: 0.01 increase  in f1
    """
    def __init__(self, delta=0.01, patience=3, metric_name="f1"):
        self.delta = delta
        self.patience = patience
        self.metric_name = metric_name
        self.best_metric = None
        self.wait = 0

    def on_evaluate(self, args, state, control, **kwargs):
        current_metric = state.metrics.get(self.metric_name)
        if current_metric is None:
            return
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric - self.best_metric > self.delta:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


def main():
    """
    DOCSTRING
    """

    # Defining model checkpoint and batch size
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Preparing Labels
    label2id = {lab: i for i, lab in enumerate(dataset["train"].features["label"].names)}

    # Tokenizing our sentences
    def preprocess_function(samples):
        return tokenizer(samples["text"], padding="longest", truncation=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=2)

    # Model Setup (Configuration and Labels)
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=len(label2id), id2label={i: lab for lab, i in label2id.items()})
    config.label2id = label2id
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)


    # Training Arguments (to tune)
    args = TrainingArguments(
        "distilbert-rottentomatoes",  # name for the model (a directory is created with this name)
        evaluation_strategy="epoch",  # we evaluate and save every epoch
        report_to="none",  # don't report to tensorboard or wandb
        save_strategy="epoch",
        learning_rate=2e-5,  # learning rate to use in Adam Optimizer.
        per_device_train_batch_size=batch_size,  # size of the batch for forward pass.
        per_device_eval_batch_size=batch_size,  # "" for eval.
        num_train_epochs=5,  # total number of training epochs
        weight_decay=0.01,  # parameter for adam optimizer
        load_best_model_at_end=True, # whether to load the best performing model at the end (best checkpoint, at the best performing moment in training.)
        metric_for_best_model="f1",  # metric for choosing the best model.
        # fp16=True # for mixed precision (lower memory usage)
    )

    # Computing Metrics
    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)

        res = f1_score(labels, predictions, average="macro")
        return {"f1": res}

    # Creating the trainer to fine-tune our model over our data
    custom_early_stopping = CustomEarlyStoppingCallback(delta=0.01, patience=3, metric_name="f1")

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[custom_early_stopping]
    )

    # Training the trainer
    trainer.train()

    # Evaluating our results on the test
    results = trainer.evaluate(tokenized_dataset["test"])
    print(f"f1 score: {round(results['eval_f1'], 4)}")


if __name__ == "__main__":
    main()

