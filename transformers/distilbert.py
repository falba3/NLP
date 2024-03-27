from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.metrics import f1_score
import numpy as np
import torch
from itertools import product #


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
    This main function will use the 'distilbert-base-uncased' model checkpoint
    to train on the rotten tomatoes dataset.
    """

    # Defining model checkpoint and batch size
    model_checkpoint = "distilbert-base-uncased"

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

    param_grid = {
        "output_dir": ["distilbert-rottentomatoes"],
        "report_to": ["none"],
        "evaluation_strategy": ["epoch"],
        "save_strategy": ["epoch"],
        "learning_rate": [1e-5, 2e-5],
        "per_device_train_batch_size": [16, 32],
        "per_device_eval_batch_size": [16, 32],
        "num_train_epochs": [10],
        "weight_decay": [0.01, 0.001],
        "fp16": [True, False],
        "load_best_model_at_end": [True],
        "metric_for_best_model": ["f1"]
    }


    # Function for computing macro f1 metric
    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)

        res = f1_score(labels, predictions, average="macro")
        return {"f1": res}

    # Instantiating the custom early stopping callback
    custom_early_stopping = CustomEarlyStoppingCallback(delta=0.01, patience=3, metric_name="f1")

    def tuner(parameter_grid):
        best_score = 0
        best_params = None
        best_trainer = None

        # Generate all combinations of parameter values
        param_combinations = product(*parameter_grid.values())

        for params in param_combinations:
            # Initialize a new Trainer object with the given parameters
            print(f"TRAINING PARAMETERS: {params}")
            param_dict = {key: value for key, value in zip(parameter_grid.keys(), params)} #
            args = TrainingArguments(**param_dict) #
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[custom_early_stopping]
            )

            # Train the model
            trainer.train()

            # Evaluate the model on validation set
            results = trainer.evaluate()

            # Check if the current combination is the best
            if results['eval_f1'] > best_score:
                best_score = results['eval_f1']
                best_params = params
                best_trainer = trainer

        # # Set the best parameters to the trainer
        # for param_name, param_value in zip(parameter_grid.keys(), best_params):
        #     setattr(trainer.args, param_name, param_value)

        # Evaluate the model on the test set using the best parameters
        test_results = best_trainer.evaluate(tokenized_dataset["test"])
        print(f"f1 test score: {round(test_results, 4)}\nbest parameters: {best_params}")

        # return best_params, best_score, test_results

    # results = trainer.evaluate(tokenized_dataset["test"])
    # print(f"f1 score: {round(results['eval_f1'], 4)}")

    tuner(parameter_grid=param_grid)


if __name__ == "__main__":
    main()

