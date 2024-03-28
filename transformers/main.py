from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch


torch.manual_seed(0) # to replicate results
dataset = load_dataset('rotten_tomatoes')


def main():
    """
    This main function will use the 'deberta-base-uncased' model checkpoint
    to train on the rotten tomatoes dataset.
    """

    # Defining model checkpoint and batch size
    model_checkpoint = "microsoft/deberta-v3-base"

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Preparing Labels
    label2id = {lab: i for i, lab in enumerate(dataset["train"].features["label"].names)}

    # Tokenizing our sentences
    def preprocess_function(samples):
        """
        This function will tokenize the dataset.
        :param samples: sentences containing text to be tokenized
        :return: succesfully tokenized samples
        """
        return tokenizer(samples["text"], padding="longest", truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=2)

    # Model Setup (Configuration and Labels)
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=len(label2id), id2label={i: lab for lab, i in label2id.items()})
    config.label2id = label2id
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)


    # Training Arguments (to tune)
    args = TrainingArguments(
        output_dir="deberta-rottentomatoes",  # name for the model (a directory is created with this name)
        evaluation_strategy="epoch",  # we evaluate and save every epoch
        report_to="none",  # don't report to tensorboard or wandb
        save_strategy="epoch",
        learning_rate=1e-5,  # learning rate to use in Adam Optimizer. # 2e-5 was a mistake but got the best score
        per_device_train_batch_size=16,  # size of the batch for forward pass.
        per_device_eval_batch_size=16,  # "" for eval.
        num_train_epochs=5,  # total number of training epochs
        weight_decay=0.01,  # parameter for adam optimizer
        load_best_model_at_end=True, # whether to load the best performing model at the end (best checkpoint, at the best performing moment in training.)
        metric_for_best_model="f1",  # metric for choosing the best model.
        # fp16=True # for mixed precision (lower memory usage)
    )

    # Function for computing macro f1 metric
    def compute_metrics(pred):
        """
        This function will compute the macro f1 score given model predictions
        :param pred: predictions computed by a model
        :return: dictionary of f1 key and corresponding score
        """
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)

        res = f1_score(labels, predictions, average="macro")
        return {"f1": res}

    # Creating the trainer to fine-tune our model over our data
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback()]
    )

    # Training the trainer
    trainer.train()

    # Evaluating our results on the test
    results = trainer.evaluate(tokenized_dataset["test"])
    print(f"f1 test score: {round(results['eval_f1'], 4)}")

    # Making predictions on test set
    def create_preds(trainer, test_set):
        """
        This function returns a pandas DataFrame of the predictions.
        :param trainer: the trainer of the fine-tuned model that will create predictions.
        :param test_set: a tensor of the set of embedding indices for reviews in the test set to predict from.
        :return: pandas DataFrame of the predictions with index column.
        """
        test_predictions = trainer.predict(test_set)
        test_pred_labels = np.argmax(test_predictions.predictions, axis=1)
        predictions = test_pred_labels.squeeze()

        pred_df = pd.DataFrame({'pred': predictions})
        index_df = pd.DataFrame(list(range(0, len(predictions))), columns=['index'])
        pred_df = pd.concat([index_df, pred_df], axis=1)
        return pred_df

    out = create_preds(trainer, tokenized_dataset["test"])
    out.to_csv('results.csv')


if __name__ == "__main__":
    main()

