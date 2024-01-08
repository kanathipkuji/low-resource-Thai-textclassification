import logging
logging.basicConfig(level=logging.INFO)

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from transformers.data.processors.utils import InputFeatures
from transformers.integrations import NeptuneCallback

from src.datasets import FinetunerDataset
from src.metrics import compute_metrics_with_labels
from src.callbacks import CustomNeptuneCallback
# from src.finetuners import Finetuner

import neptune

#argparse
import argparse
# python train_sequence_classification_huggingface.py --model_name_or_path xlm-roberta-base \
# --num_labels 5 --train_dir data/train_th --valid_dir data/valid_th --num_train_epochs 3

def optuna_hp_space(trial):
    return {
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [16, 32, 64, 128]),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-9, 1e-7, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 1.0),
    }
def compute_objective(metrics):
    return metrics["eval_loss"], metrics["eval_f1"], metrics["eval_accuracy"]


def main():
    #argparser
    parser = argparse.ArgumentParser(
        prog="train_sequence_classification_huggingface",
        description="train sequence classification with huggingface Trainer",
    )
    
    #required
    # parser.add_argument("--model_name_or_path", type=str,)
    # parser.add_argument("--num_labels", type=int,)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--valid_dir", type=str,)
    parser.add_argument("--test_dir", type=str,)

    #Dataset
    parser.add_argument("--text_column_name", type=str,)
    parser.add_argument("--label_column_name", type=str,)

   
    #Neptune AI
    parser.add_argument('--neptune_project', type=str)
    parser.add_argument('--neptune_api_token', type=str)
    
    args = parser.parse_args()

    
    #initialize tokenizers

    tokenizer = AutoTokenizer.from_pretrained(
        "airesearch/wangchanberta-base-att-spm-uncased",
        revision='main',
        model_max_length=416
    )
    
    #datasets
    train_dataset = FinetunerDataset(
        tokenizer,
        args.train_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
    )
    eval_dataset = FinetunerDataset(
        tokenizer,
        args.valid_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
    )
    test_dataset = FinetunerDataset(
        tokenizer,
        args.test_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
    )
    
    unique_labels = train_dataset.unique_labels

    neptune_project = args.neptune_project
    neptune_api_token = args.neptune_api_token
    run = neptune.init_run(
        project=neptune_project,
        api_token=neptune_api_token
    )
    
    # custom_neptune_callback = CustomNeptuneCallback(run=run, labels=unique_labels)
    neptune_callback = NeptuneCallback(project=neptune_project, api_token=neptune_api_token, run=run)
    compute_metrics = compute_metrics_with_labels(unique_labels)


    def model_init(trial):
        # initialize models using the number of unique labels train_dataset. Since labels in train_dataset is stratified so that it includes all possible labels, 
        # there should be no problem.
        return AutoModelForSequenceClassification.from_pretrained(
            "airesearch/wangchanberta-base-att-spm-uncased",
            num_labels=len(unique_labels)
        )

    training_args = TrainingArguments("test_trainer"),
    
    #initiate trainer
    trainer = Trainer(
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[neptune_callback, custom_neptune_callback]
    )

    # No finetuning

    #evaluate
    res = trainer.evaluate()
    print(res)

    run['eval'] = str(res)

    run.stop()

    
if __name__ == "__main__":
    main()