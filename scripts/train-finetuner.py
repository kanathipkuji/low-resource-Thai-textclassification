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
from transformers import DataCollatorWithPadding

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
    parser.add_argument("--num_train_epochs", type=int,)

    #Dataset
    parser.add_argument("--text_column_name", type=str,)
    parser.add_argument("--label_column_name", type=str,)
    parser.add_argument("--csv_sep", type=str, default=',')

    #Optuna
    parser.add_argument('--optuna', default=False, type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']))
    parser.add_argument("--num_trials", type=int, default=10)

    #checkpoint
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--overwrite_output_dir', default=True, type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']))
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    
    #logs
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--logging_steps", type=int, default=200)
    
    #eval
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--evaluation_strategy", type=str, default='epoch')
    parser.add_argument("--save_strategy", type=str, default='epoch')
    parser.add_argument("--metric_for_best_model", type=str, default='f1')
    
    #train hyperparameters
    # parser.add_argument("--train_max_length", type=int, default=128)
    # parser.add_argument("--eval_max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument('--dataloader_drop_last', default=False, type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']))
    
    #Neptune AI
    parser.add_argument('--neptune_project', type=str)
    parser.add_argument('--neptune_api_token', type=str)
    
    #others
    parser.add_argument("--seed", type=int, default=1412)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']))
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    
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
        sep=args.csv_sep,
    )
    eval_dataset = FinetunerDataset(
        tokenizer,
        args.valid_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    test_dataset = FinetunerDataset(
        tokenizer,
        args.test_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    
    unique_labels = train_dataset.unique_labels
    unique_labels_eval = eval_dataset.unique_labels

    print(f'# unique labels: {len(unique_labels)}')
    print(f'# unique labels eval: {len(unique_labels_eval)}')
    print('train set', len(train_dataset), train_dataset[0]['input_ids'].shape)
    print('valid set', len(eval_dataset), eval_dataset[0]['input_ids'].shape)

    neptune_project = args.neptune_project
    neptune_api_token = args.neptune_api_token
    run = neptune.init_run(
        project=neptune_project,
        api_token=neptune_api_token
    )
    run['bash'].upload('./scripts/train-finetuner.sh')
    
    custom_neptune_callback = CustomNeptuneCallback(run=run, labels=unique_labels)
    neptune_callback = NeptuneCallback(project=neptune_project, api_token=neptune_api_token, run=run)
    compute_metrics = compute_metrics_with_labels(unique_labels)

    #training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        #checkpoint
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        #logs
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        #eval
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        # load_best_model_at_end=args.load_best_model_at_end,
        load_best_model_at_end=True,
        #others
        seed=args.seed,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        dataloader_drop_last=args.dataloader_drop_last,
        report_to='none',
    )

    def model_init(trial):
        # initialize models using the number of unique labels train_dataset. Since labels in train_dataset is stratified so that it includes all possible labels, 
        # there should be no problem.
        return AutoModelForSequenceClassification.from_pretrained(
            "airesearch/wangchanberta-base-att-spm-uncased",
            num_labels=len(unique_labels),
            revision='main'
        )

    # Hyperparameter tuning
    if args.optuna:
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        best_trial = trainer.hyperparameter_search(
                        direction=["minimize", "maximize", "maximize"],
                        backend="optuna",
                        hp_space=optuna_hp_space,
                        n_trials=args.num_trials,
                        compute_objective=compute_objective,    
                    )
        print(best_trial)
        run['tuning/best_trial'] = str(best_trial)

    #initiate trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[neptune_callback, custom_neptune_callback]
    )
    
    #train
    trainer.train()
    
    #evaluate
    trainer.evaluate()

    run.stop()

    
if __name__ == "__main__":
    main()