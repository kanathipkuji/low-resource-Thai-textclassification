import logging
logging.basicConfig(level=logging.INFO)

from transformers import (
    AutoTokenizer,
)
from sklearn.dummy import DummyClassifier

from src.datasets import FinetunerDataset
from src.metrics import compute_metrics

#argparse
import argparse

def main():
    #argparser
    parser = argparse.ArgumentParser(
        prog="train_sequence_classification_dummy_classifier",
        description="train sequence classification with dummy classifier",
    )
    
    #required
    # parser.add_argument("--model_name_or_path", type=str,)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--eval_dir", type=str,)

    #checkpoint
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--overwrite_output_dir', default=True, type=lambda x: (str(x).lower() in ['true','True','T']))

    #train hyperparameters
    parser.add_argument("--strategy", type=str, default='most_frequent')
  
    
    args = parser.parse_args()

    
    #initialize models and tokenizers

    tokenizer = AutoTokenizer.from_pretrained(
        "airesearch/wangchanberta-base-att-spm-uncased",
        revision='main',
        model_max_length=416
    )
    dummy = DummyClassifier(strategy=args.strategy)

    
    
    #datasets
    train_dataset = FinetunerDataset(
        tokenizer,
        args.train_dir,
        text_column_name='filtered_fact',
        label_column_name='label',
    )
    eval_dataset = FinetunerDataset(
        tokenizer,
        args.eval_dir,
        text_column_name='filtered_fact',
        label_column_name='label',
    )

    dummy.fit(train_dataset.input_ids, train_dataset.labels)
    score = dummy.score(eval_dataset.input_ids, eval_dataset.labels)
    print('score: ', score)

if __name__ == "__main__":
    main()