import logging
logging.basicConfig(level=logging.INFO)

import torch
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
from transformers import AutoTokenizer

from src.datasets import CompressorKNNDataset

import argparse

def compute_average_lengths(dataset, tokenizer):
    num_instances = len(dataset)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_chars = 0
    total_tokens = 0

    for sample in data_loader:
        # print(sample)
        text = sample['texts'][0]
        tokens = tokenizer(text, truncation=False, padding=False)
        # print(tokens)
        total_chars += len(text)
        total_tokens += len(tokens['input_ids'])

    
    return total_tokens/num_instances, total_chars/num_instances

def main():
    model_name = "airesearch/wangchanberta-base-att-spm-uncased"

    #argparser
    parser = argparse.ArgumentParser(
        prog="compute-processed-dataset-stats.py",
        description="compute the size of training, valid, test set, # classes, avg. #tokenized words, avg. #chars",
    )
    
    #required
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--valid_dir", type=str,)
    parser.add_argument("--test_dir", type=str,)
    parser.add_argument(
        "--output_dir", type=str,
    )
    parser.add_argument(
        "--dataset_name", type=str,
    )
 
    #Dataset
    parser.add_argument("--text_column_name", type=str,)
    parser.add_argument("--label_column_name", type=str,)
    parser.add_argument("--csv_sep", type=str, default=',')

    args = parser.parse_args()

    #initialize tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision='main',
        model_max_length=416
    )
    
    print(args.train_dir)

    #datasets

    train_dataset = CompressorKNNDataset(
        args.train_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    valid_dataset = CompressorKNNDataset(
        args.valid_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    test_dataset = CompressorKNNDataset(
        args.test_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    
    unique_labels = train_dataset.unique_labels
    unique_labels_valid = valid_dataset.unique_labels
    num_labels = len(unique_labels)

    print(f'# unique labels: {len(unique_labels)}')
    print(f'# unique labels eval: {len(unique_labels_valid)}')

    combined_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])

    avg_tokens, avg_chars = compute_average_lengths(combined_dataset, tokenizer)

    for i in range(10):
        print('train set', len(train_dataset), len(train_dataset[i]['texts']))

    with open(f'{args.output_dir}/stats-{args.dataset_name}.txt', 'w') as f:
        print('writing report...')
        f.write(f'#Training: {len(train_dataset)}\n')
        f.write(f'#Valid: {len(valid_dataset)}\n')
        f.write(f'#Test: {len(test_dataset)}\n')
        f.write(f'#Classes: {num_labels}\n')
        f.write(f'Average #Tokens: {avg_tokens}\n')
        f.write(f'Average #Chars: {avg_chars}\n')

  
        print('done writing')

if __name__ == "__main__":
    main()