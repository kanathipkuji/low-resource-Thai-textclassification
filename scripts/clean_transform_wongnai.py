import argparse
import pandas as pd
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path', 
        type=str, 
        default='./data/raw/wongnai/w_review_train.csv'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./data/processed/wongnai/'
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_path, sep=';', header=None, names=['review', 'star'])
    df.to_csv(f'{args.output_dir}/wongnai_cleaned.csv', sep=';', index=False, encoding='utf-8')