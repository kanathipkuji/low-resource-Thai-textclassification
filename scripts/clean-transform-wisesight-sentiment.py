import argparse
import pandas as pd
import re
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path', 
        type=str, 
        default='./data/raw/wisesight-sentiment/wisesight-sentiment.csv'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./data/processed/wisesight-sentiment/'
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_path, sep=',')

    if os.path.exists(f'{args.output_dir}/') == False:
            os.makedirs(f'{args.output_dir}/', exist_ok=True)
    df.to_csv(f'{args.output_dir}/wisesight-sentiment-cleaned.csv', sep=',', index=False, encoding='utf-8')