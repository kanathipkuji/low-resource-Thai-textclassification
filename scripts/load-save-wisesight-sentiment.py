import argparse
import pandas as pd
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./data/raw/wisesight-sentiment/'
    )
    args = parser.parse_args()
    dataset = load_dataset('wisesight_sentiment', split='train+validation+test')
    dataset.to_csv(f'{args.output_dir}/wisesight-sentiment.csv')
    print('Done saving file')