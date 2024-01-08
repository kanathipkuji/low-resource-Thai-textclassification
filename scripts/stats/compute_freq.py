import argparse
import glob
import multiprocessing
from tqdm.auto import tqdm
import csv
from collections import Counter
import os

nb_cores = multiprocessing.cpu_count()

import functools
from src.utils import process_corpora, CORPUS_METADATA_df

def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="compute_freq.py",
        description="compute word frequencies of a given corpus",
    )

    # required
    parser.add_argument(
        "--input_dir", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str,
    )
    parser.add_argument(
        "--corpus_name", type=str,
    )
    parser.add_argument(
        '--remove_stop_words', 
        default=True, 
        type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y'])
    )
    parser.add_argument(
        '--delimiter', 
        default=',', 
        type=str,
    )
    parser.add_argument(
        '--is_csv', 
        default=True, 
        type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y'])
    )
    parser.add_argument(
        '--text_column_id', 
        default=-1, 
        type=int,
    )
    
    args = parser.parse_args()
    if args.is_csv:
        fnames = [f'{args.input_dir}/{str(x)}' for x in glob.glob(f'*.csv', root_dir=args.input_dir)]
    else:
        fnames = [f'{args.input_dir}/{str(x)}' for x in glob.glob(f'**/*.txt', root_dir=args.input_dir, recursive=True)]
    fnames = [os.path.abspath(relpath) for relpath in fnames]

    corpus_name = args.corpus_name
    is_csv = args.is_csv
    text_column_id = args.text_column_id
    delimiter = args.delimiter
    remove_stop_words = args.remove_stop_words

    df = CORPUS_METADATA_df.loc[CORPUS_METADATA_df['name'] == corpus_name]

    if not df.empty:
        print(f'Retrieving metadata from corpus {corpus_name}...')
        df = CORPUS_METADATA_df.loc[CORPUS_METADATA_df['name'] == corpus_name].iloc[0]
        is_csv = df['is_csv']
        text_column_id = df['text_column_id'].astype(int)
        delimiter = df['delimiter']
    else:
        print(f'Unknown corpus name. Using the input or default corpus metadata...')
    print(f'Retrieving files: {fnames}...')

    print(is_csv, type(is_csv))

    with multiprocessing.Pool(nb_cores) as pool:
        results = pool.map(functools.partial(process_corpora, 
                                        is_csv=is_csv,
                                        text_column_id=text_column_id,
                                        delimiter=delimiter,
                                        remove_stop_words=remove_stop_words), fnames)
    
    freqs = Counter()
    for result in results:
        freqs += result

    # Save frequencies to output file
    with open(f'{args.output_dir}/frequency_stats_{corpus_name}.csv', 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(['Word', 'Frequency'])
        for word, frequency in freqs.most_common():
            writer.writerow([word, frequency])
        print('Successfully stored frequency')

if __name__ == "__main__":
    main()

