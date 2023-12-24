import argparse
import glob
import multiprocessing
nb_cores = multiprocessing.cpu_count()
from pythainlp.tokenize import word_tokenize
from tqdm.auto import tqdm
import csv
from collections import Counter

from scipy.stats import chi2_contingency

_TOKENIZER = word_tokenize
_TOKENIZER_NAME = 'newmm'
_CORPUS_ROOT_PATHs = '../../data/raw/cleaned_data-used-in-wangchanberta/'

_COLUMN_ID_FOR_FACT_DESCRIPTION = 6

def process_text(text):
    text.strip()
    words = word_tokenize(text)
    freq = Counter(words)
    return freq

def process_corpora(fname):
    freq_combined = Counter()
    with open(fname, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            freq = process_text(row[_COLUMN_ID_FOR_FACT_DESCRIPTION])
            freq_combined += freq
    return freq_combined

def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="compute_freq_TSCC.py",
        description="compute word frequencies of corpora used in TSCC",
    )

    # required
    parser.add_argument(
        "--input_dir", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str,
    )

    
    args = parser.parse_args()
    fnames = [f'{args.input_dir}/{str(x)}' for x in glob.glob(f'*.csv', root_dir=args.input_dir)]
    print(fnames)

    with multiprocessing.Pool(nb_cores) as pool:
        results = pool.map(process_corpora, fnames)
    
    freqs = Counter()
    for result in results:
        freqs += result

    # Save frequencies to output file
    with open(f'{args.output_dir}/frequency_stats_TSCC.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Frequency'])
        for word, frequency in freqs.most_common():
            writer.writerow([word, frequency])
        print('successfully stored frequency')

if __name__ == "__main__":
    main()

