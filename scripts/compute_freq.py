import argparse
import glob
import multiprocessing
nb_cores = multiprocessing.cpu_count()
from pythainlp.tokenize import word_tokenize
from tqdm.auto import tqdm
import csv
from collections import Counter

_TOKENIZER = word_tokenize
_TOKENIZER_NAME = 'newmm'
_CORPUS_ROOT_PATHs = '../corpora/cleaned_data-used-in-wangchanberta/'

def process_text(text):
    text.strip()
    words = word_tokenize(text)
    freq = Counter(words)
    return freq

def process_corpora(fname):
    line_count = 0
    freq_combined = Counter()
    with open(fname, 'r') as f:
        for line in tqdm(f):
            freq = process_text(line)
            freq_combined += freq
            line_count += 1
    return freq_combined

def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="compute_stats.py",
        description="compute stats of corpora used in WangChanBERTa and TSCC dataset",
    )

    # required
    parser.add_argument(
        "--input_dir", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str,
    )
    
    args = parser.parse_args()
    fnames = [f'{args.input_dir}/{str(x)}' for x in glob.glob(f'**/*.txt', root_dir=args.input_dir, recursive=True)]

    with multiprocessing.Pool(nb_cores) as pool:
        results = pool.map(process_corpora, fnames)
    
    freqs = Counter()
    for result in results:
        freqs += result

    # Save frequencies to output file
    with open(f'{args.output_dir}/frequency_stats.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Frequency'])
        for word, frequency in freqs.most_common():
            writer.writerow([word, frequency])
        print('successfully stored frequency')

if __name__ == "__main__":
    main()

