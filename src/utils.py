from collections import Counter
import csv
from tqdm.auto import tqdm
import pythainlp
from pythainlp.tokenize import word_tokenize
import mmap

stopwords = pythainlp.corpus.common.thai_stopwords()

import pandas as pd

CORPUS_METADATA_df = pd.DataFrame.from_dict([
    {'name': 'tscc', 'is_csv': True, 'text_column_id': 6, 'delimiter': ','},
    {'name': 'wongnai', 'is_csv': True, 'text_column_id': 0, 'delimiter': ';'},
    {'name': 'prachathai', 'is_csv': True, 'text_column_id': 3, 'delimiter': ','},
    {'name': 'thaipbs', 'is_csv': True, 'text_column_id': 1, 'delimiter': ','},
    {'name': 'wangchanberta', 'is_csv': False},
])

def process_text(text, remove_stop_words):
    text.strip()
    words = word_tokenize(text)
    if remove_stop_words:
        words = [word for word in words if word not in stopwords]
    freq = Counter(words)
    return freq

def count_csv_lines(fname, delimiter):
    df = pd.read_csv(fname)
    return len(df)
    
import pandas as pd

def process_corpora(fname, is_csv, text_column_id, delimiter, remove_stop_words):
    # Get text from text column if the file is in csv format
    
    if is_csv:
        num_lines = count_csv_lines(fname, delimiter)
        freq_combined = Counter()
        with open(fname, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader)
            for row in tqdm(reader, total=num_lines):
                freq = process_text(row[text_column_id], remove_stop_words)
                freq_combined += freq
        return freq_combined
    else:
        line_count = 0
        freq_combined = Counter()
        with open(fname, 'r') as f:
            for line in tqdm(f):
                freq = process_text(line)
                freq_combined += freq
                line_count += 1
        return freq_combined
