import argparse
import glob
import multiprocessing
nb_cores = multiprocessing.cpu_count()
from pythainlp.tokenize import word_tokenize
from tqdm.auto import tqdm
import csv
from collections import Counter
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

_TOKENIZER = word_tokenize
_TOKENIZER_NAME = 'newmm'
_CORPUS_ROOT_PATHs = '../../data/raw/cleaned_data-used-in-wangchanberta/'
_FREQ_DIR_1 = '../../data/frequency_stats.csv'
_FREQ_DIR_2 = '../../data/frequency_stats_TSCC.csv'

def test_ll():
    # obs = np.array([[10, 10], [10, 20], [40, 50]])
    obs = np.array([[35, 10], [15, 30], [50, 60]])
    res = chi2_contingency(obs, correction=False, lambda_='log-likelihood')
    sum = 0
    N = obs.sum(axis=0)
    print('N: ', N)
    NN = N.sum()
    
    for i in range(obs.shape[0]):
        O1 = obs[i][0]
        O2 = obs[i][1]
        O = O1 + O2
        E1 = N[0] * O / NN
        E2 = N[1] * O / NN
        LL = 2 * ((O1 * np.log(O1/E1)) + (O2 * np.log(O2/E2)))
        sum += LL
        print(E1, E2)
    print('own: ', sum)
    print('res: ', res.statistic)
    print('res, expected frequency: ', res.expected_freq)
    print('res, dof: ', res.dof)

def compute_log_likelihood_and_chi2(dict1, dict2):
    '''
    First compute expected value E_i of words appeared in both dict1 and dict2. 
    For each such word, calculate:
    E_i = N_i * sum(O_i) / sum(N_i)
    where O_i is the observed value (frequency) of the current word on corpus i (dict<i>),
    and N_i is the total number of appearance of all words on that corpus.
    then, compute LL as follows:
    LL = 2 * sum(O_i * ln(O_i / E_i))
    '''

    dict1_set = set(dict1)
    dict2_set = set(dict2)
    inter = dict1_set.intersection(dict2_set)
    dif1 = dict1_set.difference(dict2_set)
    dif2 = dict2_set.difference(dict1_set)


    print('# common words of dict1, dict2:\n', len(inter))
    print('# common words of dict1, dict2:\n', len(inter))
    print('# words exclusive on dict1:\n', len(dif1))
    print('# words exclusive on dict2:\n', len(dif2))

    res = {}
    chi2 = 0

    contingency_table = np.array([[dict1[word], dict2[word]] for word in inter])
    Ns = contingency_table.sum(axis=0)
    N1 = Ns[0]
    N2 = Ns[1]
    N  = Ns.sum()
    # print(contingency_table.shape)
    # contingency_table = np.r_[contingency_table, np.array([[dict1[word], 0] for word in dif1])]
    # print(contingency_table.shape)
    # contingency_table = np.r_[contingency_table, np.array([[0, dict2[word]] for word in dif2])]
    # print(contingency_table.shape)
    chi2 = chi2_contingency(contingency_table)
    log_likelihood = chi2_contingency(contingency_table, correction=False, lambda_='log-likelihood')
    print('chi2_dof: ', chi2.dof)
    
    for word in inter:
        O1 = dict1[word]
        O2 = dict2[word]
        O = O1 + O2
        E1 = N1 * O / N
        E2 = N2 * O / N
        LL = 2 * ((O1 * np.log(O1/E1)) + (O2 * np.log(O2/E2)))
        res[word] = LL
    
    res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
    print('Descending order:')
    for i, (word, LL) in enumerate(res.items()):
        if i >= 15: break
        print(word, LL, dict1[word], dict2[word])

    print('Ascending order:')
    res2 = dict(sorted(res.items(), key=lambda x: x[1], reverse=False))
    for i, (word, LL) in enumerate(res2.items()):
        if i >= 15: break
        print(word, LL, dict1[word], dict2[word])

    return (res, chi2, log_likelihood)

def compute_stats(fname1, fname2):
    df1 = pd.read_csv(fname1)
    df2 = pd.read_csv(fname2)

    dict1 = df1.set_index('Word')['Frequency'].to_dict()
    dict2 = df2.set_index('Word')['Frequency'].to_dict()
    ex_words = ['<', '>', '</', '\n', ' ', 'discr']

    dict1_filtered = {w: f for w, f in dict1.items() if f > 5 and w not in ex_words}
    dict2_filtered = {w: f for w, f in dict2.items() if f > 5 and w not in ex_words}
    print('len of dict1, dict2:\n', len(dict1), len(dict2))
    print('len of filtered dict1, dict2:\n', len(dict1_filtered), len(dict2_filtered))

    dict1_filtered_sorted = dict(sorted(dict1_filtered.items(), key=lambda x: x[1], reverse=True))
    dict2_filtered_sorted = dict(sorted(dict2_filtered.items(), key=lambda x: x[1], reverse=True))
    print('max of dict1: ', list(dict1_filtered_sorted.items())[:30])
    print('max of dict2: ', list(dict2_filtered_sorted.items())[:30])

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=120, figsize=(18, 10))
    ax1.hist(dict1_filtered.values(), bins=300, log=True)
    ax1.set_title('Histogram of Frequency of Words in Corpora Used by WangChanBERTa')
    ax2.hist(dict2_filtered.values(), bins=300, log=True)
    ax2.set_title('Histogram of Frequency of Words in TSCC')


    res = compute_log_likelihood_and_chi2(dict1_filtered, dict2_filtered)

    return res

def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="compute_stats.py",
        description="compute stats of corpora used in WangChanBERTa and TSCC dataset",
    )

    # required
    parser.add_argument(
        "--freq1_dir", type=str, default=_FREQ_DIR_1
    )
    parser.add_argument(
        "--freq2_dir", type=str, default=_FREQ_DIR_2
    )
    parser.add_argument(
        "--output_dir", type=str
    )

    args = parser.parse_args()
    
    (LL, chi2, ll) = compute_stats(args.freq1_dir, args.freq2_dir)

    plt.savefig(f'{args.output_dir}/plots.png')
    # plt.show()
    # plt.close()

    # test_ll()


    print('chi2: ', chi2.statistic)
    print('chi2 (p value): ', chi2.pvalue)
    print('Log likelihood: ', ll.statistic)
    print('Log likelihood (p value): ', ll.pvalue)
    print('Log likelihood self created: ', np.sum(list(LL.values())))

    with open(f'{args.output_dir}/final_stats23.txt', 'w') as f:
        print('writing results...')
        f.write(str(LL))
        print('done writing')

if __name__ == "__main__":
    main()

