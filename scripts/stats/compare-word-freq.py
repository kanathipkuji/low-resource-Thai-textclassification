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
from scipy.stats import chi2_contingency, spearmanr
import os

_TOKENIZER = word_tokenize
_TOKENIZER_NAME = 'newmm'

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

def compute_stats(dict1, dict2):
    '''
    First compute expected value E_i of words appeared in both dict1 and dict2. 
    For each such word, calculate:
    E_i = N_i * sum(O_i) / sum(N_i)
    where O_i is the observed value (frequency) of the current word on corpus i (dict<i>),
    and N_i is the total number of appearance of all words on that corpus.
    then, compute LL as follows:
    LL = 2 * sum(O_i * ln(O_i / E_i))

    Then, compute Spearman's rank correlation coefficient.
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

    LL_dict = {}
    chi2 = 0

    contingency_table = np.array([[dict1[word], dict2[word]] for word in inter])
    Ns = contingency_table.sum(axis=0)
    N1 = Ns[0]
    N2 = Ns[1]
    N  = Ns.sum()

    chi2 = chi2_contingency(contingency_table, correction=False)
    log_likelihood = chi2_contingency(contingency_table, correction=False, lambda_='log-likelihood')
    cressie_read = chi2_contingency(contingency_table, correction=False, lambda_='cressie-read')
    print('chi2_dof: ', chi2.dof)
    
    for word in inter:
        O1 = dict1[word]
        O2 = dict2[word]
        O = O1 + O2
        E1 = N1 * O / N
        E2 = N2 * O / N
        LL = 2 * ((O1 * np.log(O1/E1)) + (O2 * np.log(O2/E2)))
        LL_dict[word] = LL
    
    LL_dict = dict(sorted(LL_dict.items(), key=lambda x: x[1], reverse=True))
    print('Descending order:')
    for i, (word, LL) in enumerate(LL_dict.items()):
        if i >= 15: break
        print(word, LL, dict1[word], dict2[word])

    print('Ascending order:')
    LL_dict2 = dict(sorted(LL_dict.items(), key=lambda x: x[1], reverse=False))
    for i, (word, LL) in enumerate(LL_dict2.items()):
        if i >= 15: break
        print(word, LL, dict1[word], dict2[word])

    scc = spearmanr(contingency_table[:,0], contingency_table[:, 1], alternative='greater')

    return (LL_dict, chi2, log_likelihood, cressie_read, scc)

def get_title_name_for_corpora(corpora_name):
    if str.lower(corpora_name) in ['tscc']:
        return 'TSCC'
    elif str.lower(corpora_name) in ['wangchan', 'wangchanberta']:
        return 'Corpora Used by WangChanBERTa'
    elif str.lower(corpora_name) in ['wongnai']:
        return 'Wongnai'
    else:
        return '<Unknown>'

def compare_freq(fname1, fname2, corpora_name1, corpora_name2):
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
    ax1.set_title(f'Histogram of Frequency of Words in {get_title_name_for_corpora(corpora_name1)}')
    ax2.hist(dict2_filtered.values(), bins=300, log=True)
    ax2.set_title(f'Histogram of Frequency of Words in {get_title_name_for_corpora(corpora_name2)}')


    res = compute_stats(dict1_filtered, dict2_filtered)

    return res

def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="compare-word-freq.py",
        description="compare word frequencies between 2 corpora in terms of chi-squared",
    )

    # required
    parser.add_argument(
        "--freq1_path", type=str,
    )
    parser.add_argument(
        "--freq2_path", type=str,
    )
    parser.add_argument(
        "--freq1_name", type=str,
    )
    parser.add_argument(
        "--freq2_name", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str
    )

    args = parser.parse_args()
    
    (LL, chi2, ll, cr, scc) = compare_freq(args.freq1_path, args.freq2_path, args.freq1_name, args.freq2_name)

    if os.path.exists(f'{args.output_dir}/') == False:
            os.makedirs(f'{args.output_dir}/', exist_ok=True)

    plt.savefig(f'{args.output_dir}/plots/hist-{args.freq1_name}-{args.freq2_name}.png')

    print(f'chi2: {chi2.statistic}\n')
    print(f'chi2 (p value): {chi2.pvalue}\n')
    print(f'Log likelihood: {ll.statistic}\n')
    print(f'Log likelihood (p-value): {ll.pvalue}\n')
    print(f'Log likelihood implemented by myself: {np.sum(list(LL.values()))}\n')
    print(f'Cressie-Read power divergence: {cr.statistic}\n')
    print(f'Cressie-Read power divergence (p value): {cr.pvalue}\n')
    print(f'Spearman\'s rank correlation coefficient: {scc.statistic}\n')
    print(f'Spearman\'s rank correlation coefficient (p-value): {scc.pvalue}\n')

    with open(f'{args.output_dir}/log-likelihood-{args.freq1_name}-{args.freq2_name}.txt', 'w') as f:
        print('writing log likelihood...')
        f.write(str(LL))
        print('done writing')

    with open(f'{args.output_dir}/report-{args.freq1_name}-{args.freq2_name}.txt', 'w') as f:
        print('writing report...')
        f.write(f'chi2: {chi2.statistic:.6f}\n')
        f.write(f'chi2 (p value): {chi2.pvalue:.6f}\n')
        f.write(f'Log likelihood: {ll.statistic:.6f}\n')
        f.write(f'Log likelihood (p-value): {ll.pvalue:.6f}\n')
        f.write(f'Cressie-Read power divergence: {cr.statistic}\n')
        f.write(f'Cressie-Read power divergence (p value): {cr.pvalue}\n')
        f.write(f'Spearman\'s rank correlation coefficient: {scc.statistic:.6f}\n')
        f.write(f'Spearman\'s rank correlation coefficient (p-value): {scc.pvalue:.6f}\n')
        print('done writing')

if __name__ == "__main__":
    main()

