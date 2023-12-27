import argparse
import pandas as pd
import re

def get_relevant_article_ids(df):
    lawids_df = df['lawids']
    lawids = []
    for ids in lawids_df:
        lawids.extend(ids.split(','))
    lawids_set = set(lawids)
    print('# Article IDs (before filtering out): ', len(lawids_set))

    ''' remove any articles with id <= 106 '''
    lawids = list(filter(lambda x: int(x.split('-')[1][:3]) > 106, lawids_set))
    print('# Article IDs (after filtering out): ', len(lawids))


    # # Select only OFFENCE AGAINST LIFE AND BODY (288 - 300)
    # ''' remove any articles with id > 300'''
    # lawids = list(filter(lambda x: int(x.split('-')[1][:3]) <= 300, lawids))
    # print('# Article IDs (after filtering out (>300)): ', len(lawids))
    # '''remove any articles from the list: ['CC-289(3)-00', 'CC-298-00']'''
    # lawid_filter_list = ['CC-289(3)-00', 'CC-298-00']
    # lawids = list(filter(lambda x: x not in lawid_filter_list, lawids))
    # print('# Article IDs (after filtering out (from list)): ', len(lawids))

    # Select only OFFENCE AGAINST LIBERTY AND REPUTATION (326 - 333)
    # ''' remove any articles with id > 333 or < 326'''
    # lawids = list(filter(lambda x: 326 <= int(x.split('-')[1][:3]) <= 333, lawids))
    # print('# Article IDs (after filtering out (<326 or > 333)): ', len(lawids))

    '''remove articles that appear less than 5 times among all datapoints'''
    drop_article_list = ['CC-390-00','CC-289(7)-00', 'CC-339-01', 'CC-342-00', 'CC-338-00', 'CC-335-01', 
                        'CC-335bis-00', 'CC-289(3)-00', 'CC-354-00', 'CC-298-00', 'CC-335bis-01']
    lawids = [x for x in lawids if x not in drop_article_list]
    print('# Article IDs (after dropping non frequent articles): ', len(lawids))
    return lawids

def drop_columns(df):
    df = df[['label', 'filtered_fact']]
    return df

def attach_filtered_fact(df):
    pattern = r'<discr>.*?</discr>'
    df['filtered_fact'] = df.fact.apply(lambda x: re.sub(pattern, '', x)).copy()
    return df

def containsLawID(lawids, lawid):
    if lawid in lawids:
        return 1
    else:
        return 0

def attach_multihot_encoding(df, lawids):
    '''
    Add One Column for each LawID to df
    '''
    for label in lawids:
        df[label] = df.lawids.apply(lambda lawids: containsLawID(lawids, label))
    
    '''
    Drop column 
    '''
    df = df[~(df[lawids] == 0).all(axis=1)]
    df = df.reset_index(drop=True)
    return df

def extract_relevant_labels(case_lawids):
    case_lawids = case_lawids.split(',')
    case_lawids = [lawid for lawid in case_lawids if lawid in lawids]
    return case_lawids

def get_label_frequency_order(df):
    cols = df.columns
    label_names = list(cols[20:])
    np_labels = list(map(tuple, df[label_names].sum().items()))
    np_labels.sort(key=lambda x: x[1], reverse=True)
    label2freqord = {x[1][0]: x[0] for x in enumerate(np_labels)}
    return label2freqord
    

def get_most_frequent_label(case_lawids, label2freqord):
    case_lawids.sort(key=lambda x: label2freqord[x])
    return label2id[case_lawids[0]]


def attach_label(df, lawids):
    '''
    Select one most frequent label for each datapoint
    '''
    label2freqord = get_label_frequency_order(df)

    df['filtered_lawids'] = df.lawids.apply(lambda lawids: extract_relevant_labels(lawids))
    df['label'] = df.filtered_lawids.apply(lambda lawids: get_most_frequent_label(lawids, label2freqord))
    return df

def select_unique_dekaid(df):
    max_issue_indices = df.groupby('dekaid')['issueno'].idxmax()
    return df.loc[max_issue_indices]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path', 
        type=str, 
        default='./data/tscc/tscc_v0.1-judgement.csv'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./data/tscc/processed'
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    lawids = get_relevant_article_ids(df)

    id2label = {idx:label for idx, label in enumerate(lawids)}
    label2id = {str(label):idx for idx, label in enumerate(lawids)}

    print(f'before: {len(df)}')
    df = select_unique_dekaid(df)
    print(f'after: {len(df)}')
    df = attach_filtered_fact(df)
    df = attach_multihot_encoding(df, lawids)
    df = attach_label(df, lawids)
    df = drop_columns(df)
    
    '''
    update label ids
    '''
    unique_ids = set(df.label)
    new_id = 0
    label2id_new = {}
    for id in unique_ids:
        label = id2label[id]
        label2id_new[label] = new_id
        new_id += 1
    df['label'] = df.label.apply(lambda id: label2id_new[id2label[id]])
    label2id = label2id_new
    id2label = {y: x for x, y in label2id.items()}

    print('# Article IDs (after selecting most frequent label per case): ', len(set(df.label)))

    df.to_csv(f'{args.output_path}/tscc_cleaned.csv', index=False, encoding='utf-8')