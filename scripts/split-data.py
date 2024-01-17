import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="split-data.py",
        description="Split input dataset into train valid test, in either full-shot or N-shot (per class) manner",
    )
    
    parser.add_argument(
        '--input_path', 
        type=str, 
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
    )
    parser.add_argument(
        '--label_column_name', 
        type=str,
    )
    parser.add_argument(
        '--csv_sep', 
        type=str,
        default=',',
    )
    parser.add_argument(
        '--train_ratio', 
        type=float, 
        default=0.80
    )
    parser.add_argument(
        '--valid_ratio',
        type=float,
        default=0.10
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.10
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=101
    )
    # int K for K-shot (per class), or 'full' for full shot. The freqeuency of input labels must be at least K for all labels!!!
    parser.add_argument(
        '--shot',
        type=str,
        choices=['full', '5', '20', '100'],
        default='full',
    )
    # # Specify whether low-frequency labels are to be included in datasets.
    # parser.add_argument(
    #     '--all_label',
    #     type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']),
    #     default=False
    # )
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, sep=args.csv_sep)

    label_column_name = args.label_column_name
    labels = df[label_column_name]
    label_counts = labels.astype(str).value_counts()

    shot = args.shot
    if shot == 'full':
        K = -1
    else:
        K = int(shot)

    valid_test_ratio = args.valid_ratio + args.test_ratio
    assert abs(valid_test_ratio - (1 - args.train_ratio)) < 1e-6

    if K != -1:
        label_counts = label_counts.to_dict()
        print(f'Label counts: {label_counts}')

        train_df_list = []
        valid_df_list = []
        test_df_list = []
        for label, freq in label_counts.items():
            df_single_label = df[df[label_column_name].astype(str) == label]
            # print(f'label: {label}', f'freq: {freq}', f'{df_single_label}')
            if freq * args.train_ratio >= K:
                # Select exactly K samples into train set
                train_df_single, valid_test_df_single = train_test_split(df_single_label, random_state=args.random_state, train_size=K)
                valid_df_single, test_df_single = train_test_split(valid_test_df_single, random_state=args.random_state, test_size=args.test_ratio/valid_test_ratio)
            else:
                # Select m (m < K) samples into train set according to the input ratio

                if valid_test_ratio * freq < 2:
                    train_df_single, valid_test_df_single = train_test_split(df_single_label, random_state=args.random_state, test_size=2)
                    valid_df_single, test_df_single = train_test_split(valid_test_df_single, random_state=args.random_state, train_size=1 - args.test_ratio/valid_test_ratio)
                else:
                    train_df_single, valid_test_df_single = train_test_split(df_single_label, random_state=args.random_state, test_size=valid_test_ratio)
                    valid_df_single, test_df_single = train_test_split(valid_test_df_single, random_state=args.random_state, train_size=1 - args.test_ratio/valid_test_ratio)
            train_df_list.append(train_df_single)
            valid_df_list.append(valid_df_single)
            test_df_list.append(test_df_single)
        train_df = pd.concat(train_df_list, ignore_index=True)
        valid_df = pd.concat(valid_df_list, ignore_index=True)
        test_df = pd.concat(test_df_list, ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=args.random_state, ignore_index=True)
        valid_df = valid_df.sample(frac=1, random_state=args.random_state, ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=args.random_state, ignore_index=True)

    else:
        one_freq = label_counts[label_counts==1].keys()
        one_freq_idxs = sorted(list(df[df[label_column_name].astype(str).isin(one_freq)].index), reverse=True)
        print('df label indices with only one instance: ', one_freq_idxs)

        one_freq_df = df.iloc[one_freq_idxs]
        df = df.drop(one_freq_idxs)
        
        strat_labels = [x for x in labels if str(x) not in one_freq]
        train_df, valid_test_df = train_test_split(df, random_state=args.random_state, test_size=valid_test_ratio, stratify=strat_labels)
        valid_df, test_df = train_test_split(valid_test_df, random_state=args.random_state, test_size=args.test_ratio/valid_test_ratio)

        train_df = pd.concat([train_df, one_freq_df])

    print('# datapoints in train, valid, test split: ', len(train_df), 
        len(valid_df), len(test_df))
    print('# unique labels in train, valid, test split: ', len(set(train_df[label_column_name])), 
        len(set(valid_df[label_column_name])), len(set(test_df[label_column_name])))

    split_df = { 'train': train_df, 'valid': valid_df, 'test': test_df}
    for split in ['train', 'valid', 'test']:
        print(f'INFO: Begin writing {split} split to "{args.output_dir}/{split}/{split}.txt".')
        if os.path.exists(f'{args.output_dir}/{split}') == False:
            os.makedirs(f'{args.output_dir}/{split}', exist_ok=True)

        split_df[split].to_csv(f'{args.output_dir}/{split}/{split}.csv', sep=args.csv_sep, index=False, encoding='utf-8')

    print('\nINFO: Done writing all split.')