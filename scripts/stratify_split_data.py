import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path', 
        type=str, 
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
    )
    parser.add_argument(
        '--csv_sep', 
        type=str,
        default=',',
    )
    parser.add_argument('--train_ratio', type=float, default=0.80)
    parser.add_argument('--valid_ratio', type=float, default=0.10)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, sep=args.csv_sep)
    labels = df.label
    label_counts = labels.astype(str).value_counts()
    one_freq = label_counts[label_counts==1].keys()

    one_freq_idxs = sorted(list(df[df.label.astype(str).isin(one_freq)].index), reverse=True)
    print('df label indices with only one instance: ', one_freq_idxs)

    one_freq_df = df.iloc[one_freq_idxs]
    df = df.drop(one_freq_idxs)
    
    valid_test_ratio = args.valid_ratio + args.test_ratio

    strat_labels = [x for x in labels if str(x) not in one_freq]
    train_df, valid_test_df = train_test_split(df, random_state=101, test_size=valid_test_ratio, stratify=strat_labels)
    valid_df, test_df = train_test_split(valid_test_df, random_state=101, test_size=args.test_ratio/valid_test_ratio)

    train_df = pd.concat([train_df, one_freq_df])

    print('# unique labels in train, valid, test split: ', len(set(train_df.label)), len(set(valid_df.label)), len(set(test_df.label)))


    split_df = { 'train': train_df, 'valid': valid_df, 'test': test_df}
    for split in ['train', 'valid', 'test']:
        print(f'INFO: Begin writing {split} split to "{args.output_dir}/{split}/{split}.txt".')
        if os.path.exists(f'{args.output_dir}/{split}') == False:
            os.makedirs(f'{args.output_dir}/{split}', exist_ok=True)

        split_df[split].to_csv(f'{args.output_dir}/{split}/{split}.csv', index=False, encoding='utf-8')

    print('\nINFO: Done writing all split.')