import argparse
from transformers import AutoTokenizer
import pandas as pd
import torch
import glob
from tqdm.auto import tqdm
from thai2transformers.preprocess import process_transformers

class FinetunerDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            tokenizer, 
            data_dir,
            text_column_name,
            label_column_name,
            ext='.csv',
            max_length=416,
            sep=',',
            preprocess=True,
        ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        print(self.fnames)

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.preprocess = preprocess
        self.sep = sep
        self.unique_labels = set()
        self._build()
        

    def __getitem__(self, idx):
        item = {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_masks[idx], 'labels': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)

    def _build(self):
        print('Files in the specified directory: ', self.fnames)
    
        df_list = []
        for fname in tqdm(self.fnames):
            df_temp = pd.read_csv(fname, sep=self.sep)
            df_list.append(df_temp)

        df = pd.concat(df_list)            
        texts = list(df[self.text_column_name].values)
        labels = list(df[self.label_column_name].values)
        self.unique_labels.update(labels)

        # preprocess texts
        if self.preprocess:
            texts = [process_transformers(text) for text in texts]

        # tokenize
        tokenized_inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        self.input_ids = tokenized_inputs['input_ids']
        self.attention_masks = tokenized_inputs['attention_mask']
        self.labels = torch.tensor(labels)


class CompressorKNNDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data_dir,
            text_column_name,
            label_column_name,
            ext='.csv',
            sep=',',
            preprocess=True,
        ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        print(self.fnames)

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.texts = []
        self.labels = []
        self.preprocess = preprocess
        self.sep = sep
        self.unique_labels = set()
        self._build()
        

    def __getitem__(self, idx):
        item = {'texts': self.texts[idx], 'labels': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)

    def _build(self):
        print('Files in the specified directory: ', self.fnames)
    
        df_list = []
        for fname in tqdm(self.fnames):
            df_temp = pd.read_csv(fname, sep=self.sep)
            df_list.append(df_temp)
        df = pd.concat(df_list)

        texts = list(df[self.text_column_name].values)
        labels = list(df[self.label_column_name].values)
        self.unique_labels.update(labels)

        # preprocess texts
        if self.preprocess:
            texts = [process_transformers(text) for text in texts]
        
        self.texts = texts
        self.labels = labels