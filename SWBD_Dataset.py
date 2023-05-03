import os
import shutil
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import Dataset
from typing import Callable
from knusl import KnuSL
import math
from decord import VideoReader, cpu
import numpy as np
import re

class SWBD_Dataset(Dataset):
    def __init__(self, path, tokenizer, length :float = 1.5) -> None:
        super().__init__()
        # self.path = os.path.join(path, "ETRI_Backchannel_Corpus_2022")
        print("Load SWBD_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "swbd")
        shutil.rmtree(self.path, ignore_errors=True)
        if os.path.isdir(self.path) == False:
            # os.makedirs(self.path, exist_ok=True)
            os.system(f"cp /data/datasets/swbd.tar {path}/")
            os.system(f"chmod 777 {path}/swbd.tar")
            os.system(f"tar -xvf {path}/swbd.tar -C {path}")
            os.system(f"chmod -R 777  {self.path}/*")
            os.system(f"rm {path}/swbd.tar")
        self.length = length
        self.dataframe = pd.read_csv(os.path.join(self.path, "swbd.tsv"), sep='\t', index_col=0)
        # self.dataframe.rename( columns={'Unnamed: 0':'filename'}, inplace=True )
        self.path = os.path.join(path, "swbd", "clip")
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        # print(self.dataframe)
        # self.dataframe = self.dataframe[(self.dataframe['end']-self.dataframe['start'])>(self.length)]
        # self.dataframe = self.dataframe[~self.dataframe['folder'].isin([2,12,31])]
        print(self.dataframe)
        bad_idx = []
        for idx, row in self.dataframe.iterrows():
            if idx in bad_idx:
                continue
            try:                
                path = os.path.join(self.path, f"{idx}.wav")
                audio, sr = torchaudio.load(path)
                if audio.numel() == 0:
                    print(f"audio cannot be loaded : {path}")
                    raise
            except:
                bad_idx.append(idx)
        print(f"Bad file: {bad_idx}")

        self.sentiment_dict = {}
        with open('data/subjclueslen1-HLTEMNLP05.tff', 'r', encoding='utf-8') as f:
            # sentiment_dict = { re.split(" =\n", line) for line in f.readlines()}
            for line in f.readlines():
                line = re.split("=| |\n", line)
                if line[11] == 'neutral':
                    self.sentiment_dict[line[5]] = 0
                elif line[11] == 'positive':
                    if line[0] == 'strongsubj':
                        self.sentiment_dict[line[5]] = 2
                    else:
                        self.sentiment_dict[line[5]] = 1
                elif line[11] == 'negative':
                    if line[0] == 'strongsubj':
                        self.sentiment_dict[line[5]] = -2
                    else:
                        self.sentiment_dict[line[5]] = -1
        self.dataframe = self.dataframe[~self.dataframe['filename'].isin(bad_idx)]

        df0 = self.dataframe[self.dataframe['BC'] == 0]
        df1 = self.dataframe[self.dataframe['BC'] == 1]

        self.dataframe = pd.concat([df0.sample(n=len(df1), random_state=42), df1], ignore_index=True)
        self.dataframe = self.dataframe.sample(frac=1, random_state=42)
        print(self.dataframe['BC'].value_counts())

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        ret = {}

        item = self.dataframe.iloc[index]
        # print(item)
        idx = item['filename']
        trans = item['transcript']
        lable = item['BC']
        start = item['start']
        end   = item['end']
        role  = item['role']
        role  = role == 'B'

        path = os.path.join(self.path, f"{str(idx)}.wav")
        
        audio, sr = torchaudio.load(path)
        audio = audio[role:role+1, -int(self.length*sr):]
        # print(audio.shape)
        if audio.size(1) != int(sr * 1.5):
            audio = F.pad(audio, (0, int(sr * 1.5) - audio.size(1)), "constant", 0)

        sentiment = torch.zeros(5)
        for word in trans.split(' '):
            if word in self.sentiment_dict:
                sentiment[int(self.sentiment_dict[word])] += 1
            else:
                sentiment[0] += 1
        sentiment = sentiment / sentiment.sum()
        # print(sentiment)

        # Tokenize with padding into 64
        trans = self.tokenizer(trans, padding='max_length', max_length=64, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = audio
        ret['label'] = lable
        ret['text'] = trans
        ret['sentiment'] = sentiment
        return ret