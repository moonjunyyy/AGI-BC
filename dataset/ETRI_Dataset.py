import os
import math
import logging
import pandas as pd
from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from util.knusl import KnuSL

class ETRI_Corpus_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, transform : Callable=None, length :float = 1.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "ETRI_Corpus_Clip")
        if os.path.isdir(self.path) == False:
            os.system(f"mkdir {self.path}")
            os.system(f"cp /data/datasets/ETRI_Corpus_Clip.tar {path}/")
            os.system(f"chmod 777 {path}/ETRI_Corpus_Clip.tar")
            os.system(f"tar -xvf {path}/ETRI_Corpus_Clip.tar -C {path}")
            os.system(f"chmod -R 777  {self.path}/*")
            os.system(f"rm {path}/ETRI_Corpus_Clip.tar")
        self.train = train
        self.length = length
        self.dataframe = pd.read_csv(os.path.join(self.path, "etri.tsv"), sep='\t', index_col=0)
        # self.dataframe.rename( columns={'Unnamed: 0':'filename'}, inplace=True )
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        
        generator = torch.Generator()
        generator.manual_seed(42)
        train_index = self.dataframe['folder'].unique()
        index = torch.randperm(len(train_index), generator=generator)
        train_index = train_index[index[:int(len(index)*0.8)]]
        test_index = self.dataframe['folder'].unique()
        test_index = test_index[index[int(len(index)*0.8):]]

        if self.train:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(train_index)]
            
            dataframe = self.dataframe[self.dataframe['BC'] != 0]
            dataframe = dataframe.append(self.dataframe[self.dataframe['BC'] == 0].sample(n=len(dataframe), replace=True))
            self.dataframe = dataframe
        else:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(test_index)]

        if 'class' in self.dataframe.columns:
            self.dataframe['BC'] = self.dataframe['class']
            self.dataframe = self.dataframe.drop(['class'], axis=1)
        if 'text' in self.dataframe.columns:
            self.dataframe['transcript'] = self.dataframe['text']
            self.dataframe = self.dataframe.drop(['text'], axis=1)

        print(self.dataframe)
        print(self.dataframe['BC'].value_counts())

        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        ret = {}

        item = self.dataframe.iloc[index]
        idx = item['filename']
        trans = item['transcript']
        lable = item['BC']
        path = os.path.join(self.path, "audio", f"{str(idx)}.wav")
        
        audio, sr = torchaudio.load(path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000
        audio = audio[:, -int(self.length*sr):]
        if audio.size(1) != int(sr * self.length):
            audio = F.pad(audio, (0, int(sr * self.length) - audio.size(1)), "constant", 0)
        if audio.size(0) != 1:
            audio = audio.sum(0, keepdim=True)

        sentiment = torch.zeros(5)
        for word in trans.split():
            r_word, s_word = KnuSL.data_list(word)
            if s_word != 'None':
                sentiment[int(s_word)] += 1
            else:
                sentiment[0] += 1
        sentiment = sentiment / sentiment.sum()
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = audio
        ret['label'] = lable
        ret['text'] = trans
        ret['sentiment'] = sentiment
        return ret
   
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().to_numpy()
    

class ETRI_Generation_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri_new1")
        if os.path.isdir(self.path) == False:
            os.system(f"mkdir {self.path}")
            os.system(f"cp /data/datasets/etri_new1.zip {path}/")
            os.system(f"chmod 777 {path}/etri_new1.zip")
            os.system(f"unzip etri_new1.zip -d {path}")
            os.system(f"chmod -R 777  {self.path}/*")
            os.system(f"rm {path}/etri_new1.zip")

        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.input_dataframe = pd.read_csv(os.path.join(self.path, "etri_front.tsv"), sep='\t', index_col=0)
        self.input_dataframe = self.input_dataframe.assign(filename=range(len(self.input_dataframe)))
        self.target_dataframe = pd.read_csv(os.path.join(self.path, "etri_back.tsv"), sep='\t', index_col=0)
        self.target_dataframe = self.target_dataframe.assign(filename=range(len(self.target_dataframe)))
        
        assert len(self.input_dataframe) == len(self.target_dataframe)

        generator = torch.Generator()
        generator.manual_seed(42)
        train_index = self.input_dataframe['folder'].unique()
        index = torch.randperm(len(train_index), generator=generator)
        train_index = train_index[index[:int(len(index)*0.8)]]
        test_index = self.dataframe['folder'].unique()
        test_index = test_index[index[int(len(index)*0.8):]]

        if self.train:
            self.input_dataframe = self.input_dataframe[self.input_dataframe['folder'].isin(train_index)]
            self.target_dataframe = self.target_dataframe[self.target_dataframe['folder'].isin(train_index)]
            if self.balanced:
                _no_bc_input_dataframe = self.input_dataframe[self.input_dataframe['BC'] == 0]
                _bc_input_dataframe = self.input_dataframe[self.input_dataframe['BC'] != 0]
                _no_bc_target_dataframe = self.target_dataframe[self.target_dataframe['BC'] == 0]
                _bc_target_dataframe = self.target_dataframe[self.target_dataframe['BC'] != 0]
                if len(_no_bc_input_dataframe) > len(_bc_input_dataframe):
                    _no_bc_input_dataframe = _no_bc_input_dataframe.sample(len(_bc_input_dataframe), seed=42)
                    _no_bc_target_dataframe = _no_bc_target_dataframe.sample(len(_bc_target_dataframe), seed=42)
                else:
                    _bc_input_dataframe = _bc_input_dataframe.sample(len(_no_bc_input_dataframe), seed=42)
                    _bc_target_dataframe = _bc_target_dataframe.sample(len(_no_bc_target_dataframe), seed=42)
            self.input_dataframe = pd.concat([_no_bc_input_dataframe, _bc_input_dataframe])
            self.target_dataframe = pd.concat([_no_bc_target_dataframe, _bc_target_dataframe])
        else:
            self.input_dataframe = self.input_dataframe[self.input_dataframe['folder'].isin(test_index)]
            self.target_dataframe = self.target_dataframe[self.target_dataframe['folder'].isin(test_index)]

        print(self.input_dataframe)
        print(self.input_dataframe['BC'].value_counts())

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):

        ret = {}

        input_item = self.input_dataframe.iloc[index]
        target_item = self.target_dataframe.iloc[index]

        input_idx = input_item['filename']
        target_idx = target_item['filename']

        trans = input_item['transcript']
        target_trans = target_item['transcript']
        lable = input_item['BC']

        input_path = os.path.join(self.path, "etri_front", f"{str(input_idx)}.wav")
        target_path = os.path.join(self.path, "etri_back", f"{str(target_idx)}.wav")

        input_audio, sr = torchaudio.load(input_path)
        input_audio = torchaudio.transforms.Resample(sr, 16000)(input_audio)
        sr = 16000
        input_audio = input_audio[:, -int(self.length*sr):]
        if input_audio.size(1) != int(self.length * 1.5):
            input_audio = F.pad(input_audio, (0, int(sr * self.length) - input_audio.size(1)), "constant", 0)
        if input_audio.size(0) != 1:
            input_audio = input_audio.sum(0, keepdim=True)

        target_audio, sr = torchaudio.load(target_path)
        target_audio = torchaudio.transforms.Resample(sr, 16000)(target_audio)
        sr = 16000
        target_audio = target_audio[:, -int(self.predict_length*sr):]
        if target_audio.size(1) != int(self.predict_length * 1.5):
            target_audio = F.pad(target_audio, (0, int(sr * self.predict_length) - target_audio.size(1)), "constant", 0)
        if target_audio.size(0) != 1:
            target_audio = target_audio.sum(0, keepdim=True)
            
        sentiment = torch.zeros(5)
        for word in trans.split():
            r_word, s_word = KnuSL.data_list(word)
            if s_word != 'None':
                sentiment[int(s_word)] += 1
            else:
                sentiment[0] += 1
        sentiment = sentiment / sentiment.sum()
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")
        for k in trans.keys(): trans[k] = trans[k].squeeze()

        target_trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")
        for k in target_trans.keys() : target_trans[k] = target_trans.squeeze()
      
        ret['audio'] = input_audio
        ret['target_audio'] = target_audio
        ret['label'] = lable
        ret['text'] = trans
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().to_numpy()