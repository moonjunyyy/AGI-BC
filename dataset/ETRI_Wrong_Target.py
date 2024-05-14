   
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
from utils.knusl import KnuSL

class ETRI_2022_In_BC_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2022_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2022_2s")
        if os.path.isdir(self.path) == False:
            print("Copy etri2022_2s.zip")
            import shutil
            import zipfile
            shutil.copy("/data/datasets/etri2022_2s.zip", path)
            zipfile.ZipFile(f"{path}/etri2022_2s.zip").extractall(path)
            shutil.rmtree(f"{path}/etri2022_2s.zip", ignore_errors=True)
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "etri2022_2s.tsv"), sep='\t', index_col=0)
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        
        assert len(self.dataframe) == len(self.dataframe)

        generator = torch.Generator()
        generator.manual_seed(42)
        train_index = self.dataframe['folder'].unique()
        index = torch.randperm(len(train_index), generator=generator)
        train_index = train_index[index[:int(len(index)*0.8)]]
        test_index = self.dataframe['folder'].unique()
        test_index = test_index[index[int(len(index)*0.8):]]

        if self.train:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(train_index)]
            if self.balanced:
                _no_bc_dataframe = self.dataframe[self.dataframe['BC'] == 0]
                _bc_dataframe = self.dataframe[self.dataframe['BC'] != 0]
                if len(_no_bc_dataframe) > len(_bc_dataframe):
                    _no_bc_dataframe = _no_bc_dataframe.sample(len(_bc_dataframe), random_state=42)
                else:
                    _bc_dataframe = _bc_dataframe.sample(len(_no_bc_dataframe), random_state=42)
            self.dataframe = pd.concat([_no_bc_dataframe, _bc_dataframe])
        else:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(test_index)]

        # self.input_dataframe = self.input_dataframe[self.input_dataframe['BC'] > 1]
        # self.target_dataframe = self.target_dataframe[self.target_dataframe['BC'] > 1]
        print(self.dataframe)
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):

        ret = {}

        item = self.dataframe.iloc[index]

        idx = item['filename']
        trans = item['transcript']
        label = item['BC']
        target_trans = item['back']
        input_path = os.path.join(self.path, "audio", "front", f"{str(idx)}.wav")
        if not self.train:
            sample = self.dataframe[self.dataframe['BC'] == label].sample(1, random_state=index).iloc[0]
            idx = sample['filename']
            target_trans = sample['back']
        target_path = os.path.join(self.path, "audio", "back", f"{str(idx)}.wav")

        input_audio, sr = torchaudio.load(input_path)
        input_audio = torchaudio.transforms.Resample(sr, 16000)(input_audio)
        sr = 16000
        input_audio = input_audio[:, -int(self.length*sr):]
        if input_audio.size(1) != int(self.length * sr):
            input_audio = F.pad(input_audio, (0, int(sr * self.length) - input_audio.size(1)), "constant", 0)
        if input_audio.size(0) != 1:
            input_audio = input_audio.sum(0, keepdim=True)

        target_audio, sr = torchaudio.load(target_path)
        target_audio = torchaudio.transforms.Resample(sr, 16000)(target_audio)
        sr = 16000
        target_audio = target_audio[:, :int(self.predict_length*sr)]
        if target_audio.size(1) != int(self.predict_length * sr):
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
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        target_trans = self.tokenizer(target_trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = input_audio
        ret['target_audio'] = target_audio
        ret['label'] = label
        ret['text'] = trans
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_2023_In_BC_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2023_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2023_2s")
        if os.path.isdir(self.path) == False:
            print("Copy etri2023_2s.zip")
            import shutil
            import zipfile
            shutil.copy("/data/datasets/etri2023_2s.zip", path)
            zipfile.ZipFile(f"{path}/etri2023_2s.zip").extractall(path)
            shutil.rmtree(f"{path}/etri2023_2s.zip", ignore_errors=True)
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "etri2023_2s.tsv"), sep='\t', index_col=0)
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        
        assert len(self.dataframe) == len(self.dataframe)

        generator = torch.Generator()
        generator.manual_seed(42)
        train_index = self.dataframe['folder'].unique()
        index = torch.randperm(len(train_index), generator=generator)
        train_index = train_index[index[:int(len(index)*0.8)]]
        test_index = self.dataframe['folder'].unique()
        test_index = test_index[index[int(len(index)*0.8):]]

        if self.train:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(train_index)]
            if self.balanced:
                _no_bc_dataframe = self.dataframe[self.dataframe['BC'] == 0]
                _bc_dataframe = self.dataframe[self.dataframe['BC'] != 0]
                if len(_no_bc_dataframe) > len(_bc_dataframe):
                    _no_bc_dataframe = _no_bc_dataframe.sample(len(_bc_dataframe), random_state=42)
                else:
                    _bc_dataframe = _bc_dataframe.sample(len(_no_bc_dataframe), random_state=42)
            self.dataframe = pd.concat([_no_bc_dataframe, _bc_dataframe])
        else:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(test_index)]

        # self.input_dataframe = self.input_dataframe[self.input_dataframe['BC'] > 1]
        # self.target_dataframe = self.target_dataframe[self.target_dataframe['BC'] > 1]
        print(self.dataframe)
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):

        ret = {}

        item = self.dataframe.iloc[index]

        idx = item['filename']
        trans = item['transcript']
        label = item['BC']
        target_trans = item['back']
        # get the random id from in same label 
        input_path = os.path.join(self.path, "audio", "front", f"{str(idx)}.wav")
        if not self.train:
            sample = self.dataframe[self.dataframe['BC'] == label].sample(1, random_state=index).iloc[0]
            idx = sample['filename']
            target_trans = sample['back']
        target_path = os.path.join(self.path, "audio", "back", f"{str(idx)}.wav")

        input_audio, sr = torchaudio.load(input_path)
        input_audio = torchaudio.transforms.Resample(sr, 16000)(input_audio)
        sr = 16000
        input_audio = input_audio[:, -int(self.length*sr):]
        if input_audio.size(1) != int(self.length * sr):
            input_audio = F.pad(input_audio, (0, int(sr * self.length) - input_audio.size(1)), "constant", 0)
        if input_audio.size(0) != 1:
            input_audio = input_audio.sum(0, keepdim=True)

        target_audio, sr = torchaudio.load(target_path)
        target_audio = torchaudio.transforms.Resample(sr, 16000)(target_audio)
        sr = 16000
        target_audio = target_audio[:, :int(self.predict_length*sr)]
        if target_audio.size(1) != int(self.predict_length * sr):
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
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        target_trans = self.tokenizer(target_trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = input_audio
        ret['target_audio'] = target_audio
        ret['label'] = label
        ret['text'] = trans
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_All_In_BC_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_In_BC_Dataset(path, tokenizer, train, balanced, length, predict_length)
        self.dataset_2023 = ETRI_2023_In_BC_Dataset(path, tokenizer, train, balanced, length, predict_length)

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()

class ETRI_2022_Wrong_Target_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2022_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2022_2s")
        if os.path.isdir(self.path) == False:
            print("Copy etri2022_2s.zip")
            import shutil
            import zipfile
            shutil.copy("/data/datasets/etri2022_2s.zip", path)
            zipfile.ZipFile(f"{path}/etri2022_2s.zip").extractall(path)
            shutil.rmtree(f"{path}/etri2022_2s.zip", ignore_errors=True)
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "etri2022_2s.tsv"), sep='\t', index_col=0)
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        
        assert len(self.dataframe) == len(self.dataframe)

        generator = torch.Generator()
        generator.manual_seed(42)
        train_index = self.dataframe['folder'].unique()
        index = torch.randperm(len(train_index), generator=generator)
        train_index = train_index[index[:int(len(index)*0.8)]]
        test_index = self.dataframe['folder'].unique()
        test_index = test_index[index[int(len(index)*0.8):]]

        if self.train:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(train_index)]
            if self.balanced:
                _no_bc_dataframe = self.dataframe[self.dataframe['BC'] == 0]
                _bc_dataframe = self.dataframe[self.dataframe['BC'] != 0]
                if len(_no_bc_dataframe) > len(_bc_dataframe):
                    _no_bc_dataframe = _no_bc_dataframe.sample(len(_bc_dataframe), random_state=42)
                else:
                    _bc_dataframe = _bc_dataframe.sample(len(_no_bc_dataframe), random_state=42)
            self.dataframe = pd.concat([_no_bc_dataframe, _bc_dataframe])
        else:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(test_index)]

        # self.input_dataframe = self.input_dataframe[self.input_dataframe['BC'] > 1]
        # self.target_dataframe = self.target_dataframe[self.target_dataframe['BC'] > 1]
        print(self.dataframe)
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):

        ret = {}

        item = self.dataframe.iloc[index]

        idx = item['filename']
        trans = item['transcript']
        target_trans = item['back']
        label = item['BC']
        input_path = os.path.join(self.path, "audio", "front", f"{str(idx)}.wav")
        if not self.train:
            sample = self.dataframe.sample(1, random_state=index).iloc[0]
            idx = sample['filename']
            target_trans = sample['back']
        target_path = os.path.join(self.path, "audio", "back", f"{str(idx)}.wav")

        input_audio, sr = torchaudio.load(input_path)
        input_audio = torchaudio.transforms.Resample(sr, 16000)(input_audio)
        sr = 16000
        input_audio = input_audio[:, -int(self.length*sr):]
        if input_audio.size(1) != int(self.length * sr):
            input_audio = F.pad(input_audio, (0, int(sr * self.length) - input_audio.size(1)), "constant", 0)
        if input_audio.size(0) != 1:
            input_audio = input_audio.sum(0, keepdim=True)

        target_audio, sr = torchaudio.load(target_path)
        target_audio = torchaudio.transforms.Resample(sr, 16000)(target_audio)
        sr = 16000
        target_audio = target_audio[:, :int(self.predict_length*sr)]
        if target_audio.size(1) != int(self.predict_length * sr):
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
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        target_trans = self.tokenizer(target_trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = input_audio
        ret['target_audio'] = target_audio
        ret['label'] = label
        ret['text'] = trans
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_2023_Wrong_Target_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2023_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2023_2s")
        if os.path.isdir(self.path) == False:
            print("Copy etri2023_2s.zip")
            import shutil
            import zipfile
            shutil.copy("/data/datasets/etri2023_2s.zip", path)
            zipfile.ZipFile(f"{path}/etri2023_2s.zip").extractall(path)
            shutil.rmtree(f"{path}/etri2023_2s.zip", ignore_errors=True)
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "etri2023_2s.tsv"), sep='\t', index_col=0)
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        
        assert len(self.dataframe) == len(self.dataframe)

        generator = torch.Generator()
        generator.manual_seed(42)
        train_index = self.dataframe['folder'].unique()
        index = torch.randperm(len(train_index), generator=generator)
        train_index = train_index[index[:int(len(index)*0.8)]]
        test_index = self.dataframe['folder'].unique()
        test_index = test_index[index[int(len(index)*0.8):]]

        if self.train:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(train_index)]
            if self.balanced:
                _no_bc_dataframe = self.dataframe[self.dataframe['BC'] == 0]
                _bc_dataframe = self.dataframe[self.dataframe['BC'] != 0]
                if len(_no_bc_dataframe) > len(_bc_dataframe):
                    _no_bc_dataframe = _no_bc_dataframe.sample(len(_bc_dataframe), random_state=42)
                else:
                    _bc_dataframe = _bc_dataframe.sample(len(_no_bc_dataframe), random_state=42)
            self.dataframe = pd.concat([_no_bc_dataframe, _bc_dataframe])
        else:
            self.dataframe = self.dataframe[self.dataframe['folder'].isin(test_index)]

        # self.input_dataframe = self.input_dataframe[self.input_dataframe['BC'] > 1]
        # self.target_dataframe = self.target_dataframe[self.target_dataframe['BC'] > 1]
        print(self.dataframe)
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):

        ret = {}

        item = self.dataframe.iloc[index]

        idx = item['filename']
        trans = item['transcript']
        target_trans = item['back']
        label = item['BC']
        # get the random id from in same label 
        input_path = os.path.join(self.path, "audio", "front", f"{str(idx)}.wav")
        if not self.train:
            sample = self.dataframe.sample(1, random_state=index).iloc[0]
            idx = sample['filename']
            target_trans = sample['back']
        target_path = os.path.join(self.path, "audio", "back", f"{str(idx)}.wav")

        input_audio, sr = torchaudio.load(input_path)
        input_audio = torchaudio.transforms.Resample(sr, 16000)(input_audio)
        sr = 16000
        input_audio = input_audio[:, -int(self.length*sr):]
        if input_audio.size(1) != int(self.length * sr):
            input_audio = F.pad(input_audio, (0, int(sr * self.length) - input_audio.size(1)), "constant", 0)
        if input_audio.size(0) != 1:
            input_audio = input_audio.sum(0, keepdim=True)

        target_audio, sr = torchaudio.load(target_path)
        target_audio = torchaudio.transforms.Resample(sr, 16000)(target_audio)
        sr = 16000
        target_audio = target_audio[:, :int(self.predict_length*sr)]
        if target_audio.size(1) != int(self.predict_length * sr):
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
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        target_trans = self.tokenizer(target_trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = input_audio
        ret['target_audio'] = target_audio
        ret['label'] = label
        ret['text'] = trans
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_All_Wrong_Target_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Wrong_Target_Dataset(path, tokenizer, train, balanced, length, predict_length)
        self.dataset_2023 = ETRI_2023_Wrong_Target_Dataset(path, tokenizer, train, balanced, length, predict_length)

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()

