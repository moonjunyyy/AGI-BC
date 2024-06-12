import os
import math
import logging
import pandas as pd
import decord
from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from utils.knusl import KnuSL
from konlpy.tag import Okt 
import gensim
import random
import torchvision
import numpy as np
import av

class ETRI22_2S_Video_Generation_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI2022_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2022_end")

        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")
    
        self.dataframe = pd.read_csv(os.path.join(self.path, "etri2022_end.tsv"), sep='\t', index_col=0)
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))

        self.drop_list =   [6660, 6661, 6670, 6671, 6678, 6679, 6680, 6681, 6682, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6691,
                            6692, 6693, 6708, 6709, 6710, 6711, 6720, 6721, 6722, 6723, 6730, 6731, 6744, 6745, 6746, 6747, 6758, 6759,
                            6760, 6761, 6774, 6775, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793, 6798, 6799, 6800, 6801, 6808, 6809,
                            6810, 6811, 6824, 6825, 6832, 6833, 6834, 6835, 6836, 6837, 6848, 6849, 6850, 6851, 6866, 6867, 6868, 6869,
                            6870, 6871, 6872, 6873, 6874, 6875, 6886, 6887, 6900, 6901, 6902, 6903, 6910, 6911, 6912, 6913, 6916, 6917,
                            6924, 6925, 6926, 6927, 6928, 6929, 6930, 6931, 6932, 6933, 6936, 6937, 6938, 6939, 6942, 6943, 6944, 6945,
                            6946, 6947, 6948, 6949, 6950, 6951, 6952, 6953, 6954, 6955, 6968, 6969, 6970, 6971, 6976, 6977, 6978, 6979,
                            6980, 6981, 6992, 6993, 6994, 6995, 6996, 6997, 6998, 6999, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009,
                            7010, 7011, 7014, 7015, 7018, 7019, 7020, 7021, 7022, 7023, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7035,
                            7036, 7037, 7038, 7039, 7040, 7041, 7042, 7043, 7044, 7045, 7046, 7047, 7048, 7049, 7050, 7051, 7052, 7053,
                            7054, 7055, 7056, 7057, 7058, 7059, 7060, 7061, 7062, 7063, 7074, 7075, 7076, 7077, 7078, 7079, 7080, 7081,
                            7082, 7083, 7084, 7085, 7096, 7097, 7098, 7099, 7100, 7101, 7102, 7103, 7104, 7105, 7116, 7117, 7118, 7119,
                            7120, 7121, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135]
                        #    16968, 12025, 15058]
        
        self.dataframe.drop(self.drop_list, inplace=True)    

        assert len(self.dataframe) == len(self.dataframe)

        self.BC_dataframe = self.dataframe[self.dataframe["BC"]==0]
        self.NoBC_dataframe = self.dataframe.drop(self.BC_dataframe.index)

        self.BC_train_dataframe = self.BC_dataframe.sample(frac=0.8, random_state=42)
        self.BC_test_dataframe = self.BC_dataframe.drop(self.BC_train_dataframe.index)
        
        self.NoBC_train_dataframe = self.NoBC_dataframe.sample(frac=0.8, random_state=42)
        self.NoBC_test_dataframe = self.NoBC_dataframe.drop(self.NoBC_train_dataframe.index)
        """
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)
        """
           
        if self.train:
            self.dataframe = pd.concat([self.BC_train_dataframe, self.NoBC_train_dataframe]).sample(frac=1, random_state=42)
        else:
            self.dataframe = pd.concat([self.BC_test_dataframe, self.NoBC_test_dataframe]).sample(frac=1, random_state=42)
    #    print(self.dataframe)
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
        
        input_path = os.path.join(self.path, "audio/front", f"{str(idx)}.wav")
        target_path = os.path.join(self.path, "audio/back", f"{str(idx)}.wav")    
    
        audio, sr = torchaudio.load(input_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000
        audio = audio[:, -int(self.length*sr):]
        if audio.size(1) != int(sr * self.length):
            audio = F.pad(audio, (0, int(sr * self.length) - audio.size(1)), "constant", 0)
        if audio.size(0) != 1:
            audio = audio.sum(0, keepdim=True)
            
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
        

        video_path = os.path.join(self.path, "video/front", f"{str(idx)}.mp4")
        video_reader = decord.VideoReader(video_path)
        length = video_reader.get_length()
        fps = video_reader.get_avg_fps()
        indices = np.linspace(start=0, stop=int(fps*1.5), num=16, endpoint=False)
        video = video_reader.get_batch(indices).asnumpy()
        video = video.transpose(0,3,1,2)
        target_indices = np.linspace(start=length-int(fps*0.5), stop=length-1, num=8, endpoint=False)
        target_video = video_reader.get_batch(target_indices).asnumpy()
        target_video = target_video.transpose(0,3,1,2)

        # container = av.open(video_path)
        # indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        # video = self.read_video_pyav(container, indices)
        
        
        ret['video'] = video
        ret['audio'] = audio
        ret['label'] = label
        ret['text'] = trans
        ret['target_video'] = target_video
        ret['target_audio'] = target_audio
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
   
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()    
    
    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
    #    converted_len = int(clip_len * frame_sample_rate)
    #    end_idx = np.random.randint(converted_len, seg_len)
    #    start_idx = end_idx - converted_len
    #    indices = np.linspace(start_idx, end_idx, num=clip_len)
    #    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        indices = np.linspace(seg_len-46, seg_len-1, num=clip_len)
        indices = np.clip(indices, seg_len-46, seg_len-1).astype(np.int64)
        return indices

    
    def read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)

        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
class ETRI23_2S_Video_Generation_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI2023_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2023_end")
        
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")
    
        self.dataframe = pd.read_csv(os.path.join(self.path, "etri2023_end.tsv"), sep='\t', index_col=0)
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        
        assert len(self.dataframe) == len(self.dataframe)
        
        self.BC_dataframe = self.dataframe[self.dataframe["BC"]==0]
        self.NoBC_dataframe = self.dataframe.drop(self.BC_dataframe.index)

        self.BC_train_dataframe = self.BC_dataframe.sample(frac=0.8, random_state=42)
        self.BC_test_dataframe = self.BC_dataframe.drop(self.BC_train_dataframe.index)
        
        self.NoBC_train_dataframe = self.NoBC_dataframe.sample(frac=0.8, random_state=42)
        self.NoBC_test_dataframe = self.NoBC_dataframe.drop(self.NoBC_train_dataframe.index)
        """
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)
        """
           
        if self.train:
            self.dataframe = pd.concat([self.BC_train_dataframe, self.NoBC_train_dataframe]).sample(frac=1, random_state=42)
        else:
            self.dataframe = pd.concat([self.BC_test_dataframe, self.NoBC_test_dataframe]).sample(frac=1, random_state=42)
            
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
        
        input_path = os.path.join(self.path, "audio/front", f"{str(idx)}.wav")
        target_path = os.path.join(self.path, "audio/back", f"{str(idx)}.wav")    
    
        audio, sr = torchaudio.load(input_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000
        audio = audio[:, -int(self.length*sr):]
        if audio.size(1) != int(sr * self.length):
            audio = F.pad(audio, (0, int(sr * self.length) - audio.size(1)), "constant", 0)
        if audio.size(0) != 1:
            audio = audio.sum(0, keepdim=True)
            
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
        

        video_path = os.path.join(self.path, "video/front", f"{str(idx)}.mp4")
        video_reader = decord.VideoReader(video_path)
        length = video_reader.get_length()
        fps = video_reader.get_avg_fps()
        indices = np.linspace(start=0, stop=int(fps*1.5), num=16, endpoint=False)
        video = video_reader.get_batch(indices).asnumpy()
        video = video.transpose(0,3,1,2)
        target_indices = np.linspace(start=length-int(fps*0.5), stop=length-1, num=8, endpoint=False)
        target_video = video_reader.get_batch(target_indices).asnumpy()
        target_video = target_video.transpose(0,3,1,2)

        # container = av.open(video_path)
        # indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        # video = self.read_video_pyav(container, indices)
        
        
        ret['video'] = video
        ret['audio'] = audio
        ret['label'] = label
        ret['text'] = trans
        ret['target_video'] = target_video
        ret['target_audio'] = target_audio
        ret['target_text'] = target_trans
        ret['sentiment'] = sentiment
        return ret
   
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()

class ETRI_ALL_2S_Video_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI22_2S_Video_Generation_Dataset(path, tokenizer, train, balanced, length, predict_length)
        self.dataset_2023 = ETRI23_2S_Video_Generation_Dataset(path, tokenizer, train, balanced, length, predict_length)

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()