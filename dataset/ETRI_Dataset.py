import os
import math
import logging
import asyncio
import threading
import subprocess
import pandas as pd
from typing import Callable, Optional
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from utils.knusl import KnuSL
from utils.threads import Thread_With_Return_Value
import decord
from decord import AudioReader, cpu, gpu
from decord.bridge import bridge_out
from utils.wavfile import WavFile
from utils.mp4file import Mp4File
decord.bridge._GLOBAL_BRIDGE_TYPE = 'torch'

folder_list = [
    "220918_남정희_김우진",
    "220918_남정희_백보경",
    "220918_남정희_손석규",
    "220918_남정희_정정연",
    "220918_남정희_차지수",
    "220920_강명진_오주현",
    "220920_강명진_윤수진",
    "220920_강명진_정은영",
    "220920_강명진_조주현",
    "220922_강명진_오은숙",
    "220922_강명진_정준호",
    "220922_강주영_강태랑",
    "220922_강주영_김은영",
    "220922_강주영_이준혁",
    "220922_강주영_정유림",
    "220922_강주영_최보규",
    "220925_남정희_김민석",
    "220925_남정희_박종길",
    "220925_남정희_서지원",
    "220925_남정희_서혜연",
    "220925_남정희_이주왕",
    "220925_남정희_장은태",
    "220925_남정희_한성민",
    "220925_남정희_허세민",
    "220929_강명진_김민수",
    "220929_강명진_김정현",
    "220929_강명진_류호정",
    "220929_강명진_유채이",
    "220929_강주영_김영미",
    "220929_강주영_류서영",
    "220929_강주영_송선희",
    "220929_강주영_임지윤",
    "221006_윤지선_박일용",
    "221006_윤지선_안수진",
    "221006_윤지선_용금여",
    "221006_윤지선_임현숙",
    "221006_윤지선_조영현",
    "221006_윤지선_채원석",
    "221006_윤지선_최주희",
    "220918_남정희_김지수",
]     

class ETRI_2022_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2022_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2022_whole")
        if os.path.isdir(self.path) == False:
            print("Copy etri2022_whole.zip")
            import shutil
            import zipfile
            shutil.copy("/data/datasets/etri2022_whole.zip", path)
            zipfile.ZipFile(f"{path}/etri2022_whole.zip").extractall(path)
            shutil.rmtree(f"{path}/etri2022_whole.zip", ignore_errors=True)
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "annotation.tsv"), sep='\t', index_col=0)
        # self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)
        # print(self.dataframe)
        
        self.audios = {}
        for filename in self.dataframe['folder'].unique():
            try:
                self.audios[filename +"_counselor"] = WavFile(os.path.join(self.path, "audio", f"{filename}_counselor.wav"))
                self.audios[filename +"_client"] = WavFile(os.path.join(self.path, "audio", f"{filename}_client.wav"))
            except:
                print(f"{filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != filename]
        print(f"ETRI 2023 Dataset {self.train} {self.balanced} :")
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
    
    def __get_audio__(self, name, start, end):
        audio = self.audios[name][start:end]
        audio = torch.tensor(audio)
        audio = torchaudio.transforms.Resample(16000, 16000)(audio)
        if audio.shape[1] != int(self.length * 16000):
            audio = F.pad(audio, (0, int(16000 * self.length) - audio.size(1)), "constant", 0)
        return audio
    
    def __getitem__(self, index):
        ret = {}

        item = self.dataframe.iloc[index]
        sample_rate = self.audios[item['folder']+'_'+item['role']].sample_rate
        trans = item['transcript']
        target_trans = item['back']
        time = int(item['bc_start']*sample_rate)
        length = int(self.length*sample_rate)
        predict_length = int(self.predict_length*sample_rate)

        audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+item['role'], time-length, time), name="audio"); audio.start()
        if item['BC'] == 0:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), time, time+predict_length), name="target_audio"); target_audio.start()
        else:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), time, time+predict_length), name="target_audio"); target_audio.start()
        trans = Thread_With_Return_Value(target=self.tokenizer, args=(trans,), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(target=self.tokenizer, args=(target_trans,), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        # trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()
        # target_trans = self.tokenizer(target_trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['label'] = item['BC']
        ret['text'] = trans.join()['input_ids'].squeeze()
        ret['target_text'] = target_trans.join()['input_ids'].squeeze()

        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_2023_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2023_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2023_whole")
        if os.path.isdir(self.path) == False:
            print("Copy etri2023_whole.zip")
            import shutil
            import zipfile
            
            shutil.copy("/data/datasets/etri2023_whole.zip", path)
            zipfile.ZipFile(f"{path}/etri2023_whole.zip").extractall(path)
            shutil.rmtree(f"{path}/etri2023_whole.zip", ignore_errors=True)
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "annotation.tsv"), sep='\t', index_col=0)
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)

        self.audios = {}
        for filename in self.dataframe['folder'].unique():
            try:
                self.audios[filename +"_counselor"] = WavFile(os.path.join(self.path, "audio", f"{filename}_counselor.wav"))
                self.audios[filename +"_client"] = WavFile(os.path.join(self.path, "audio", f"{filename}_client.wav"))
            except:
                print(f"{filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != filename]

        print(f"ETRI 2023 Dataset {self.train} {self.balanced} :")
        print(self.dataframe['BC'].value_counts().sort_index())
    
    def __len__(self):
        return len(self.dataframe)
    
    def __get_audio__(self, name, start, end):
        audio = self.audios[name][start:end]
        audio = torch.tensor(audio)
        audio = torchaudio.transforms.Resample(16000, 16000)(audio)
        if audio.shape[1] != int(self.length * 16000):
            audio = F.pad(audio, (0, int(16000 * self.length) - audio.size(1)), "constant", 0)
        return audio
    
    def __getitem__(self, index):
        ret = {}

        item = self.dataframe.iloc[index]
        sample_rate = self.audios[item['folder']+'_'+item['role']].sample_rate
        trans = item['transcript']
        target_trans = item['back']
        time = int(item['bc_start']*sample_rate)
        length = int(self.length*sample_rate)
        predict_length = int(self.predict_length*sample_rate)

        audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+item['role'], time-length, time), name="audio"); audio.start()
        if item['BC'] == 0:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), time, time+predict_length), name="target_audio"); target_audio.start()
        else:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), time, time+predict_length), name="target_audio"); target_audio.start()
        trans = Thread_With_Return_Value(target=self.tokenizer, args=(trans,), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(target=self.tokenizer, args=(target_trans,), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        # trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()
        # target_trans = self.tokenizer(target_trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['label'] = item['BC']
        ret['text'] = trans.join()['input_ids'].squeeze()
        ret['target_text'] = target_trans.join()['input_ids'].squeeze()

        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_All_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Dataset(path, tokenizer, train, balanced, length, predict_length)
        self.dataset_2023 = ETRI_2023_Dataset(path, tokenizer, train, balanced, length, predict_length)

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()
    
class ETRI_2022_Video_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2022_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2022_whole")
        if os.path.isdir(self.path) == False:
            print("Copy etri2022_whole.zip")
            subprocess.run(["cp", "/data/datasets/etri2022_whole.zip", path])
            subprocess.run(["unzip", f"{path}/etri2022_whole.zip", "-d", path])
            subprocess.run(["rm", "-rf", f"{path}/etri2022_whole.zip"])
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "annotation.tsv"), sep='\t', index_col=0)
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)

        self.videos = {}
        for filename in self.dataframe['folder'].unique():
            try:
                self.videos[filename +"_counselor"] = Mp4File(os.path.join(self.path, "video", f"{filename}_counselor.mp4"))
                self.videos[filename +"_client"] = Mp4File(os.path.join(self.path, "video", f"{filename}_client.mp4"))
            except:
                print(f"{filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != filename]
        self.audios = {}
        for filename in self.dataframe['folder'].unique():
            try:
                self.audios[filename +"_counselor"] = WavFile(os.path.join(self.path, "audio", f"{filename}_counselor.wav"))
                self.audios[filename +"_client"] = WavFile(os.path.join(self.path, "audio", f"{filename}_client.wav"))
            except:
                print(f"{filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != filename]

        print(f"ETRI 2022 Dataset {self.train} {self.balanced} :")
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
    
    def __get_audio__(self, name, start, end):
        audio = self.audios[name][start:end]
        audio = torch.tensor(audio)
        audio = torchaudio.transforms.Resample(16000, 16000)(audio)
        if audio.shape[1] != int(self.length * 16000):
            audio = F.pad(audio, (0, int(16000 * self.length) - audio.size(1)), "constant", 0)
        return audio
    
    def __get_video__(self, name, start, end):
        video = self.videos[name][tuple([int(start+((end-start)*i)/16) for i in range(16)])]
        video = torch.tensor(video)
        video = video.permute(0, 3, 1, 2)
        video = F.interpolate(video, (224, 224), mode='bilinear')
        return video
    
    def __getitem__(self, index):
        ret = {}

        item = self.dataframe.iloc[index]
        sample_rate = self.audios[item['folder']+'_'+item['role']].sample_rate

        a_time = int(item['bc_start']*sample_rate)
        a_length = int(self.length*sample_rate)
        a_predict_length = int(self.predict_length*sample_rate)

        v_time = int(item['bc_start']*self.videos[item['folder']+'_'+item['role']].frame_rate)
        v_length = int(self.length*self.videos[item['folder']+'_'+item['role']].frame_rate)
        v_predict_length = int(self.predict_length*self.videos[item['folder']+'_'+item['role']].frame_rate)

        audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+item['role'], a_time-a_length, a_time), name="audio"); audio.start()
        video = Thread_With_Return_Value(target=self.__get_video__, args=(item['folder']+'_'+item['role'], v_time-v_length, v_time), name="video"); video.start()
        if item['BC'] == 0:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), a_time, a_time+a_predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(target=self.__get_video__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), v_time, v_time+v_predict_length), name="target_video"); target_video.start()
        else:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), a_time, a_time+a_predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(target=self.__get_video__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), v_time, v_time+v_predict_length), name="target_video"); target_video.start()
        trans = Thread_With_Return_Value(target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['video'] = video.join()
        ret['target_video'] = target_video.join()
        ret['text'] = trans.join()['input_ids'].squeeze()
        ret['target_text'] = target_trans.join()['input_ids'].squeeze()
        ret['label'] = item['BC']

        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_2023_Video_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_2023_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "etri2023_whole")
        if os.path.isdir(self.path) == False:
            print("Copy etri2023_whole.zip")
            subprocess.run(["cp", "/data/datasets/etri2023_whole.zip", path])
            subprocess.run(["unzip", f"{path}/etri2023_whole.zip", "-d", path])
            subprocess.run(["rm", "-rf", f"{path}/etri2023_whole.zip"])
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        if self.balanced and not self.train:
            logging.warning("The balance is only for training dataset")

        self.dataframe = pd.read_csv(os.path.join(self.path, "annotation.tsv"), sep='\t', index_col=0)
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)

        self.videos = {}
        for filename in self.dataframe['folder'].unique():
            try:
                self.videos[filename +"_counselor"] = Mp4File(os.path.join(self.path, "video", f"{filename}_counselor.mp4"))
                self.videos[filename +"_client"] = Mp4File(os.path.join(self.path, "video", f"{filename}_client.mp4"))
            except:
                print(f"{filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != filename]
        self.audios = {}
        for filename in self.dataframe['folder'].unique():
            try:
                self.audios[filename +"_counselor"] = WavFile(os.path.join(self.path, "audio", f"{filename}_counselor.wav"))
                self.audios[filename +"_client"] = WavFile(os.path.join(self.path, "audio", f"{filename}_client.wav"))
            except:
                print(f"{filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != filename]

        print(f"ETRI 2023 Dataset {self.train} {self.balanced} :")
        print(self.dataframe['BC'].value_counts().sort_index())

    def __len__(self):
        return len(self.dataframe)
        
    def __get_audio__(self, name, start, end):
        audio = self.audios[name][start:end]
        audio = torch.tensor(audio)
        audio = torchaudio.transforms.Resample(16000, 16000)(audio)
        if audio.shape[1] != int(self.length * 16000):
            audio = F.pad(audio, (0, int(16000 * self.length) - audio.size(1)), "constant", 0)
        return audio
    
    def __get_video__(self, name, start, end):
        video = self.videos[name][tuple([int(start+((end-start)*i)/16) for i in range(16)])]
        video = torch.tensor(video)
        video = video.permute(0, 3, 1, 2)
        video = F.interpolate(video, (224, 224), mode='bilinear')
        return video
    
    def __getitem__(self, index):
        ret = {}

        item = self.dataframe.iloc[index]
        sample_rate = self.audios[item['folder']+'_'+item['role']].sample_rate
        a_time = int(item['bc_start']*sample_rate)
        a_length = int(self.length*sample_rate)
        a_predict_length = int(self.predict_length*sample_rate)

        v_time = int(item['bc_start']*self.videos[item['folder']+'_'+item['role']].frame_rate)
        v_length = int(self.length*self.videos[item['folder']+'_'+item['role']].frame_rate)
        v_predict_length = int(self.predict_length*self.videos[item['folder']+'_'+item['role']].frame_rate)

        audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+item['role'], a_time-a_length, a_time), name="audio"); audio.start()
        video = Thread_With_Return_Value(target=self.__get_video__, args=(item['folder']+'_'+item['role'], v_time-v_length, v_time), name="video"); video.start()
        if item['BC'] == 0:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), a_time, a_time+a_predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(target=self.__get_video__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), v_time, v_time+v_predict_length), name="target_video"); target_video.start()
        else:
            target_audio = Thread_With_Return_Value(target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), a_time, a_time+a_predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(target=self.__get_video__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), v_time, v_time+v_predict_length), name="target_video"); target_video.start()
        trans = Thread_With_Return_Value(target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['video'] = video.join()
        ret['target_video'] = target_video.join()
        ret['text'] = trans.join()['input_ids'].squeeze()
        ret['target_text'] = target_trans.join()['input_ids'].squeeze()
        ret['label'] = item['BC']

        return ret
    
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_All_Video_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Video_Dataset(path, tokenizer, train, balanced, length, predict_length)
        self.dataset_2023 = ETRI_2023_Video_Dataset(path, tokenizer, train, balanced, length, predict_length)

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()