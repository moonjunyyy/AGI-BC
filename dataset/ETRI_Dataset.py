import os
import gc
import subprocess
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from dataclasses import dataclass
from utils.wavfile import WavFile
from utils.mp4file import Mp4File
from M00NNY_Utils.threads import Thread_With_Return_Value

class ETRI_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.path = path
        self.train = train
        self.length = length
        self.predict_length = predict_length
        self.balanced = balanced
        self.sample_rate = sample_rate
        self.num_frames = num_frames

    def _load_data(self, zip_file, tsv_file):
        if os.path.isdir(self.path) == False:
            print(f"Copy {zip_file}")
            subprocess.run(["cp", f"/data/datasets/{zip_file}", self.path])
            subprocess.run(["unzip", f"{self.path}/{zip_file}", "-d", self.path])
            subprocess.run(["rm", "-rf", f"{self.path}/{zip_file}"])
        self.path = os.path.join(self.path, zip_file.split(".")[0])
        self.dataframe = pd.read_csv(os.path.join(self.path, tsv_file), sep='\t', index_col=0)
        trainset = self.dataframe.sample(frac=0.8, random_state=42)
        if self.train:
            self.dataframe = trainset
        else:
            self.dataframe = self.dataframe.drop(trainset.index)
        if self.balanced:
            bc_num = self.dataframe['BC'].value_counts().sort_index().to_numpy()
            bc_num = min(bc_num[0], bc_num[1:].sum())
            self.dataframe = pd.concat([self.dataframe[self.dataframe['BC'] == 0].sample(bc_num, replace=False, random_state=42)] + [self.dataframe[self.dataframe['BC'] != 0].sample(bc_num, replace=False, random_state=42)])

    def _load_audio(self):
        self.audios = {}
        def _load(filename): return WavFile(os.path.join(self.path, "audio", f"{filename}.wav"))
        counselor = {}
        client = {}
        for idx, filename in enumerate(self.dataframe['folder'].unique()):
            counselor[filename] = Thread_With_Return_Value(daemon=True, target=_load, args=(f"{filename}_counselor",))
            client[filename] = Thread_With_Return_Value(daemon=True, target=_load, args=(f"{filename}_client",))
            counselor[filename].start()
            client[filename].start()
        for key in set(list(counselor.keys()) + list(client.keys())):
            try:
                _counselor = counselor[key].join()
                _client = client[key].join()
                if _counselor is None or _client is None: raise Exception
                self.audios[key+"_counselor"] = _counselor
                self.audios[key+"_client"] = _client
            except Exception as e:
                print(e)
                print(f"Audio : {key} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != key]
                counselor.pop(key, None)
                client.pop(key, None)

    def _load_video(self):
        self.videos = {}
        def _load(filename): return Mp4File(os.path.join(self.path, "video", f"{filename}.mp4"))
        counselor = {}
        client = {}
        for idx, filename in enumerate(self.dataframe['folder'].unique()):
            counselor[filename] = Thread_With_Return_Value(daemon=True, target=_load, args=(f"{filename}_counselor",))
            client[filename] = Thread_With_Return_Value(daemon=True, target=_load, args=(f"{filename}_client",))
            counselor[filename].start()
            client[filename].start()
        for key in set(list(counselor.keys()) + list(client.keys())):
            try:
                _counselor = counselor[key].join()
                _client = client[key].join()
                if _counselor is None or _client is None: raise Exception
                self.videos[key+"_counselor"] = _counselor
                self.videos[key+"_client"] = _client
            except Exception as e:
                print(f"Video : {filename} is not found")
                self.dataframe = self.dataframe[self.dataframe['folder'] != key]
                counselor.pop(key, None)
                client.pop(key, None)
        del counselor, client

    def __len__(self):
        return len(self.dataframe)
    
    def __get_audio__(self, name, time, length):
        if length < 0: start_time = time + length; end_time = time;
        else: start_time = time; end_time = time + length;
        start_frame = int(start_time * self.audios[name].sample_rate)
        end_frame = int(end_time * self.audios[name].sample_rate)
        if start_frame < 0: start_frame = 0
        if end_frame > len(self.audios[name]): end_frame = len(self.audios[name])
        audio = self.audios[name][start_frame:end_frame]
        audio = torch.tensor(audio)
        audio = torchaudio.transforms.Resample(self.audios[name].sample_rate, self.sample_rate)(audio)
        length = length if length > 0 else -length
        audio = audio[:, -int(self.sample_rate * length):]
        if audio.shape[1] < int(self.sample_rate * length):
            audio = F.pad(audio, (int(self.sample_rate * length) - audio.shape[1], 0), "constant", 0)
        return audio
    
    def __get_video__(self, name, time, length):
        if length < 0: start_time = time + length; end_time = time;
        else: start_time = time; end_time = time + length;
        start_frame = int(start_time * self.videos[name].frame_rate)
        end_frame = int(end_time * self.videos[name].frame_rate)
        if start_frame <= 0: start_frame = 1
        if end_frame >= len(self.videos[name]): end_frame = len(self.videos[name]) - 1
        video = self.videos[name][tuple([int(start_frame+((end_frame-start_frame)*i)/self.num_frames) for i in range(self.num_frames)])]
        video = torch.tensor(video)
        video = video.permute(0, 3, 1, 2)
        video = F.interpolate(video, (224, 224), mode='bilinear')
        return video

    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().sort_index().to_numpy()
    
class ETRI_2022_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2022_Dataset...")
        self._load_data("etri2022_whole.zip", "annotation.tsv")
        self._load_audio()
        if verbose:
            print(f"ETRI_2022_Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.dataframe['BC'].value_counts().sort_index())
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")
    
    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="audio"); audio.start()        
        target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_audio"); target_audio.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['label'] = item['BC']
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        target_trans = target_trans.join()
        ret['target_text'] = target_trans['input_ids'].squeeze()
        ret['target_text_attention_mask'] = target_trans['attention_mask'].squeeze()
        ret['target_text_token_type_ids'] = target_trans['token_type_ids'].squeeze()
        return ret
    
class ETRI_2023_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2023_Dataset...")
        self._load_data("etri2023_whole.zip", "annotation.tsv")
        self._load_audio()
        if verbose:
            print(f"ETRI 2023 Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.dataframe['BC'].value_counts().sort_index())
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")
    
    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="audio"); audio.start()
        target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_audio"); target_audio.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['label'] = item['BC']
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        target_trans = target_trans.join()
        ret['target_text'] = target_trans['input_ids'].squeeze()
        ret['target_text_attention_mask'] = target_trans['attention_mask'].squeeze()
        ret['target_text_token_type_ids'] = target_trans['token_type_ids'].squeeze()
        return ret
    
class ETRI_All_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        self.dataset_2023 = ETRI_2023_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        if verbose:
            print(f"ETRI ALL Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.get_sample_in_class())
            total_len = len(self.dataset_2022) * self.length + len(self.dataset_2023) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()
    
class ETRI_2022_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2022_Dataset...")
        self._load_data("etri2022_whole.zip", "annotation.tsv")
        self._load_audio()
        self._load_video()
        if verbose:
            print(f"ETRI_2022_Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.dataframe['BC'].value_counts().sort_index())
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")
    
    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        if item['BC'] == 0:
            audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start']-2., -self.length), name="audio"); audio.start()
            video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['bc_start']-2., -self.length), name="video"); video.start()
            target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), item['bc_start']-2., self.predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), item['bc_start']-2., self.predict_length), name="target_video"); target_video.start()
        else:
            audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="audio"); audio.start()
            video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="video"); video.start()
            target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_video"); target_video.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['video'] = video.join()
        ret['target_video'] = target_video.join()
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        target_trans = target_trans.join()
        ret['target_text'] = target_trans['input_ids'].squeeze()
        ret['target_text_attention_mask'] = target_trans['attention_mask'].squeeze()
        ret['target_text_token_type_ids'] = target_trans['token_type_ids'].squeeze()
        ret['label'] = item['BC']
        return ret
    
class ETRI_2023_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2023_Dataset...")
        self._load_data("etri2023_whole.zip", "annotation.tsv")
        self._load_audio()
        self._load_video()
        if verbose:
            print(f"ETRI 2023 Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.dataframe['BC'].value_counts().sort_index())
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        if item['BC'] == 0:
            audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start']-2., -self.length), name="audio"); audio.start()
            video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['bc_start']-2., -self.length), name="video"); video.start()
            target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), item['bc_start']-2., self.predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+ ('counselor' if item['role']=='counselor' else 'client'), item['bc_start']-2., self.predict_length), name="target_video"); target_video.start()
        else:
            audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="audio"); audio.start()
            video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="video"); video.start()
            target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_audio"); target_audio.start()
            target_video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_video"); target_video.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['video'] = video.join()
        ret['target_video'] = target_video.join()
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        target_trans = target_trans.join()
        ret['target_text'] = target_trans['input_ids'].squeeze()
        ret['target_text_attention_mask'] = target_trans['attention_mask'].squeeze()
        ret['target_text_token_type_ids'] = target_trans['token_type_ids'].squeeze()
        ret['label'] = item['BC']
        return ret
    
class ETRI_All_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        self.dataset_2023 = ETRI_2023_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        if verbose:
            print(f"ETRI ALL Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.get_sample_in_class())
            total_len = len(self.dataset_2022) * self.length + len(self.dataset_2023) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()
    
class ETRI_2022_TT_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2022_Dataset...")
        self._load_data("etri2022_whole.zip", "annotation_tt.tsv")
        self._load_audio()
        self._load_video()
        if verbose:
            print(f"ETRI_2022_Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.dataframe['BC'].value_counts().sort_index())
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="audio"); audio.start()
        video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="video"); video.start()
        target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_audio"); target_audio.start()
        target_video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_video"); target_video.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['video'] = video.join()
        ret['target_video'] = target_video.join()
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        target_trans = target_trans.join()
        ret['target_text'] = target_trans['input_ids'].squeeze()
        ret['target_text_attention_mask'] = target_trans['attention_mask'].squeeze()
        ret['target_text_token_type_ids'] = target_trans['token_type_ids'].squeeze()
        ret['label'] = item['BC']
        return ret
    
class ETRI_2023_TT_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2023_Dataset...")
        self._load_data("etri2023_whole.zip", "annotation_tt.tsv")
        self._load_audio()
        self._load_video()
        if verbose:
            print(f"ETRI 2023 Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.dataframe['BC'].value_counts().sort_index())
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")
    
    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="audio"); audio.start()
        video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['bc_start'], -self.length), name="video"); video.start()
        target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_audio"); target_audio.start()
        target_video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+ ('client' if item['role']=='counselor' else 'counselor'), item['bc_start'], self.predict_length), name="target_video"); target_video.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['back'],), kwargs={'padding':'max_length', 'max_length':5, 'truncation':True, 'return_tensors':"pt"}); target_trans.start()

        ret['audio'] = audio.join()
        ret['target_audio'] = target_audio.join()
        ret['video'] = video.join()
        ret['target_video'] = target_video.join()
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        target_trans = target_trans.join()
        ret['target_text'] = target_trans['input_ids'].squeeze()
        ret['target_text_attention_mask'] = target_trans['attention_mask'].squeeze()
        ret['target_text_token_type_ids'] = target_trans['token_type_ids'].squeeze()
        ret['label'] = item['BC']
        return ret
    
class ETRI_All_TT_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=True, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_TT_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        self.dataset_2023 = ETRI_2023_TT_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        if verbose:
            print(f"ETRI ALL Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.get_sample_in_class())
            total_len = len(self.dataset_2022) * self.length + len(self.dataset_2023) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]
        
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()

@dataclass
class ETRI_Threashold:
    thresholds = {
        "220918_남정희_손석규":	1979.022,
        "220922_강주영_강태랑":	1868.511,
        "220929_강명진_김정현":	1953.196,
        "221006_윤지선_조영현":	1538.848,
        "1ST":	2643.797,
        "6ST":	2254.096,
        "11ST":	2093.002,
        "16ST":	2488.419,
        }
    
class ETRI_2022_Dialog_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=False, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose=True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2022_Dataset...")
        self._load_data("etri2022_whole.zip", "words.tsv")

        # 필요한 변수 초기화
        transcripts = []
        folder_list = []
        role_list = []
        end_times = []
        role_change = []

        # 5단어를 담을 리스트와 현재 role을 추적하는 변수
        current_transcript = []
        current_role = None
        current_folder = None

        # DataFrame을 순차적으로 탐색
        for i, row in self.dataframe.iterrows():
            if row['folder'] in ETRI_Threashold.thresholds: continue
            # 현재 row의 역할과 비교하여 현재 역할이 동일하면 계속 추가
            if current_role is None or row['role'] == current_role and row['folder'] == current_folder:
                current_transcript.append(row['transcript'])
                current_role = row['role']
                current_folder = row['folder']
                current_end_time = row['end']
            else:
                # 5단어씩 슬라이딩 윈도우로 끊어서 저장
                for j in range(0, len(current_transcript) - 4 if len(current_transcript) > 4 else 1):
                    transcripts.append(' '.join(current_transcript[j:j + 5]))
                    folder_list.append(current_folder)
                    role_list.append(current_role)
                    end_times.append(current_end_time)
                    role_change.append(False)
                role_change[-1] = True
                
                # 새로운 역할로 초기화
                current_transcript = [row['transcript']]
                current_role = row['role']
                current_folder = row['folder']
                current_end_time = row['end']

        # 마지막으로 남은 transcript 처리 (5단어씩 슬라이딩 윈도우로 끊어서 저장)
        if current_transcript:
            for j in range(0, len(current_transcript), 5):
                transcripts.append(' '.join(current_transcript[j:j + 5]))
                folder_list.append(current_folder)
                role_list.append(current_role)
                end_times.append(current_end_time)
                role_change.append(False)
            role_change[-1] = True

        # 최종 DataFrame 생성
        self.dataframe = pd.DataFrame({
            'transcript': transcripts,
            'folder': folder_list,
            'role': role_list,
            'role_change': role_change,
            'time': end_times
        })

        self._load_audio()
        self._load_video()
        if verbose:
            print(f"ETRI_2022_Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['time'], -self.length), name="audio"); audio.start()
        video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['time'], -self.length), name="video"); video.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['time'], self.predict_length), name="target_audio"); target_audio.start()

        ret['audio'] = audio.join()
        ret['video'] = video.join()
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        ret['target_audio'] = target_audio.join()
        return ret

class ETRI_2023_Dialog_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=False, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose = True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_2023_Dataset...")
        self._load_data("etri2023_whole.zip", "words.tsv")

        # 필요한 변수 초기화
        transcripts = []
        folder_list = []
        role_list = []
        end_times = []
        role_change = []

        # 5단어를 담을 리스트와 현재 role을 추적하는 변수
        current_transcript = []
        current_role = None
        current_folder = None

        # DataFrame을 순차적으로 탐색
        for i, row in self.dataframe.iterrows():
            if row['folder'] in ETRI_Threashold.thresholds: continue
            # 현재 row의 역할과 비교하여 현재 역할이 동일하면 계속 추가
            if current_role is None or row['role'] == current_role and row['folder'] == current_folder:
                current_transcript.append(row['transcript'])
                current_role = row['role']
                current_folder = row['folder']
                current_end_time = row['end']
            else:
                # 5단어씩 슬라이딩 윈도우로 끊어서 저장
                for j in range(0, len(current_transcript) - 4 if len(current_transcript) > 4 else 1):
                    transcripts.append(' '.join(current_transcript[j:j + 5]))
                    folder_list.append(current_folder)
                    role_list.append(current_role)
                    end_times.append(current_end_time)
                    role_change.append(False)
                role_change[-1] = True
                
                # 새로운 역할로 초기화
                current_transcript = [row['transcript']]
                current_role = row['role']
                current_folder = row['folder']
                current_end_time = row['end']

        # 마지막으로 남은 transcript 처리 (5단어씩 슬라이딩 윈도우로 끊어서 저장)
        if current_transcript:
            for j in range(0, len(current_transcript), 5):
                transcripts.append(' '.join(current_transcript[j:j + 5]))
                folder_list.append(current_folder)
                role_list.append(current_role)
                end_times.append(current_end_time)
                role_change.append(False)
            role_change[-1] = True

        # 최종 DataFrame 생성
        self.dataframe = pd.DataFrame({
            'transcript': transcripts,
            'folder': folder_list,
            'role': role_list,
            'role_change': role_change,
            'time': end_times
        })

        self._load_audio()
        self._load_video()

        if verbose:
            print(f"ETRI_2023_Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            total_len = len(self.dataframe) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __getitem__(self, index):
        ret = {}
        item = self.dataframe.iloc[index]
        audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['time'], -self.length), name="audio"); audio.start()
        video = Thread_With_Return_Value(daemon=True, target=self.__get_video__, args=(item['folder']+'_'+item['role'], item['time'], -self.length), name="video"); video.start()
        trans = Thread_With_Return_Value(daemon=True, target=self.tokenizer, args=(item['transcript'],), kwargs={'padding':'max_length', 'max_length':20, 'truncation':True, 'return_tensors':"pt"}); trans.start()
        target_audio = Thread_With_Return_Value(daemon=True, target=self.__get_audio__, args=(item['folder']+'_'+item['role'], item['time'], self.predict_length), name="target_audio"); target_audio.start()
        
        ret['audio'] = audio.join()
        ret['video'] = video.join()
        trans = trans.join()
        ret['text'] = trans['input_ids'].squeeze()
        ret['text_attention_mask'] = trans['attention_mask'].squeeze()
        ret['text_token_type_ids'] = trans['token_type_ids'].squeeze()
        ret['target_audio'] = target_audio.join()
        return ret
    
class ETRI_All_Dialog_Video_Dataset(ETRI_Dataset):
    def __init__(self, path, tokenizer, train = False, balanced=False, length :float = 1.5, predict_length:float = 0.5, sample_rate = 16000, num_frames = 16, verbose = True) -> None:
        super().__init__(path=path, tokenizer=tokenizer, train=train, balanced=balanced, length=length, predict_length=predict_length, sample_rate=sample_rate, num_frames=num_frames)
        if verbose: print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Dialog_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, verbose=False)
        self.dataset_2023 = ETRI_2023_Dialog_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, verbose=False)
        if verbose:
            print(f"ETRI ALL Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            total_len = len(self.dataset_2022) * self.length + len(self.dataset_2023) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")

    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    
    def __getitem__(self, index):
        if index < len(self.dataset_2022):
            return self.dataset_2022[index]
        else:
            return self.dataset_2023[index - len(self.dataset_2022)]

class ETRI_Threshold_Dataset(ETRI_Dataset):
    def _load_data(self, zip_file, tsv_file):
        if os.path.isdir(self.path) == False:
            print(f"Copy {zip_file}")
            subprocess.run(["cp", f"/data/datasets/{zip_file}", self.path])
            subprocess.run(["unzip", f"{self.path}/{zip_file}", "-d", self.path])
            subprocess.run(["rm", "-rf", f"{self.path}/{zip_file}"])
        self.path = os.path.join(self.path, zip_file.split(".")[0])
        self.dataframe = pd.read_csv(os.path.join(self.path, tsv_file), sep='\t', index_col=0)

        mask_in_keys = self.dataframe['folder'].isin(ETRI_Threashold.thresholds.keys())
        thresholds = self.dataframe['folder'].map(ETRI_Threashold.thresholds)
        # thresholds = thresholds.fillna(0)

        mask_to_keep = (~mask_in_keys) | (self.dataframe['bc_start'] < thresholds)
        trainset = self.dataframe[mask_to_keep]
        if self.train: self.dataframe = trainset
        else: self.dataframe = self.dataframe.drop(trainset.index)

        if self.balanced:
            bc_num = self.dataframe['BC'].value_counts().sort_index().to_numpy()
            bc_num = min(bc_num[0], bc_num[1:].sum())
            self.dataframe = pd.concat([self.dataframe[self.dataframe['BC'] == 0].sample(bc_num, replace=False, random_state=42)] + [self.dataframe[self.dataframe['BC'] != 0].sample(bc_num, replace=False, random_state=42)])

class ETRI_2022_Threshold_Video_Dataset(ETRI_Threshold_Dataset, ETRI_2022_Video_Dataset): pass
class ETRI_2023_Threshold_Video_Dataset(ETRI_Threshold_Dataset, ETRI_2023_Video_Dataset): pass
class ETRI_All_Threshold_Video_Dataset(ETRI_Threshold_Dataset):
    def __init__(self, path, tokenizer, train=False, balanced=True, length = 1.5, predict_length = 0.5, sample_rate=16000, num_frames=16, verbose=True) -> None:
        super().__init__(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames)
        if verbose: print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_Threshold_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        self.dataset_2023 = ETRI_2023_Threshold_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        if verbose:
            print(f"ETRI ALL Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.get_sample_in_class())
            total_len = len(self.dataset_2022) * self.length + len(self.dataset_2023) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")
    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    def __getitem__(self, index):
        if index < len(self.dataset_2022): return self.dataset_2022[index]
        else: return self.dataset_2023[index - len(self.dataset_2022)]
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()
    
class ETRI_2022_TT_Threshold_Video_Dataset(ETRI_Threshold_Dataset, ETRI_2022_TT_Video_Dataset): pass
class ETRI_2023_TT_Threshold_Video_Dataset(ETRI_Threshold_Dataset, ETRI_2023_TT_Video_Dataset): pass
class ETRI_All_TT_Threshold_Video_Dataset(ETRI_Threshold_Dataset):
    def __init__(self, path, tokenizer, train=False, balanced=True, length = 1.5, predict_length = 0.5, sample_rate=16000, num_frames=16, verbose=True) -> None:
        super().__init__(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames)
        if verbose: print("Load ETRI_Corpus_Dataset...")
        self.dataset_2022 = ETRI_2022_TT_Threshold_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        self.dataset_2023 = ETRI_2023_TT_Threshold_Video_Dataset(path, tokenizer, train, balanced, length, predict_length, sample_rate, num_frames, False)
        if verbose:
            print(f"ETRI ALL Dataset {'Train' if self.train else 'Test'} {'Balanced' if self.balanced else 'Imbalanced'}")
            print(self.get_sample_in_class())
            total_len = len(self.dataset_2022) * self.length + len(self.dataset_2023) * self.length
            print(f"Total Sample Length : {int(total_len // 3600)}:{int(total_len % 3600 // 60):02d}:{total_len % 60:02.2f}s")
    def __len__(self):
        return len(self.dataset_2022) + len(self.dataset_2023)
    def __getitem__(self, index):
        if index < len(self.dataset_2022): return self.dataset_2022[index]
        else: return self.dataset_2023[index - len(self.dataset_2022)]
    def get_sample_in_class(self):
        return self.dataset_2022.get_sample_in_class() + self.dataset_2023.get_sample_in_class()