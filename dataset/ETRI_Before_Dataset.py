import os
import gc
import shutil
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import Dataset
from typing import Callable
from util.knusl import KnuSL
import math
from decord import VideoReader, cpu
import numpy as np
from konlpy.tag import Okt

class ETRI_Word_Dataset(Dataset):
    def __init__(self, path, tokenizer, transform : Callable=None, length :int = 5) -> None:
        super().__init__()
        # self.path = os.path.join(path, "ETRI_Backchannel_Corpus_2022")
        print("Load ETRI_Word_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "ETRI_Word")
        # os.system(f"rm -r {self.path}/")
        if os.path.isdir(self.path) == False:
            os.system(f"cp /data/datasets/ETRI_Word.tar {path}/")
            os.system(f"chmod 777 {path}/ETRI_Word.tar")
            os.system(f"tar -xvf {path}/ETRI_Word.tar -C {path}")
            os.system(f"chmod -R 777  {self.path}/*")
            os.system(f"rm {path}/ETRI_Word.tar")
        self.length = length
        self.dataframe = pd.read_csv(os.path.join(self.path, "etri.tsv"), sep='\t', index_col=0)
        # self.dataframe.rename( columns={'Unnamed: 0':'filename'}, inplace=True )
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        # print(self.dataframe)
        # self.dataframe = self.dataframe[(self.dataframe['end']-self.dataframe['start'])>(self.length)]
        # self.dataframe = self.dataframe[~self.dataframe['folder'].isin([2,12,31])]
        
        print("Load ETRI_Word_Dataset Done!")
        print(f"Total {len(self.dataframe)}")
        print(self.dataframe)
        print("Check the file is exist or not...")

        bad_idx = []
        for path in self.dataframe['folder'].unique():
            print(path)
            try:
                audio_path = os.path.join(self.path, f"{path}", "client.wav")
                audio, sr = torchaudio.load(audio_path)
                if audio.numel() == 0:
                    print(f"audio cannot be loaded : {audio_path}")
                    raise
                del(audio)
                audio_path = os.path.join(self.path, f"{path}", "counselor.wav")
                audio, sr = torchaudio.load(audio_path)
                if audio.numel() == 0:
                    print(f"audio cannot be loaded : {audio_path}")
                    raise
                del(audio)
            except:
                bad_idx.append(path)
            gc.collect()


        # for idx, row in self.dataframe.iterrows():
        #     if idx in bad_idx:
        #         continue
        #     try:
        #         # path = os.path.join(self.path, "video", f"{idx}.mp4")
        #         # frames = self.load_video_decord(path)
        #         # if len(frames) == 0:
        #         #     print(f"video cannot be loaded : {path}")
        #         #     raise
                
        #         # del(frames)
        #         path = os.path.join(self.path, "audio", f"{idx}.wav")
        #         audio, sr = torchaudio.load(path)
        #         if audio.numel() == 0:
        #             print(f"audio cannot be loaded : {path}")
        #             raise
        #         del(audio)
        #     except:
        #         # print(f"Bad file: {idx}")
        #         bad_idx.append(idx)
        #     gc.collect()
        print(f"Bad file: {bad_idx}")
        #print number of the labels

        self.dataframe = self.dataframe[~self.dataframe['folder'].isin(bad_idx)]
        
        self.dataframe = self.dataframe[self.dataframe['end']-self.dataframe['start']>0]
        self.dataframe = self.dataframe[self.dataframe['end']-self.dataframe['start']<5]
        print(self.dataframe['class'].value_counts())
        
        self.transform = transform

        print(max(self.dataframe['end'] - self.dataframe['start']))
        print(min(self.dataframe['end'] - self.dataframe['start']))

        print(len(self.dataframe['folder'].unique()))
        
        self.length = 5

    def __len__(self):
        # return len(self.dataframe) - self.length
        return len(self.dataframe['folder'].unique())
    
    def __getitem__(self, index):

        dataframe = self.dataframe[self.dataframe['folder'] == index + 1]
        # print(index, " : ", len(dataframe))
        index = torch.randint(0, len(dataframe)-self.length, (1,)).item()

        # self.okt = Okt()
        ret = {}
        
        audio_list = []
        text_list = []
        label_list = []

        for l in range(self.length):
            item = self.dataframe.iloc[index + l]
            # print(item)
            idx = item['folder'] - 1
            role = item['role']
            trans = item['transcript']
            lable = item['class']
            start = item['start']
            end   = item['end']
            role = item['role']

            # path = os.path.join(self.path, "video", f"{str(idx)}.mp4")
            # frames = self.load_video_decord(path)

            path = os.path.join(self.path, f"{str(idx)}", f"{role}.wav")

            audio, sr = torchaudio.load(path)
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            sr = 16000
            audio = audio[:, int(start*sr):int(end*sr)]
            if audio.size(1) != int(sr * 5):
                audio = F.pad(audio, (0, int(sr * 5) - audio.size(1)), "constant", 0)
            if audio.size(0) != 1:
                audio = audio.sum(0, keepdim=True)

            trans = self.tokenizer(trans, padding='max_length', max_length=5, truncation=True, return_tensors="pt")['input_ids'].squeeze()

            audio_list.append(audio)
            text_list.append(trans)
            label_list.append(lable)
        
        audio = torch.stack(audio_list)
        trans = torch.stack(text_list)
        lable = torch.tensor(label_list)

        # if len(frames) != 8:
        #     print("!\t",len(frames), index, idx)
        # if audio.size(1) != 24000:
        #     print("!!\t", index, idx)
        # if len(trans) != 1:    
        #     print("!!!\t", index, idx)
        # if len(sentiment) != 5:
        #     print("!!!!\t", index, idx)
            
        # ret['frames'] = torch.tensor(frames)
        ret['audio'] = audio
        ret['label'] = lable
        ret['text'] = trans
        # ret['sentiment'] = sentiment
        return ret
        
    def load_video_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []
        
        keep_aspect_ratio = True
        try:
            if keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            raise
            return []
          
        #   if self.mode == 'test':
        #        all_index = []
        #        tick = len(vr) / float(self.num_segment)
        #        all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
        #                            [int(tick * x) for x in range(self.num_segment)]))
        #        while len(all_index) < (self.num_segment * self.test_num_segment):
        #             all_index.append(all_index[-1])
        #        all_index = list(np.sort(np.array(all_index))) 
        #        vr.seek(0)
        #        buffer = vr.get_batch(all_index).asnumpy()
        #        return buffer

        self.num_segment = 8
        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        
        # all_index = range(self.num_segment)
        # all_index = [i * average_duration for i in all_index] 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy().astype(np.float32)
        return buffer

        #  def __len__(self):
        #       if self.mode != 'test':
        #            return len(self.dataset_samples)
        #       else:
        #            return len(self.test_dataset)