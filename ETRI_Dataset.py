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
from knusl import KnuSL
import math
from decord import VideoReader, cpu
import numpy as np
from konlpy.tag import Okt

class ETRI_Corpus_Dataset(Dataset):
    def __init__(self, path, tokenizer, transform : Callable=None, length :float = 1.5) -> None:
        super().__init__()
        # self.path = os.path.join(path, "ETRI_Backchannel_Corpus_2022")
        print("Load ETRI_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "ETRI_Corpus_Clip")
        if not os.environ['SLURM_NODELIST'] in ['ariel-g1', 'ariel-g3', 'ariel-g5']:
            os.system(f"rm -r {self.path}/")
            os.makedirs(self.path, exist_ok=True)
            os.system(f"cp /data/datasets/ETRI_Corpus_Clip.tar {path}/")
            os.system(f"chmod 777 {path}/ETRI_Corpus_Clip.tar")
            os.system(f"tar -xvf {path}/ETRI_Corpus_Clip.tar -C {path}")
            os.system(f"chmod -R 777  {self.path}/*")
            os.system(f"rm {path}/ETRI_Corpus_Clip.tar")
        if os.path.isdir(self.path) == False:
            os.system(f"cp /data/datasets/ETRI_Corpus_Clip.tar {path}/")
            os.system(f"chmod 777 {path}/ETRI_Corpus_Clip.tar")
            os.system(f"tar -xvf {path}/ETRI_Corpus_Clip.tar -C {path}")
            os.system(f"chmod -R 777  {self.path}/*")
            os.system(f"rm {path}/ETRI_Corpus_Clip.tar")
        self.length = length
        self.dataframe = pd.read_csv(os.path.join(self.path, "etri.tsv"), sep='\t', index_col=0)
        # self.dataframe.rename( columns={'Unnamed: 0':'filename'}, inplace=True )
        self.dataframe = self.dataframe.assign(filename=range(len(self.dataframe)))
        # print(self.dataframe)
        # self.dataframe = self.dataframe[(self.dataframe['end']-self.dataframe['start'])>(self.length)]
        # self.dataframe = self.dataframe[~self.dataframe['folder'].isin([2,12,31])]
        # bad_idx = [1078, 1739, 2733, 2745, 3104, 3754, 5895, 6220, 6488, 6581, 7172, 15052, 15145, 15204, 19349, 19350, 20893, 20894, 20895, 20896, 20897, 20898, 20899, 20900, 20901, 20902, 20903, 20904, 20905, 20906, 20907, 20908, 20909, 20910, 20911, 20912, 20913, 20914, 20915, 20916, 20917, 20918, 20919, 20920, 20921, 20922, 20923, 20924, 20925, 20926, 20927, 20928, 20929, 20930, 20931, 20932, 20933, 20934, 20935, 20936, 20937, 20938, 20939, 20940, 20941, 20942, 20943, 20944, 20945, 20946, 20947, 20948, 20949, 20950, 20951, 20952, 20953, 20954, 20955, 20956, 20957, 20958, 20959, 20960, 20961, 20962, 20963, 20964, 20965, 20966, 20967, 20968, 20969, 20970, 20971, 20972, 20973, 20974, 20975, 20976, 20977, 20978, 20979, 20980, 20981, 20982, 20983, 20984, 20985, 20986, 20987, 20988, 20989, 20990, 20991, 20992, 20993, 20994, 20995, 20996, 20997, 20998, 20999, 21000, 21001, 21002, 21003, 21004, 21005, 21006, 21007, 21008, 21009, 21010, 21011, 21012, 21013, 21014, 21015, 21016, 21017, 21018, 21019, 21020, 21021, 21022, 21023, 21024, 21025, 21026, 21027, 21028, 21029, 21030, 21031, 21032, 21033, 21034]
        # bad_idx = [19863, 19864, 19865, 19866, 19867, 19868, 19869, 19870, 19871, 19872, 19873, 19874, 19875, 19876, 19877, 19878, 19879, 19880, 19881, 19882, 19883, 19884, 19885, 19886, 19887, 19888, 19889, 19890, 19891, 19892, 19893, 19894, 19895, 19896, 19897, 19898, 19899, 19900, 19901, 19902, 19903, 19904, 19905, 19906, 19907, 19908, 19909, 19910, 19911, 19912, 19913, 19914, 19915, 19916, 19917, 19918, 19919, 19920, 19921, 19922, 19923, 19924, 19925, 19926, 19927, 19928, 19929, 19930, 19931, 19932, 19933, 19934, 19935, 19936, 19937, 19938, 19939, 19940, 19941, 19942, 19943, 19944, 19945, 19946, 19947, 19948, 19949, 19950, 19951, 19952, 19953, 19954, 19955, 19956, 19957, 19958, 19959, 19960, 19961, 19962, 19963, 19964, 19965, 19966, 19967, 19968, 19969, 19970, 19971, 19972, 19973, 19974, 19975, 19976, 19977, 19978, 19979, 19980, 19981, 19982, 19983, 19984, 19985, 19986, 19987, 19988, 19989, 19990, 19991, 19992]
        bad_idx = []
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
        #    gc.collect()
        print(f"Bad file: {bad_idx}")
        #print number of the labels
        self.dataframe = self.dataframe[~self.dataframe['filename'].isin(bad_idx)]

        print(self.dataframe)
        print(self.dataframe['BC'].value_counts())
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        # self.okt = Okt()
        ret = {}

        item = self.dataframe.iloc[index]
        # print(item)
        idx = item['filename']
        trans = item['transcript']
        lable = item['BC']
        start = item['start']
        end   = item['end']
        role = item['role']

        # path = os.path.join(self.path, "video", f"{str(idx)}.mp4")
        # frames = self.load_video_decord(path)

        path = os.path.join(self.path, "audio", f"{str(idx)}.wav")
        
        audio, sr = torchaudio.load(path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000
        audio = audio[:, -int(self.length*sr):]
        if audio.size(1) != int(sr * 1.5):
            audio = F.pad(audio, (0, int(sr * 1.5) - audio.size(1)), "constant", 0)
        if audio.size(0) != 1:
            audio = audio.sum(0, keepdim=True)

        sentiment = torch.zeros(5)
        # for word in self.okt.morphs(trans):
        for word in trans.split():
            # print(word)
            r_word, s_word = KnuSL.data_list(word)
            if s_word != 'None':
                sentiment[int(s_word)] += 1
            else:
                sentiment[0] += 1
        sentiment = sentiment / sentiment.sum()
        
        trans = self.tokenizer(trans, padding='max_length', max_length=10, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        # if len(frames) != 8:
        #     print("!\t",len(frames), index, idx)
        if audio.size(1) != 24000:
            print("!!\t", index, idx)
        if len(trans) != 10:    
            print("!!!\t", index, idx)
        if len(sentiment) != 5:
            print("!!!!\t", index, idx)
            
        # ret['frames'] = torch.tensor(frames)
        ret['audio'] = audio
        ret['label'] = lable
        ret['text'] = trans
        ret['sentiment'] = sentiment
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