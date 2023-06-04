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
        # os.system(f"rm -r {self.path}/")
        if os.path.isdir(self.path) == False:
            if not os.environ['SLURM_NODELIST'][0] in ['ariel-g2']:
                os.makedirs(self.path, exist_ok=True)
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
        bad_idx = [5413, 5414, 5415, 5416, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425, 5426, 5427, 5428, 5429, 5430, 5431, 5432, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5440, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5449, 5450, 5451, 5452, 5453, 5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5462, 5463, 5464, 5465, 5466, 5467, 5468, 5469, 5470, 5471, 5472, 5473, 5474, 5475, 5476, 5477, 5478, 5479, 5480, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491, 5492, 5493, 5494, 5495, 5496, 5497, 5498, 5499, 5500, 5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 5528, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 5550, 5551, 5552, 5553, 5554, 5555, 5556, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 5569, 5570, 5571, 5572, 5573, 5574, 5575, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591, 5592, 5593, 5594, 5595, 5596, 5597, 5598, 5599, 5600, 5601, 5602, 5603, 5604, 5605, 5606, 5607, 5608, 5609, 5610, 5611, 5612, 5613, 5614, 5615, 5616, 5617, 5618, 5619, 5620, 5621, 5622, 5623, 5624, 5625, 5626, 5627, 5628, 5629, 5630, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5652, 5653, 5654, 5655, 5656, 5657, 5658, 5659, 5660, 5661, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5669, 5670, 5671, 5672, 5673, 5674, 5675, 5676, 5677, 5678, 5679, 5680, 5681, 5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 5693, 5694, 5695, 5696, 5697, 5698, 5699, 5700, 5701, 5702, 5703, 5704, 5705, 13386, 13387, 13388, 13389, 13390, 13391, 13392, 13393, 13394, 13395, 13396, 13397, 13398, 13399, 13400, 13401, 13402, 13403, 13404, 13405, 13406, 13407, 13408, 13409, 13410, 13411, 13412, 13413, 13414, 13415, 13416, 13417, 13418, 13419, 13420, 13421, 13422, 13423, 13424, 13425, 13426, 13427, 13428, 13429, 13430, 13431, 13432, 13433, 13434, 13435, 13436, 13437, 13438, 13439, 13440, 13441, 13442, 13443, 13444, 13445, 13446, 13447, 13448, 13449, 13450, 13451, 13452, 13453, 13454, 13455, 13456, 13457, 13458, 13459, 13460, 13461, 13462, 13463, 13464, 13465, 13466, 13467, 13468, 13469, 13470, 13471, 13472, 13473, 13474, 13475, 13476, 13477, 13478, 13479, 13480, 13481, 13482, 13483, 13484, 13485, 13486, 13487, 13488, 13489, 13490, 13491, 13492, 13493, 13494, 13495, 13496, 13497, 13498, 13499, 13500, 13501, 13502, 13503, 13504, 13505, 13506, 13507, 13508, 13509, 13510, 13511, 13512, 13513, 13514, 13515, 13516, 13517]
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
        print(self.dataframe['BC'].value_counts())

        self.dataframe = self.dataframe[~self.dataframe['filename'].isin(bad_idx)]
        # print(self.dataframe)
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        self.okt = Okt()
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
        for word in self.okt.morphs(trans):
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