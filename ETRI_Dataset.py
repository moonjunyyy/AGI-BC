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
        bad_idx = []
        # bad_idx = [1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1552, 1553, 2326, 2327, 2328, 2329, 2330, 2332, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3083, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3096, 3097, 3098, 3099, 3100, 3101, 3103, 3104, 3105, 3107, 3108, 3109, 3110, 3111, 3112, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3126, 4022, 4023, 4024, 4027, 4028, 4030, 4034, 4035, 4036, 4039, 4041, 4045, 4047, 4049, 4051, 4053, 4054, 4057, 4059, 4060, 4062, 4064, 4065, 4068, 4070, 4072, 4073, 4075, 4077, 4080, 4082, 4085, 4086, 4090, 4093, 4099, 4100, 4103, 4104, 4107, 4116, 4122, 4125, 4127, 4130, 4133, 4137, 4139, 4141, 4143, 4146, 4148, 4149, 4151, 4153, 4155, 4159, 4160, 4162, 4164, 4167, 4170, 4172, 4174, 4176, 4178, 4181, 4183, 4184, 4186, 4188, 4189, 4190, 4194, 4196, 4198, 4207, 4211, 4215, 4217, 4221, 4223, 4226, 4228, 4232, 4233, 4235, 4236, 4237, 4239, 4240, 4241, 4243, 4244, 4246, 4247, 4250, 4252, 4253, 4255, 4257, 4259, 4261, 4264, 4266, 4268, 4269, 4271, 4273, 4274, 4278, 4279, 4283, 4285, 4287, 4288, 4289, 4291, 4293, 4295, 4297, 4302, 4585, 4586, 4587, 4588, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4608, 4610, 4611, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4634, 4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643, 4644, 5046, 5047, 5048, 5049, 5050, 5052, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061, 5646, 5647, 5648, 5649, 6164, 6168, 6170, 6171, 6172, 6173, 6174, 6175, 6178, 6179, 6180, 6181, 6182, 6184, 6185, 6187, 6188, 6189, 6190, 6191, 6192, 6193, 6194, 6195, 6196, 6197, 6198, 6199, 6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6213, 6214, 6215, 6216, 6217, 6218, 6219, 6220, 6221, 6222, 6223, 6224, 6225, 6226, 6227, 6228, 6229, 6230, 6231, 6232, 6234, 6235, 6236, 6237, 6238, 6239, 6240, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7140, 7493, 7494, 7495, 7496, 7497, 7498, 7499, 7500, 8114, 8115, 8116, 8118, 8119, 8120, 8121, 8122, 8123, 8124, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8132, 8133, 8134, 8135, 8136, 8137, 8138, 8139, 8140, 8141, 8142, 8143, 8144, 8145, 8146, 8147, 8150, 8151, 8152, 8153, 8154, 8155, 8156, 8157, 8158, 8159, 8160, 8162, 8163, 8164, 8165, 8166, 8167, 8168, 8169, 8170, 8172, 8173, 8174, 8175, 8176, 8177, 8178, 8179, 8180, 8181, 8182, 8183, 8184, 8185, 9420, 9421, 9423, 9425, 9426, 9427, 9428, 11158, 11160, 11161, 11162, 11163, 11164, 11165, 11166, 11169, 11170, 11171, 11172, 11173, 11175, 11177, 11178, 11179, 11181, 11183, 11184, 11185, 11186, 11187, 11188, 11189, 11190, 11191, 11193, 11194, 11195, 11196, 11197, 11198, 11199, 11200, 11201, 11202, 11203, 11204, 11206, 11207, 11209, 11210, 11399, 11400, 11401, 11402, 11403, 11404, 11406, 11407, 11408, 11410, 11411, 11412, 11413, 11416, 11420, 11421, 11423, 11424, 11426, 11427, 11428, 11429, 11430, 11431, 11432, 11435, 11437, 11439, 11440, 11441, 11442, 11444, 11447, 11448, 11451, 11456, 11457, 11460, 11462, 11463, 11465, 11466, 11467, 11468, 11469, 11470, 11471, 11472, 11474, 11476, 11478, 11480, 11481, 11482, 11484, 11485, 12333, 12334, 12335, 12336, 12337, 12338, 12340, 12341, 12342, 12343, 12345, 12346, 12347, 12348, 12349, 12350, 12351, 12352, 12354, 12355, 12356, 12357, 12358, 12359, 12361, 12362, 12363, 12366, 12367, 12368, 12369, 12370, 12371, 12372, 12373, 12374, 12375, 12376, 12377, 12378, 12379, 12380, 12381, 12382, 12383, 12384, 12385, 12386, 12387, 12388, 12389, 12390, 12658, 12659, 12660, 12661, 12662, 12663, 12664, 12665, 12666, 12667, 12668, 12669, 12670, 12671, 12672, 12673, 12674, 12675, 12676, 12677, 12678, 12679, 12680]
        for idx, row in self.dataframe.iterrows():
            if idx in bad_idx:
                continue
            try:
                # path = os.path.join(self.path, "video", f"{idx}.mp4")
                # frames = self.load_video_decord(path)
                # if len(frames) == 0:
                #     print(f"video cannot be loaded : {path}")
                #     raise
                
                # del(frames)
                path = os.path.join(self.path, "audio", f"{idx}.wav")
                audio, sr = torchaudio.load(path)
                if audio.numel() == 0:
                    print(f"audio cannot be loaded : {path}")
                    raise
                del(audio)
            except:
                # print(f"Bad file: {idx}")
                bad_idx.append(idx)
            gc.collect()
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
        audio = audio[:, -int(self.length*sr):]
        if audio.size(1) != int(sr * 1.5):
            audio = F.pad(audio, (0, int(sr * 1.5) - audio.size(1)), "constant", 0)


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