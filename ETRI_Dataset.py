import os
import gc
import shutil
from torch.fft import fft, ifft
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
from torch import Tensor
from typing import Optional, Tuple
     
def _stretch_waveform(
    waveform: Tensor,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: int = 512,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    window: Optional[Tensor] = None,
) -> Tensor:
    '''
        Stretching the audio from TorchAudio Implementation
        Due to the pitch shift cannot be used in the dataloader, we use this function to stretch the audio
    '''
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(window_length=win_length, device=waveform.device)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    ori_len = shape[-1]
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    phase_advance = torch.linspace(0, math.pi * hop_length, spec_f.shape[-2], device=spec_f.device)[..., None]
    spec_stretch = torchaudio.functional.phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    waveform_stretch = torch.istft(
        spec_stretch, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=len_stretch
    )
    return waveform_stretch

class ETRI_Corpus_Dataset(Dataset):
    def __init__(self, path, tokenizer, train = False, transform : Callable=None, length :float = 1.5) -> None:
        super().__init__()
        print("Load ETRI_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join(path, "ETRI_Corpus_Clip")
        # if not os.environ['SLURM_NODELIST'] in ['ariel-g1', 'ariel-g2', 'ariel-g4', 'ariel-v8', 'ariel-v13']:
        # if not os.environ['SLURM_NODELIST'] in ['ariel-g4']:
        if not os.environ['SLURM_NODELIST'] in ['ariel-g1']:
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

        # augment the audio
        # if self.train:

        #     # pitch shift            
        #     pitch = torch.rand(1).item() * 4 - 8
        #     audio_shape = audio.shape
        #     audio = _stretch_waveform(audio, pitch)
        #     audio = F.interpolate(audio.unsqueeze(0), size=audio_shape[1], mode='linear', align_corners=True).squeeze(0)
        #     audio = audio[:, :int(sr * 1.5)]

            # random noise
            # audio = audio + torch.randn_like(audio) * 0.1

            # # random volume
            # audio = audio * torch.rand(1).item() * 0.5 + audio

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
        
        trans = self.tokenizer(trans, padding='max_length', max_length=20, truncation=True, return_tensors="pt")['input_ids'].squeeze()

        # if len(frames) != 8:
        #     print("!\t",len(frames), index, idx)
        if audio.size(1) != 24000:
            print("!!\t", index, idx)
        if len(trans) != 20:    
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
   
    def get_sample_in_class(self):
        return self.dataframe['BC'].value_counts().to_numpy()