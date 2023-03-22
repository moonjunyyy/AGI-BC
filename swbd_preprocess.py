import os
import pandas as pd
import torch
import torchvision
import torchaudio
from torch.utils.data import Dataset
from typing import Callable
import math
import datetime

df = pd.read_csv("/data/datasets/swbd/swbd.tsv", sep='\t', encoding='utf-8')
print(df)
badfile = []

for idx, row in df.iterrows():
    print(idx)
    if len(badfile) != 0:
        if idx not in badfile:
            continue
    start = row['start']
    end   = row['end']

    if start > end:
        start, end = end, start

    start = str(datetime.timedelta(seconds=start))
    end = str(datetime.timedelta(seconds=end))
    
    # path = f"/data/datasets/ETRI_resize/{row['folder']}/{row['role']}.mp4"
    # print(f'ffmpeg -y -i {path} -ss {start} -to {end} -c copy /data/datasets/ETRI_resize/{idx}.mp4')
    # os.system(f'ffmpeg -y -i {path} -ss {start} -to {end} /data/datasets/ETRI_resize/{idx}.mp4')
    # # os.system(f'ffmpeg -i {path} -ss {start} -to {end} -c copy /data/datasets/ETRI_resize/{idx}.mp4')

    path = f"/data/datasets/swbd/adc/sw{row['folder']}.wav"
    print(f'ffmpeg -y -i {path} -ss {start} -to {end} -c copy /data/datasets/swbd/clip/{idx}.wav')
    os.system(f'ffmpeg -y -i {path} -ss {start} -to {end}  /data/datasets/swbd/clip/{idx}.wav')
    # if idx == 1:
    #     break