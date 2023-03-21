import torch
from torchaudio.transforms import MFCC
from ETRI_Dataset import ETRI_Corpus_Dataset
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import numpy as np
from BPM_MT import BPM_MT, BPM_ST

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchaudio.transforms import MFCC
from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
from ETRI_Dataset import ETRI_Corpus_Dataset
import re

sentiment_dict = {}
with open('data/subjclueslen1-HLTEMNLP05.tff', 'r', encoding='utf-8') as f:
    # sentiment_dict = { re.split(" =\n", line) for line in f.readlines()}
    for line in f.readlines():
        line = re.split("=| |\n", line)
        if line[11] == 'neutral':
            sentiment_dict[line[5]] = 0
        elif line[11] == 'positive':
            if line[0] == 'strongsubj':
                sentiment_dict[line[5]] = 2
            else:
                sentiment_dict[line[5]] = 1
        elif line[11] == 'negative':
            if line[0] == 'strongsubj':
                sentiment_dict[line[5]] = -2
            else:
                sentiment_dict[line[5]] = -1