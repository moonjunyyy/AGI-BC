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
from SWBD_Dataset import SWBD_Dataset
import re

from transformers import BertModel
from transformers import AutoTokenizer

input_size = 13
hidden_size = 13
num_layers = 4
batch_first = True
bidirectional = True
dropout = 0.3
num_class = 4
output_size = 128
audio_feature_size = 13

lstm = torch.nn.LSTM(input_size=audio_feature_size, hidden_size=audio_feature_size, num_layers=6, batch_first=True, bidirectional=True)
x = torch.randn(1, 100, 13)
x, _ = lstm(x)
print(x.shape)