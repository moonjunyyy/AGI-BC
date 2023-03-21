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

tokenizer = SentencepieceTokenizer(get_tokenizer())
bert, vocab = get_pytorch_kobert_model()

ETRI_Dataset = ETRI_Corpus_Dataset('/local_datasets', tokenizer, vocab)
dataloader = DataLoader(ETRI_Dataset, batch_size=1, shuffle=True, num_workers=8)

labels = [0,0,0,0]
for i, batch in enumerate(dataloader):
    for lbl in batch['label']:
        labels[lbl] += 1
print(labels)