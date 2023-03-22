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

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, output_hidden_states=True, output_attentions=True)

SWBD_Dataset = SWBD_Dataset("/data/datasets", None, tokenizer)
print(len(SWBD_Dataset))