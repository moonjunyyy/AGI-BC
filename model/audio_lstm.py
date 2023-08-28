import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC

class Audio_LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mfcc_extractor = MFCC(sample_rate=16000, n_mfcc=13)
        self.lstm = nn.LSTM(13, 13, 4, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.mfcc_extractor(x).squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x
    
    def get_feature_size(self):
        return 13 * 2