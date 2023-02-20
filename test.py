import torch
from torchaudio.transforms import MFCC

mfcc = MFCC(sample_rate=16000, n_mfcc=13)
audio = torch.randn(5, 16000)
mfcc_audio = mfcc(audio)

print(mfcc_audio.shape)
mfcc_audio = mfcc_audio.permute(0, 2, 1)

lstm = torch.nn.LSTM(input_size=13, hidden_size=13, num_layers=4, batch_first=True, bidirectional=True)
lstm_audio, _ = lstm(mfcc_audio)
print(lstm_audio.shape)