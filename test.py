import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import os
import math
import pandas as pd
import torchaudio
from typing import Optional, Tuple
from torch import Tensor

def _stretch_waveform(
    waveform: Tensor,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: int = 512,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    window: Optional[Tensor] = None,
) -> Tensor:
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

audio, sr = torchaudio.load("./sample.wav")
audio = torchaudio.transforms.Resample(sr, 16000)(audio)
sr = 16000
audio = audio[:, -int(10*sr):]

audio_shape = audio.shape
print(audio_shape)
pitch = (torch.rand(1).item() - 0.5) * 12
print(pitch)
audio = _stretch_waveform(audio, pitch)
print(audio.shape)
audio = F.interpolate(audio.unsqueeze(0), size=audio_shape[1], mode='linear', align_corners=True).squeeze(0)
print(audio.shape)

torchaudio.save("./sample2.wav", audio, sr)


# df = pd.read_csv("/data/datasets/ETRI_Word/etri.tsv", sep='\t', index_col=0)
# print(df)
# dataset = pd.DataFrame(columns=['folder', 'role', 'start', 'end', 'transcript', 'BC'])

# text = []
# start = []
# end = 0
# role = None
# cls = -1
# folder = None

# for i, dat in df.iterrows():

#     if folder != dat['folder']:
#         folder = dat['folder']
#         start = []
#         end = 0
#         cls = -1
#         text = []
#         role = None
    
#     start.append(dat['start'])
#     end = dat['end']
#     text.append(dat['transcript'])
#     text.append(" ")
    
#     if role != None:
#         if dat['class'] != 5 and cls not in [1, 2, 3, 4]:
#             print("".join(text[:-1]))
#             dataset = pd.concat([dataset, pd.DataFrame([[dat['folder'], role, start[0], end, "".join(text[:-1]), dat['class'] if dat['class'] != 5 else 0]], columns=['folder', 'role', 'start', 'end', 'transcript', 'BC'])])
#         if role != dat['role']:
#             text = []
#             start = []
#             end = 0
#             role = None
#             folder = None
#             cls = -1

#     role = dat['role']
#     cls = dat['class']

#     # if dat['class'] != 5:
#     #     if role != None:
#     #         print("".join(text[:-1]))
#     #         dataset = pd.concat([dataset, pd.DataFrame([[dat['folder'], role, start[0], end, "".join(text[:-1]), dat['class'] if dat['class'] != 5 else 0]], columns=['folder', 'role', 'start', 'end', 'text', 'class'])])

#     if len(text) > 8:
#         text.pop(0)
#         text.pop(0)
#         start.pop(0)

# dataset = dataset.reset_index(drop=True)
# print(dataset)
# print(dataset['BC'].value_counts())
# dataset.to_csv("/data/datasets/ETRI_Word/etri_5words.tsv", sep='\t')

# df = pd.read_csv("/data/datasets/ETRI_Corpus_Clip/etri.tsv", sep='\t', index_col=0)
# folder = 1
# for i, dat in df.iterrows():
#     if dat['folder']==folder:
#         print(dat)
#         folder += 1