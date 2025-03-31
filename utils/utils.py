import torch.distributed as dist
def get_audio_model(name):
    from model.hubert import HuBert
    from model.audio_lstm import Audio_LSTM
    from model.audio_stft_cnn import AudioStftCnn

    if name == 'LSTM':
        audio_model = Audio_LSTM()
    elif name == 'HuBert':
        audio_model = HuBert(sample_rate=16000)
    elif name == 'STFT_CNN':
        audio_model = AudioStftCnn()
    else:
        raise NotImplementedError
    return audio_model

def get_video_model(name):
    from model.video_mae import VideoMAE
    if name == 'VideoMAE':
        video_model = VideoMAE()
    else:
        raise NotImplementedError
    return video_model

def get_language_model(name):
    if name == 'koBert':
        import json
        from kobert_tokenizer import KoBERTTokenizer
        from transformers import BertModel, DistilBertModel, XLNetTokenizer
        # from kobert_tokenizer import KoBertTokenizer
        # from kobert.pytorch_kobert import get_pytorch_kobert_model
        # bert, vocab = get_pytorch_kobert_model()
        # tokenizer = KoBertTokenizer.from_pretrained('skt/kobert-base-v1')
        tokenizer = XLNetTokenizer.from_pretrained('skt/kobert-base-v1')
        # tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        bert = BertModel.from_pretrained("skt/kobert-base-v1")
        sentiment_dict = json.load(open('data/SentiWord_info.json', encoding='utf-8-sig', mode='r'))
    # elif name == 'Bert':
        # from transformers import BertModel, AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
    elif name == 'ELECTRA':
        from transformers import AutoTokenizer, AutoModelForPreTraining
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        bert = AutoModelForPreTraining.from_pretrained("google/electra-base-discriminator")
    else:
        raise NotImplementedError
    return tokenizer, bert

def get_dataset(name, path, tokenizer):
    from torch.utils.data import Subset
    from dataset.SWBD_Dataset import SWBD_Dataset
    from dataset.ETRI_Dataset import ETRI_All_Video_Dataset, ETRI_All_TT_Video_Dataset, ETRI_All_Dialog_Video_Dataset, ETRI_All_Threshold_Video_Dataset, ETRI_All_TT_Threshold_Video_Dataset
    if name == 'SWBD':
        dataset = SWBD_Dataset(path = './data', tokenizer=tokenizer, length=1.5)
        train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))
        val_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))
        num_class = 3
    elif name == 'ETRI_ALL_Video':
        train_dataset = ETRI_All_Video_Dataset(path = path, train=True, tokenizer=tokenizer, length=1.5, balanced=False)
        val_dataset   = ETRI_All_Video_Dataset(path = path, train=False, tokenizer=tokenizer,  length=1.5, balanced=False)
        num_class = 4
    elif name == 'ETRI_ALL_TT_Video':
        train_dataset = ETRI_All_TT_Video_Dataset(path = path, train=True,  tokenizer=tokenizer, length=1.5, balanced=False)
        val_dataset   = ETRI_All_TT_Video_Dataset(path = path, train=False, tokenizer=tokenizer, length=1.5, balanced=False)
        # train_dataset = ETRI_All_TT_Clip_Dataset(path = path, train=True, tokenizer=tokenizer, length=1.5)
        # val_dataset = ETRI_All_TT_Clip_Dataset(path = path, train=False, tokenizer=tokenizer,  length=1.5)
        num_class = 4
    elif name == 'ETRI_ALL_Dialog_Video':
        train_dataset = ETRI_All_Dialog_Video_Dataset(path = path, train=True,  tokenizer=tokenizer, length=1.5)
        val_dataset   = ETRI_All_Dialog_Video_Dataset(path = path, train=False, tokenizer=tokenizer, length=1.5)
        num_class = 4
    elif name == 'ETRI_ALL_Dialog_Video_3S':
        train_dataset = ETRI_All_Dialog_Video_Dataset(path = path, train=True,  tokenizer=tokenizer, length=3)
        val_dataset   = ETRI_All_Dialog_Video_Dataset(path = path, train=False, tokenizer=tokenizer, length=3)
        num_class = 4
    elif name == 'ETRI_ALL_Threshold_Video':
        train_dataset = ETRI_All_Threshold_Video_Dataset(path = path, train=True,  tokenizer=tokenizer, length=3, balanced=False)
        val_dataset   = ETRI_All_Threshold_Video_Dataset(path = path, train=False, tokenizer=tokenizer, length=3, balanced=False)
        num_class = 4
    elif name == 'ETRI_ALL_TT_Threshold_Video':
        train_dataset = ETRI_All_TT_Threshold_Video_Dataset(path = path, train=True,  tokenizer=tokenizer, length=3, balanced=False)
        val_dataset   = ETRI_All_TT_Threshold_Video_Dataset(path = path, train=False, tokenizer=tokenizer, length=3, balanced=False)
        num_class = 4
    else:
        NotImplementedError

    return train_dataset, val_dataset, num_class

def get_backchannel_prediction_model(name):
    from model.bpm_mt import BPM_MT, BPM_ST, BPM_ST_Target, BPM_ST_Target_Token
    from model.ours import Ours
    from model.lora import LoRA_BC
    from model.finetune import Finetune
    from model.Proxy_Prototype import Proxy_Prototype
    from model.diffused_backchannel import Diffused_Backchannel
    from model.ours_video import Ours_Video
    from model.ours_video_allign import Ours_Video_Align
    from model.ours_video_missing_align import Ours_Video_Missing_Align

    try:
        return {
            'BPM_MT': BPM_MT,
            'BPM_ST': BPM_ST,
            'BPM_ST_Target': BPM_ST_Target,
            'BPM_ST_Target_Token': BPM_ST_Target_Token,
            'LoRA' : LoRA_BC,
            'Finetune' : Finetune,
            'Proxy_Prototype' : Proxy_Prototype,
            'Diffused_Backchannel' : Diffused_Backchannel,
            'Ours': Ours,
            'Ours_video' : Ours_Video,
            'Ours_video_align' : Ours_Video_Align,
            'Ours_video_missing_align' : Ours_Video_Missing_Align,
        }[name]
    except:
        raise NotImplementedError
    

import torch
import torchaudio
from torch import Tensor
import math
from typing import Optional

def stretch_waveform(
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

def all_gather(item, device='cuda'):
    _device = item.device
    item = item.to(device)
    local_size = torch.tensor(item.size(), device=device)
    all_sizes = [torch.empty_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = torch.max(torch.stack(all_sizes), dim=0).values
    if (max_size > local_size).any():
        item = item.to_padded_tensor(0.,max_size)
    all_qs_padded = [torch.empty_like(item) for _ in range(dist.get_world_size())]
    dist.all_gather(all_qs_padded, item)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        for i in range(len(size)):
            q = q.index_select(i, torch.arange(size[i], device=device))
            all_qs.append(q)
    return all_qs