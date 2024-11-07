import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from util.utils import get_audio_model, get_language_model
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from util.contrastive_loss import NormSoftmaxLoss, MMS_Loss, IntensiveClassNormSoftmaxLoss, ClassNormSoftmaxLoss
import random
import torch.distributed as dist
import torchaudio
from layer.lora import LoRA
from marlin_pytorch import Marlin

class Ours_video(nn.Module):
    def __init__(self, language_model=None, audio_model=None, video_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours_video, self).__init__()

        self.mode = mode
        self.num_classes = num_class
        self.register_module("language_model", language_model)
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        self.register_module("audio_model", audio_model)
        self.register_module("video_model", video_model)
        self.audio_feature_size = audio_model.get_feature_size()

        self.dropout = nn.Dropout(dropout)
        
        self.relu = nn.ReLU()
        self.fc_layer_1 = nn.Linear(192*3, output_size)
        self.classifier = nn.Linear(output_size, num_class)

        self.loss = "sample"
        if self.loss == "intensive":
            self.contrastive_loss = IntensiveClassNormSoftmaxLoss(temperature=0.05)
            print("loss type : Intensive")
        elif self.loss == "class":
            self.contrastive_loss = ClassNormSoftmaxLoss(temperature=0.05)
            print("loss type : class")
        elif self.loss == "cosine":
            self.contrastive_loss = CosineLoss()
        else:
            self.contrastive_loss = NormSoftmaxLoss(temperature=0.05)
            print("loss type : sample")
        
        self.cross_attn_ta = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=12) for _ in range(1)])
        self.cross_attn_va = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=12) for _ in range(1)])
        self.cross_attn_vt = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=12) for _ in range(1)])
        
        self.a_downproject = nn.Linear(768, 192) 
        self.t_downproject = nn.Linear(768, 192) 
        self.v_downproject = nn.Linear(384, 192) 
         
    def forward(self, x, train="train", warmup=False):
        
        if warmup:
            y = {}
            audio = x["audio"]
            text  = x["text"]
            audio = audio[:, 0, :]
            video = x["video"]
            AB, AL = audio.shape
            TB, TL = text.shape

            y["InfoNCE"] = 0

            audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
            audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
            audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
            audio = self.audio_model.model.encoder.layer_norm(audio)
            audio = self.audio_model.model.encoder.dropout(audio)    
        
            text = self.language_model.embeddings(text)
            video = self.video_model.model.embeddings(video.float(), None) 
            
            for i, (audio_layer, text_layer, video_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer, self.video_model.model.encoder.layer)):
                if i < 9:
                    audio = audio_layer(audio)[0]
                    text = text_layer(text)[0]
                    video = video_layer(video)[0]
            
                else:
                    audio = audio_layer(audio)[0]
                    text = text_layer(text)[0]
                    video = video_layer(video)[0]

                    _audio = self.a_downproject(audio)
                    _text = self.t_downproject(text)
                    _video = self.v_downproject(video)
                    
                    for layer in self.cross_attn_ta:
                        ta = layer(_text, _audio)  
                    for layer in self.cross_attn_vt:
                        vt = layer(_video, _text) 
                    for layer in self.cross_attn_va:
                        va = layer(_video, _audio)  
                        
                    _audio = _audio.mean(1)
                    _text = _text.mean(1)
                    _video = _video.mean(1)
                        
                    ta = ta.mean(1)
                    vt = vt.mean(1)
                    va = va.mean(1)
                    
                    if self.loss == "class":
                        y["InfoNCE"] += 0.1 * (self.contrastive_loss(self.sim_matrix(_audio, _text), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(_audio, _video), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(_video, _text), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(ta, _video), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(vt, _audio), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(va, _text), AB, labels))
                    elif self.loss == "intensive":
                        a_t_strong, a_t_weak = self.contrastive_loss(self.sim_matrix(_audio, _text), AB, labels)
                        a_v_strong, a_v_weak = self.contrastive_loss(self.sim_matrix(_audio, _video), AB, labels)
                        v_ta_strong, v_ta_weak = self.contrastive_loss(self.sim_matrix(ta, _video), AB, labels)
                        t_va_strong, t_va_weak = self.contrastive_loss(self.sim_matrix(va, _text), AB, labels)
                        
                        y["InfoNCE"] += (0.1 * (a_t_strong + a_v_strong + v_ta_strong + t_va_strong) 
                                        + 0.02 * (a_t_weak + a_v_weak + v_ta_weak + t_va_weak))
                    else:
                        
                        y["InfoNCE"] += 1.0 * (self.contrastive_loss(self.sim_matrix(_audio, _text))
                                        + self.contrastive_loss(self.sim_matrix(_audio, _video))
                                        + self.contrastive_loss(self.sim_matrix(_video, _text))
                                        + self.contrastive_loss(self.sim_matrix(ta, _video))
                                        + self.contrastive_loss(self.sim_matrix(vt, _audio))
                                        + self.contrastive_loss(self.sim_matrix(va, _text)))
                
            return y
            
        else:
            y = {}
            audio = x["audio"]
            text  = x["text"]
            audio = audio[:, 0, :]
            video = x["video"]
            labels = x["label"]
            AB, AL = audio.shape
            TB, TL = text.shape

            y["InfoNCE"] = 0
            with torch.no_grad():
                audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
                audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
                audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
                audio = self.audio_model.model.encoder.layer_norm(audio)
                audio = self.audio_model.model.encoder.dropout(audio)    
            
                text = self.language_model.embeddings(text)
                video = self.video_model.model.embeddings(video.float(), None) 
            
            for i, (audio_layer, text_layer, video_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer, self.video_model.model.encoder.layer)):
                if i < 9:
                    audio = audio_layer(audio)[0]
                    text = text_layer(text)[0]
                    video = video_layer(video)[0]
            
                else:
                    audio = audio_layer(audio)[0]
                    text = text_layer(text)[0]
                    video = video_layer(video)[0]

                    _audio = self.a_downproject(audio)
                    _text = self.t_downproject(text)
                    _video = self.v_downproject(video)
                    
                    for layer in self.cross_attn_ta:
                        ta = layer(_text, _audio)  
                    for layer in self.cross_attn_vt:
                        vt = layer(_video, _text) 
                    for layer in self.cross_attn_va:
                        va = layer(_video, _audio)  
                        
                    _audio = _audio.mean(1)
                    _text = _text.mean(1)
                    _video = _video.mean(1)
                        
                    ta = ta.mean(1)
                    vt = vt.mean(1)
                    va = va.mean(1)
                    
                    if self.loss == "class":
                        y["InfoNCE"] += 0.1 * (self.contrastive_loss(self.sim_matrix(_audio, _text), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(_audio, _video), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(_video, _text), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(ta, _video), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(vt, _audio), AB, labels)
                                        + self.contrastive_loss(self.sim_matrix(va, _text), AB, labels))
                    elif self.loss == "intensive":
                        a_t_strong, a_t_weak = self.contrastive_loss(self.sim_matrix(_audio, _text), AB, labels)
                        a_v_strong, a_v_weak = self.contrastive_loss(self.sim_matrix(_audio, _video), AB, labels)
                        v_ta_strong, v_ta_weak = self.contrastive_loss(self.sim_matrix(ta, _video), AB, labels)
                        t_va_strong, t_va_weak = self.contrastive_loss(self.sim_matrix(va, _text), AB, labels)
                        
                        y["InfoNCE"] += (0.1 * (a_t_strong + a_v_strong + v_ta_strong + t_va_strong) 
                                        + 0.02 * (a_t_weak + a_v_weak + v_ta_weak + t_va_weak))
                    else:
                        
                        y["InfoNCE"] += 0.1 * (self.contrastive_loss(self.sim_matrix(_audio, _text))
                                        + self.contrastive_loss(self.sim_matrix(_audio, _video))
                                        + self.contrastive_loss(self.sim_matrix(_video, _text))
                                        + self.contrastive_loss(self.sim_matrix(ta, _video))
                                        + self.contrastive_loss(self.sim_matrix(vt, _audio))
                                        + self.contrastive_loss(self.sim_matrix(va, _text)))

            audio = self.a_downproject(audio)  
            text = self.t_downproject(text)
            video = self.v_downproject(video)

            
            for layer in self.cross_attn_ta:
                ta = layer(text, audio)  
            for layer in self.cross_attn_va:
                va = layer(video, audio)  
            
            audio = audio.mean(1)
            ta = ta.mean(1)
            va = va.mean(1)
            
            concat = torch.cat((audio, ta, va), dim=1) 
            
            y["logit"] = self.fc_layer_1(self.dropout(concat))
            y["logit"] = self.relu(y["logit"])
            y["logit"] = self.classifier(self.dropout(y["logit"]))

            return y
    
    def normalize_embeddings(self, a, eps=1e-8):
        a_n = a.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        return a_norm

    def sim_matrix(self, a, b, eps=1e-8):
        a = self.normalize_embeddings(a, eps)
        b = self.normalize_embeddings(b, eps)

        sim_mt = torch.mm(a, b.transpose(0, 1))
        return sim_mt
    
    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=item.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=item.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        all_qs = torch.cat(all_qs, dim=0)
        return all_qs
    
                


