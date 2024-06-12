import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from util.utils import get_audio_model, get_language_model
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from util.contrastive_loss import NormSoftmaxLoss, MMS_Loss
import random
import torch.distributed as dist
import torchaudio
from layer.lora import LoRA

class Ours_video(nn.Module):
    def __init__(self, language_model=None, audio_model=None, video_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours_video, self).__init__()

        self.mode = mode
        self.num_classes = num_class
        self.register_module("language_model", language_model)
        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        self.register_module("audio_model", audio_model)
        self.register_module("video_model", video_model)

        self.audio_feature_size = audio_model.get_feature_size()

        self.dropout = nn.Dropout(dropout)

        self.fc_layer_1 = nn.Linear(192*3, output_size)
        self.contrastive_loss = NormSoftmaxLoss(temperature=0.05)

        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)
        self.cross_attn0 = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=12) for _ in range(1)])
        self.cross_attn1 = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=12) for _ in range(1)])
        self.cross_attn2 = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=12) for _ in range(1)])
        self.audio_downproject = nn.Linear(768, 192) 
        self.text_downproject = nn.Linear(768, 192) 
        self.video_downproject = nn.Linear(384, 192) 
        
    def forward(self, x, train="train"):
        
        y = {}
        audio = x["audio"]
        text  = x["text"]
        audio = audio[:, 0, :]
        video = x["video"]
        AB, AL = audio.shape
        TB, TL = text.shape
        
        video = self.video_model(video)
        video = self.video_downproject(video)  
             
        audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)    
        
        for layer in self.audio_model.model.encoder.layers:
            audio = layer(audio)[0]
        audio = self.audio_downproject(audio)  
         
        text = self.language_model.embeddings(text)
        text = self.language_model.encoder(text)[0]
        text = self.text_downproject(text)
    
        for layer in self.cross_attn0:
            at = layer(audio, text)  
        for layer in self.cross_attn1:
            tv = layer(text, video) 
        for layer in self.cross_attn2:
            av = layer(audio, video)  
        
        at = at.mean(1)
        tv = tv.mean(1)
        av = av.mean(1)
        audio = audio.mean(1)
        video = video.mean(1)
        text = text.mean(1)
        
        y["InfoNCE"] = 0.2 * (  self.contrastive_loss(self.sim_matrix(audio,video)) 
                              + self.contrastive_loss(self.sim_matrix(audio,text)) 
                              + self.contrastive_loss(self.sim_matrix(text,video))
                              + self.contrastive_loss(self.sim_matrix(at,video))
                              + self.contrastive_loss(self.sim_matrix(tv,audio))
                              + self.contrastive_loss(self.sim_matrix(av,text)))
           
        concat = torch.cat((audio, text, video), dim=1)
        
    #    y["audio"] = audio
    #    y["text"] = text 
    #    y["video"] = video
        
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