import os
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from utils.utils import get_audio_model, get_language_model, all_gather
from utils.kmeans import KMeans
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE
import logging
import matplotlib.pyplot as plt

class Diffused_Backchannel(nn.Module):
    BATCH_SIZE = 2 ** 14

    def __init__(self,
                 language_model=None,
                 audio_model=None,
                 video_model=None,
                 sentiment_dict = None,
                 output_size=128,
                 num_class=4,
                 sentiment_output_size=64,
                 dropout=0.3,
                 mode="cross_entropy"):
        super(Diffused_Backchannel, self).__init__()

        self.multi_modal = False
        self.class_wise = False
        self.cross_attn = False
        self.consistency = False

        self.mode = mode
        self.num_classes = num_class

        if language_model is not None:
            self.register_module("language_model", language_model)
            # if bert and vocab are not provided, raise an error
            assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        if audio_model is not None:
            self.register_module("audio_model", audio_model)
            # define the LSTM layer, 4 of layers
            self.audio_feature_size = audio_model.get_feature_size()

        if video_model is not None:
            self.register_module("video_model", video_model)
            self.video_feature_size = video_model.get_feature_size()

        self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

        self.language_linear = nn.ModuleDict()
        self.language_lora = nn.ModuleDict()
        self.audio_linear = nn.ModuleDict()
        self.audio_lora = nn.ModuleDict()

        for name, module in self.audio_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                self.audio_lora[name.replace('.', '_')] = LoRA(module, 4, alpha=8)
                self.audio_linear[name.replace('.', '_')] = module

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                self.language_lora[name.replace('.', '_')] = LoRA(module, 4, alpha=8)
                self.language_linear[name.replace('.', '_')] = module

        self.lora_on()
        
        self.register_buffer("betas", torch.arange(0, 1, 1/2000).to(torch.float32))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        print("Betas: ", self.betas)
        print("Alphas: ", self.alphas)
        print("Alpha Bars: ", self.alpha_bars)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768 + self.audio_model.get_feature_size(), num_class)
        self.internal_counter = 1

    def lora_on(self):
        for name, module in self.audio_model.named_modules():
            # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.audio_lora[name.replace('.', '_')])

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.language_lora[name.replace('.', '_')])

    def lora_off(self):
        for name, module in self.audio_model.named_modules():
            # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.audio_linear[name.replace('.', '_')])

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.language_linear[name.replace('.', '_')])

    def forward(self, x):
        self.lora_on()
        # Extract the features from the audio and text
        device = self.parameters().__next__().device
        audio = x["audio"]
        text  = x["text"]
        target_audio = x["target_audio"]
        target_text = x["target_text"]
        y = {}
        # get audio only one channel
        audio = audio[:, 0, :]
        target_audio = target_audio[:, 0, :]
        AB, AL = audio.shape
        TB, TL = text.shape
        random_steps = torch.randint(0, self.internal_counter, (AB,)).to(device)
        
        audio = self.audio_model.model.feature_extractor(audio)
        audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        
        target_audio = self.audio_model.model.feature_extractor(target_audio)
        target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))

        text_embedding = self.language_model.embeddings(text)

        target_text_embedding = self.language_model.embeddings(target_text)

        if self.training:
            noise_audio = torch.sqrt(1 - self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * torch.randn_like(target_audio_embedding) + \
                        torch.sqrt(self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * target_audio_embedding

            audio = torch.cat((audio_embedding, target_audio_embedding), dim=1)
            noise_audio = torch.cat((audio_embedding, noise_audio), dim=1)

            audio = self.audio_model.model.encoder(audio)[0]
            noise_audio = self.audio_model.model.encoder(noise_audio)[0]

            y['audio'] = F.mse_loss(audio, noise_audio)

            noise_text = torch.sqrt(1 - self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * torch.randn_like(target_text_embedding) + \
                        torch.sqrt(self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * target_text_embedding

            text = torch.cat((text_embedding, target_text_embedding), dim=1)
            noise_text = torch.cat((text_embedding, noise_text), dim=1)

            text = self.language_model.encoder(text)[0]
            noise_text = self.language_model.encoder(noise_text)[0]

            y['text'] = F.mse_loss(text, noise_text)

            audio = (audio.mean(dim=1) + noise_audio.mean(dim=1)) / 2
            text = (text[:, 0, :] + noise_text[:, 0, :]) / 2
        
        else:
            audio = torch.cat((audio_embedding, torch.rand_like(target_audio_embedding)), dim=1)
            audio = self.audio_model.model.encoder(audio)[0]

            text = torch.cat((text_embedding, torch.rand_like(target_text_embedding)), dim=1)
            text = self.language_model.encoder(text)[0]

            audio = audio.mean(dim=1)
            text = text[:, 0, :]

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        y["logit"] = self.classifier(self.dropout(concat))
        if self.internal_counter != 2000:
            self.internal_counter += 1
        return y
    
    def post_epoch(self, dataloader):
        with torch.no_grad():
            acc = 0
            total = 0
            for i, x in enumerate(dataloader):
                self.lora_on()
                # Extract the features from the audio and text
                device = self.parameters().__next__().device
                audio = x["audio"].to(device)
                text  = x["text"].to(device)
                target_audio = x["target_audio"].to(device)
                target_text = x["target_text"].to(device)
                y = {}
                # get audio only one channel
                audio = audio[:, 0, :]
                target_audio = target_audio[:, 0, :]
                AB, AL = audio.shape
                TB, TL = text.shape
                random_steps = torch.randint(1, self.internal_counter+1, (AB,)).to(device)
                
                audio = self.audio_model.model.feature_extractor(audio)
                audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                
                target_audio = self.audio_model.model.feature_extractor(target_audio)
                target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))

                text_embedding = self.language_model.embeddings(text)

                target_text_embedding = self.language_model.embeddings(target_text)

                audio = torch.cat((audio_embedding, target_audio_embedding), dim=1)
                audio = self.audio_model.model.encoder(audio)[0]

                text = torch.cat((text_embedding, target_text_embedding), dim=1)
                text = self.language_model.encoder(text)[0]

                audio = audio.mean(dim=1)
                text = text[:, 0, :]

                if self.mode == "audio_only" or self.mode == "text_only":
                    concat = audio if self.mode == "audio_only" else text
                else :
                    concat = torch.cat((audio, text), dim=1)
                y["logit"] = self.classifier(self.dropout(concat))
                if self.internal_counter != 2000:
                    self.internal_counter += 1

                acc += (y["logit"].argmax(dim=1) == x["label"].to(device)).sum().item()
                total += x["label"].shape[0]
                print(f"Accuracy: {acc/total}", end='\r')
            print()