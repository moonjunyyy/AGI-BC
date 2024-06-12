import os
import copy
import torch
import asyncio
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from utils.utils import get_audio_model, get_language_model, all_gather
from utils.kmeans import KMeans
from layer.lora import LoRA, SelectionLoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE
import logging
import matplotlib.pyplot as plt

class LoRASelection(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(LoRASelection, self).__init__()

        self.num_lora = 16
        self.mode = mode
        self.num_classes = num_class

        self.register_module("language_model", language_model)
        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        self.register_module("audio_model", audio_model)
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()

        # self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

        for param in self.audio_model.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

        for name, module in self.audio_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                for n in range(self.num_lora):
                    _name = name.split('.')
                    _module = self.audio_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], LoRA(module, self.num_lora, 4, alpha=8))
        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                for n in range(self.num_lora):
                    _name = name.split('.')
                    _module = self.language_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], LoRA(module, self.num_lora, 4, alpha=8))

        self.audio_key = nn.Parameter(torch.randn(self.num_lora, 768))
        self.text_key = nn.Parameter(torch.randn(self.num_lora, 768))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768 * 2, num_class)

    def forward(self, x):
        # Extract the features from the audio and text
        device = self.parameters().__next__().device
        audio = x["audio"].to(device)
        text  = x["text"].to(device)
        # target_audio = x["target_audio"].to(device)
        # target_text = x["target_text"].to(device)
        y = {}
        # get audio only one channel
        audio = audio[:, 0, :]
        # target_audio = target_audio[:, 0, :]
        AB, AL = audio.shape
        TB, TL = text.shape

        audio_embed = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
        audio_embed = self.audio_model.model.feature_extractor(audio_embed)
        audio_embed = self.audio_model.model.feature_projection(audio_embed.transpose(1, 2))
        text_embed = self.language_model.embeddings(text)

        audio_query = self.audio_model.model.encoder(audio_embed)[0]
        audio_query = audio_query.mean(dim=1)
        text_query = self.language_model.encoder(text_embed)[0]
        text_query = text_query[:, 0, :]

        audio_similarity = F.cosine_similarity(audio_query.unsqueeze(1), self.audio_key.unsqueeze(0), dim=-1)
        text_similarity = F.cosine_similarity(text_query.unsqueeze(1), self.text_key.unsqueeze(0), dim=-1)

        audio_selection = audio_similarity.argmax(dim=1)
        text_selection = text_similarity.argmax(dim=1)
        for name, module in self.audio_model.named_modules():
            if isinstance(module, SelectionLoRA):
                module.set_selection(audio_selection)
        for name, module in self.language_model.named_modules():
            if isinstance(module, SelectionLoRA):
                module.set_selection(text_selection)

        audio_feature = self.audio_model.model.encoder(audio_embed)[0]
        audio_feature = audio_feature.mean(dim=1)
        text_feature = self.language_model.encoder(text_embed)[0]
        text_feature = text_feature[:, 0, :]

        for name, module in self.audio_model.named_modules():
            if isinstance(module, SelectionLoRA):
                module.set_selection(None)
        for name, module in self.language_model.named_modules():
            if isinstance(module, SelectionLoRA):
                module.set_selection(None)
        concat = torch.cat((audio_query, text_query), dim=-1)

        y["logit"] = self.classifier(self.dropout(concat))
        y["audio_key_loss"] = (1 - audio_similarity[torch.arange(AB), audio_selection].clone()).mean() + F.cosine_similarity(self.audio_key.unsqueeze(1), self.audio_key.unsqueeze(0), dim=-1).mean()
        y["text_key_loss"] = (1 - text_similarity[torch.arange(TB), text_selection].clone()).mean() + F.cosine_similarity(self.text_key.unsqueeze(1), self.text_key.unsqueeze(0), dim=-1).mean()
        return y