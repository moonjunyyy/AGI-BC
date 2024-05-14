import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from utils.utils import get_audio_model, get_language_model
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class LoRA_BC(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(LoRA_BC, self).__init__()

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

        for name, param in self.audio_model.named_parameters():
            param.requires_grad = False
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False

        self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

        self.language_linear = nn.ModuleDict()
        self.language_lora = nn.ModuleDict()
        self.audio_linear = nn.ModuleDict()
        self.audio_lora = nn.ModuleDict()

        for name, module in self.audio_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                self.audio_lora[name.replace('.', '_')] = LoRA(module, 64, alpha=16)
                self.audio_linear[name.replace('.', '_')] = module

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                self.language_lora[name.replace('.', '_')] = LoRA(module, 64, alpha=16)
                self.language_linear[name.replace('.', '_')] = module

        self.dropout = nn.Dropout(dropout)
        if self.mode == "audio_only" or self.mode == "text_only":
            self.fc_layer_1 = nn.Linear(768, output_size)
        elif self.mode == "flatten":
            self.fc_layer_1 = nn.Linear(768 * 65, output_size)
        else:
            # self.fc_layer_1 = nn.Linear(794, output_size)
            self.fc_layer_1 = nn.Linear(768 + self.audio_model.get_feature_size(), num_class)
        self.relu = nn.ReLU()
        if self.mode == "hierarchical":
            self.classifier = nn.Linear(output_size, num_class - 1)
            self.BC_classifier = nn.Linear(output_size, 2)
        else:
            self.classifier = nn.Linear(output_size, num_class)
        self.BC_classifier = nn.Linear(output_size, 2)

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

    def k_means(self, x, k, max_iter=1000, batch_size=128, init="kmeans++"):
        N, D = x.shape
        if init == "random":
            c = x[torch.randperm(N)[:k]]
        elif init == "kmeans++":
            c = torch.empty(k, D, device=x.device)
            c[0] = x[torch.randint(N, (1,))]
            for i in range(1, k):
                diff = torch.cdist(c[:i], x, p=2)
                min_dist, _ = torch.min(diff, dim=0)
                farest = torch.argmax(min_dist)
                c[i] = x[farest]
        # c = c.unsqueeze(1) # (k, D)
        x = x # (N, D)
        for i in range(max_iter):
            print(f"{i+1}/{max_iter}", end='\r')
            _c = c.clone().detach()
            cluster = []
            for n in range(0, N, batch_size):
                diff = torch.cdist(c, x[n:n+batch_size], p=2)
                # diff = 1 - torch.cosine_similarity(c, x[n:n+batch_size], dim=2)
                # diff = diff.squeeze()
                _, _cluster = torch.min(diff, dim=0)
                cluster.append(_cluster)
            cluster = torch.cat(cluster, dim=0)
            for j in range(k):
                if x[cluster==j].numel() != 0:
                    c[j] = x[cluster==j].mean(dim=0).unsqueeze(0)
            if i > 0 and torch.equal(c, _c):
                break
        return c, cluster

    def forward(self, x):
        
        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape
        
        device = self.parameters().__next__().device
        # self.lora_off()
        # original_audio = audio.clone()
        # original_text = text.clone()
        
        audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.deidentifier(audio)

        text = self.language_model.embeddings(text)

        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)
        
        text = self.language_model.encoder(text).last_hidden_state
        audio = self.audio_model.model.encoder(audio).last_hidden_state

        audio = audio.reshape(AB, -1, 768).mean(dim=1)
        text = text.reshape(TB, -1, 768)[:, 0, :]

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        y["logit"] = self.fc_layer_1(self.dropout(concat))
        # y["logit"] = self.relu(y["logit"])
        # y["logit"] = self.classifier(self.dropout(y["logit"]))
        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))

        return y