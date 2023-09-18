import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torchaudio
from itertools import permutations
from util.utils import get_audio_model, get_language_model
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from layer.lora import LoRA
from typing import Tuple

class time_embedding(nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.dimension = dimension
        self.fc1 = nn.Linear(self.dimension, self.dimension)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.dimension, self.dimension)

    def forward(self, step):
        half_dim = self.dimension // 2
        embeding = torch.arange(half_dim, dtype=torch.float32, device=self.fc1.weight.device)
        embeding = embeding.unsqueeze(0).repeat(2, 1)
        embeding = embeding.transpose(0, 1).reshape(-1)
        embeding = torch.exp(-math.log(10000) * embeding / (half_dim - 1))
        embeding = step[:, None] * embeding[None, :]
        embeding = torch.cat([torch.sin(embeding[:, 0::2]), torch.cos(embeding[:, 1::2])], dim=-1)
        embeding = embeding
        embeding = self.fc1(embeding)
        embeding = self.activation(embeding)
        embeding = self.fc2(embeding)
        return embeding.unsqueeze(1)
    
class Adversarial(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy", tokenizer=None):
        super(Adversarial, self).__init__()

        self.mode = mode
        if self.mode != "audio_only":
            self.register_module("language_model", language_model)
            assert self.language_model != None
            for param in self.language_model.parameters():
                param.requires_grad = False

        if self.mode != "text_only":
            self.register_module("audio_model", audio_model)
            assert self.audio_model != None
            for param in self.audio_model.parameters():
                param.requires_grad = False

        self.tokenizer = tokenizer

        self.predict_length = 2
        self.audio_mask_token = nn.Parameter(torch.randn(1, 1, 768), requires_grad=True)

        self.generator_audio_to_text_attention = nn.ModuleList([CrossAttentionLayer(768, 8, 0.3) for _ in range(12)])
        self.generator_text_to_audio_attention = nn.ModuleList([CrossAttentionLayer(768, 8, 0.3) for _ in range(12)])

        self.discriminotor_audio_to_text_attention = nn.ModuleList([CrossAttentionLayer(768, 8, 0.3) for _ in range(12)])
        self.discriminotor_text_to_audio_attention = nn.ModuleList([CrossAttentionLayer(768, 8, 0.3) for _ in range(12)])

        self.audio_to_text_attention = nn.ModuleList([CrossAttentionLayer(768, 8, 0.3) for _ in range(12)])
        self.text_to_audio_attention = nn.ModuleList([CrossAttentionLayer(768, 8, 0.3) for _ in range(12)])
       
        self.num_steps = 100
        betas = torch.linspace(0.0001, 0.02, self.num_steps)
        self.register_buffer('betas', betas)
        alphas = 1 - self.betas
        self.register_buffer('alphas', alphas)
        alpha_bars = self.alphas.cumprod(dim=0)
        self.register_buffer('alpha_bars', alpha_bars)

        embedding_dim = 768
        class ConvBlock(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.conv1 = nn.Conv1d(embedding_dim*4, embedding_dim*4, 3, padding=1)
                self.norm1 = nn.BatchNorm1d((embedding_dim*4, ))
                self.activation = nn.GELU()
            def forward(self, x):
                return x + self.activation(self.norm1(self.conv1(x)))
        self.register_module('audio_decoder', nn.Sequential(
            nn.Conv1d(embedding_dim*2, embedding_dim*4, 1),
            *[ConvBlock() for _ in range(12)],
            nn.Conv1d(embedding_dim*4, embedding_dim, 1),
        ))
        self.text_decoder = nn.Sequential(
            nn.Conv1d(embedding_dim*2, embedding_dim*4, 1),
            *[ConvBlock() for _ in range(12)],
            nn.Conv1d(embedding_dim*4, embedding_dim, 1),
        )
        self.register_module('time_embedding', time_embedding(embedding_dim * 2))

    def pretext_forward(self, x):
        audio = x["audio"][:, 0, :]
        text  = x["text"]
        target_audio = x["target_audio"][:, 0, :]
        target_text  = x["target_text"]

        B, L = text.shape
        step = torch.randint(0, self.num_steps, (B,)).to(text.device)
        beta = self.betas[step].unsqueeze(-1).unsqueeze(-1)
        alpha = self.alphas[step].unsqueeze(-1).unsqueeze(-1)
        alpha_bar = self.alpha_bars[step].unsqueeze(-1).unsqueeze(-1)

        text_sep_pos = (text == 3).nonzero()

        text = text[:, text_sep_pos[:,1].max()+1:]
        target_text = target_text[:, 1:2+self.predict_length]

        audio_tokens = self.audio_model.model.feature_extractor(audio)
        audio_tokens = self.audio_model.model.feature_projection(audio_tokens.transpose(1,2))
        text_tokens = self.language_model.embeddings(text)

        target_audio_tokens = self.audio_model.model.feature_extractor(target_audio)
        target_audio_tokens = self.audio_model.model.feature_projection(target_audio_tokens.transpose(1,2))
        target_text_tokens = self.language_model.embeddings(target_text)

        audio_added_noise = torch.randn_like(target_audio_tokens, device=target_audio_tokens.device)
        text_added_noise = torch.randn_like(target_text_tokens, device=target_text_tokens.device)

        noise_added_audio = target_audio_tokens * alpha ** 0.5 + audio_added_noise * (1 - alpha) ** 0.5
        noise_added_text = target_text_tokens * alpha_bar ** 0.5 + text_added_noise * (1 - alpha_bar) ** 0.5

        audio_tokens = torch.cat((audio_tokens, noise_added_audio), dim=1)
        text_tokens = torch.cat((text_tokens, noise_added_text), dim=1)

        audio_tokens = self.audio_model.model.encoder(audio_tokens).last_hidden_state
        text_tokens = self.language_model.encoder(text_tokens).last_hidden_state

        audio_tokens = torch.cat((audio_tokens.mean(1, keepdim=True).repeat(1, noise_added_audio.shape[1], 1), noise_added_audio), dim=-1) + self.time_embedding(step)
        text_tokens = torch.cat((text_tokens[:, :1, :].repeat(1, noise_added_text.shape[1], 1), noise_added_text), dim=-1) + self.time_embedding(step)
        audio_tokens = self.audio_decoder(audio_tokens.transpose(1,2)).transpose(1,2)
        text_tokens = self.text_decoder(text_tokens.transpose(1,2)).transpose(1,2)

        loss = F.mse_loss(audio_tokens, noise_added_audio) + F.mse_loss(text_tokens, noise_added_text)
        return  loss

    def generate(self, x : dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            audio = x["audio"][:, 0, :]
            text = x["text"]
            target_audio = x["target_audio"][:, 0, :]
            target_text = x["target_text"]

            audio_tokens = self.audio_model.model.feature_extractor(audio)
            audio_tokens = self.audio_model.model.feature_projection(audio_tokens.transpose(1,2))
            text_tokens = self.language_model.embeddings(text)

            target_audio_tokens = self.audio_model.model.feature_extractor(target_audio)
            target_audio_tokens = self.audio_model.model.feature_projection(target_audio_tokens.transpose(1,2))
            target_text_tokens = self.language_model.embeddings(target_text)

            audio_start_noise = torch.randn_like(target_audio_tokens, device=target_audio_tokens.device)
            text_start_noise = torch.randn_like(target_text_tokens, device=text_tokens.device)
            steps = torch.arange(1, self.num_steps).flip(0).tolist()
            for step in steps:
                step = torch.tensor(step).repeat(x["audio"].shape[0]).to(x["audio"].device)

                beta = self.betas[step].unsqueeze(-1).unsqueeze(-1)
                alpha = self.alphas[step].unsqueeze(-1).unsqueeze(-1)
                alpha_bar = self.alpha_bars[step].unsqueeze(-1).unsqueeze(-1)

                audio_tokens = torch.cat((audio_tokens, audio_start_noise), dim=1)
                text_tokens = torch.cat((text_tokens, text_start_noise), dim=1)

                audio_tokens = self.audio_model.model.encoder(audio_tokens).last_hidden_state
                text_tokens = self.language_model.encoder(text_tokens).last_hidden_state

                audio_tokens = torch.cat((audio_tokens.mean(1, keepdim=True).repeat(1, audio_start_noise.shape[1], 1), audio_start_noise), dim=-1) + self.time_embedding(step)
                text_tokens = torch.cat((text_tokens[:, :1, :].repeat(1, text_start_noise.shape[1], 1), text_start_noise), dim=-1) + self.time_embedding(step)
                audio_tokens = self.audio_decoder(audio_tokens.transpose(1,2)).transpose(1,2)
                text_tokens = self.text_decoder(text_tokens.transpose(1,2)).transpose(1,2)

                audio_start_noise = (audio_start_noise - beta / ((1 - alpha_bar)**0.5) * audio_tokens) / alpha ** 0.5
                text_start_noise = (text_start_noise - beta / ((1 - alpha_bar)**0.5) * text_tokens) / alpha ** 0.5
                if (step > 0).all():
                    audio_start_noise += beta ** 0.5 * torch.randn_like(audio_start_noise)
                    text_start_noise += beta ** 0.5 * torch.randn_like(text_start_noise)
        return audio_start_noise, text_start_noise

    def real_tokens(self, x : dict) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = x["audio"][:, 0, :].clone()
        text = x["text"].clone()
        
        target_text = x["target_text"].clone()
        target_audio = x["target_audio"][:, 0, :].clone()

        text = self.concat_text_token(text, target_text)
        text_tokens = self.language_model.embeddings(text)

        # [B, 99, 768] for 2 sec
        audio = torch.cat((audio, target_audio), dim=1)
        audio_tokens = self.audio_model.model.feature_extractor(audio)
        audio_tokens = self.audio_model.model.feature_projection(audio_tokens.transpose(1,2))
        return audio_tokens, text_tokens

    def lora_mode(self, mode : str, requires_grad:bool=True) -> None:
        if mode == 'generator':
            dictionary = self.audio_generator_lora
        elif mode == 'discriminator':
            dictionary = self.audio_discriminator_lora
        elif mode == 'classifier':
            dictionary = self.audio_classifier_lora
        for name, modules in self.audio_model.named_modules():
            # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            if isinstance(modules, nn.Linear) or isinstance(modules, LoRA):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], dictionary[name.replace('.', '')])
                dictionary[name.replace('.', '')].requires_grad = requires_grad
        for k, v in dictionary.items():
            v.requires_grad_(requires_grad)

        if mode == 'generator':
            dictionary = self.text_generator_lora
        elif mode == 'discriminator':
            dictionary = self.text_discriminator_lora
        elif mode == 'classifier':
            dictionary = self.text_classifier_lora
        for name, modules in self.language_model.named_modules():
            if isinstance(modules, nn.Linear) or isinstance(modules, LoRA):
            # if 'query' in name or 'key' in name or 'value' in name:
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], dictionary[name.replace('.', '')])
                dictionary[name.replace('.', '')].requires_grad = requires_grad
        for k, v in dictionary.items():
            v.requires_grad_(requires_grad)

        if mode == 'generator':
            self.t2aca = self.generator_text_to_audio_attention
            self.a2tca = self.generator_audio_to_text_attention
        elif mode == 'discriminator':
            self.t2aca = self.discriminotor_text_to_audio_attention
            self.a2tca = self.discriminotor_audio_to_text_attention
        elif mode == 'classifier':
            self.t2aca = self.text_to_audio_attention
            self.a2tca = self.audio_to_text_attention

    def pad_mask_token(self, text:torch.Tensor) -> torch.Tensor:
        B, L = text.shape
        device = text.device
        _sep = (text == 3).nonzero()
        _msk_insert = _sep.repeat_interleave(self.predict_length, dim=0)
        _msk_insert += torch.cat((torch.zeros_like(_msk_insert[:,0:1]), torch.arange(self.predict_length).repeat(B).unsqueeze(1).to(device)), dim=1)
        if _msk_insert[:,1].max() >= L:
            text = torch.cat((text, torch.ones(B, _msk_insert[:,1].max()-L+1).to(device)), dim=1)
        B, L = text.shape
        text[_msk_insert[:,0], _msk_insert[:,1]] = 4
        return text.long()

    def concat_text_token(self, text1:torch.Tensor, text2:torch.Tensor) -> torch.Tensor:
        B1, L1 = text1.shape
        B2, L2 = text2.shape
        assert B1 == B2
        device = text1.device
        _t1_sep_pos = (text1 == 3).nonzero()
        _t2_sep_pos = (text2 == 3).nonzero()
        if _t1_sep_pos[:,1].max() + _t2_sep_pos[:,1].max() >= L1:
            text1 = torch.cat((text1, torch.ones(B1, _t1_sep_pos[:,1].max()+_t2_sep_pos[:,1].max()-L1+1).long().to(device)), dim=1)
        for b in range(B1):
            _sep1 = _t1_sep_pos[b]
            _sep2 = _t2_sep_pos[b]
            text_to_insert = text2[b, 1:_sep2[1]+1]
            text1[b, _sep1[1]:_sep1[1]+len(text_to_insert)] = text_to_insert[:len(text_to_insert)]
        return text1
    
    def split_text_token(self, text:torch.Tensor, pos:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        B, D = text.shape
        device = text.device
        _sep = (text == 3).nonzero()

        _t1_sep_pos = pos.clone()
        _t2_sep_pos = _sep.clone()

        _t2_max = (_t2_sep_pos[:,1] - _t1_sep_pos[:,1]).max()
        _t2_copy_pos = _t2_sep_pos.repeat_interleave(_t2_max, dim=0)
        _t2_copy_pos -= torch.cat((torch.zeros_like(_t2_copy_pos[:,0:1]), torch.arange(_t2_max).repeat(B, 1).to(device)), dim=1)
        _t2_insert_pos = torch.cat((torch.arange(B).repeat_interleave(_t2_max, dim=0), torch.arange(_t2_max).repeat(B)),dim=1).to(device) # First token is [CLS]
        _t2_copy_pos = _t2_copy_pos[_t2_copy_pos[:,1] > _t1_sep_pos[:,1].repeat_interleave(_t2_max)]
        _t2_insert_pos = _t2_insert_pos[_t2_copy_pos[:,1] > _t1_sep_pos[:,1].repeat_interleave(_t2_max)]

        text1 = text.clone()
        text2 = torch.ones_like(text)
        text2[:,0] = 2
        text2[_t2_insert_pos[:,0], _t2_copy_pos[:,1]] = text1[_t2_copy_pos[:,0], _t2_copy_pos[:,1]]
        text1[_t2_copy_pos[:,0], _t2_copy_pos[:,1]] = 1
        text1[_t1_sep_pos[:,0], _t1_sep_pos[:,1]] = 3

        return text1, text2

    def decode_audio(self, audio_tokens:torch.Tensor, length:int)->torch.Tensor:
        B, L, D = audio_tokens.shape
        synthtic_audio = torch.randn(B, length, requires_grad=True, device=audio_tokens.device)
        optimizer = torch.optim.Adam([synthtic_audio], lr=0.1)
        # print("\nDecoding Audio")
        self.audio_model.eval()
        for i in range(1000):
            optimizer.zero_grad()
            audio = self.audio_model.model.feature_extractor(synthtic_audio)
            audio = self.audio_model.model.feature_projection(audio.transpose(1,2))
            loss = F.mse_loss(audio, audio_tokens)
            loss.backward(retain_graph=True)
            optimizer.step()
        self.audio_model.train()
        print()
        return synthtic_audio
    
    def decode_text(self, text_tokens:torch.Tensor)->torch.Tensor:
        text_predict = (text_tokens @ self.language_model.embeddings.word_embeddings.weight.t()).argmax(dim=-1)
        text_predict = self.tokenizer.batch_decode(text_predict, skip_special_tokens=True)
        return text_predict

    def unified_encoder(self, audio_tokens:torch.Tensor, text_tokens:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        audio_tokens = self.audio_model.model.encoder.pos_conv_embed(audio_tokens)
        audio_tokens = self.audio_model.model.encoder.layer_norm(audio_tokens)
        audio_tokens = self.audio_model.model.encoder.dropout(audio_tokens)
        for layer, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            audio_tokens = a_layer(audio_tokens)[0]
            text_tokens = t_layer(text_tokens)[0]
            if layer in [10,11]:
                _audio_tokens = self.audio_to_text_attention[layer](audio_tokens, text_tokens)
                _text_tokens = self.text_to_audio_attention[layer](text_tokens, audio_tokens)
                audio_tokens = _audio_tokens
                text_tokens = _text_tokens
        return audio_tokens, text_tokens

    def forward(self, x):
        y = {}

        text = x["text"]
        audio = x["audio"][:, 0, :]

        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1,2))
        text = self.language_model.embeddings(text)

        generated_audio, generated_text = self.generate(x)

        AB = generated_audio.shape[0]
        TB = generated_text.shape[0]

        text = torch.cat((text, generated_text), dim=1)
        audio = torch.cat((audio, generated_audio), dim=1)
        audio, text = self.unified_encoder(generated_audio, generated_text)

        if self.mode == "flatten":
            audio = audio.reshape(AB, -1)
            text = text[:,0,:]
            concat = torch.cat((audio, text.reshape(TB, -1)), dim=1)
        elif self.mode == "audio_only":
            concat = audio.mean(dim=1)
        elif self.mode == "text_only":
            concat = text[:, 0, :]
        else:
            audio = audio.mean(dim=1)
            text = text[:, 0, :]
            concat = torch.cat((audio, text), dim=1)
        concat = self.fc_layer_1(self.dropout(concat))
        concat = F.relu(concat)
        y["logit"] = self.classifier(self.dropout(concat))
        
        return y