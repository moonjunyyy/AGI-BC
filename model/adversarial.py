import os
import copy
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

class Adversarial(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy", tokenizer=None, path=None):
        super(Adversarial, self).__init__()

        self.path = path

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

        self.audio_generator_lora = nn.ModuleDict()
        self.text_generator_lora = nn.ModuleDict()

        self.audio_discriminator_lora = nn.ModuleDict()
        self.text_discriminator_lora = nn.ModuleDict()

        self.audio_classifier_lora = nn.ModuleDict()
        self.text_classifier_lora = nn.ModuleDict()

        for name, modules in self.audio_model.named_modules():
            if isinstance(modules, nn.Linear) and\
               'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                self.audio_generator_lora[name.replace('.', '')] = LoRA(modules, 768, 64, alpha=32)
                self.audio_discriminator_lora[name.replace('.', '')] = LoRA(modules, 768, 64, alpha=32)
                self.audio_classifier_lora[name.replace('.', '')] = LoRA(modules, 768, 64, alpha=32)

        for name, modules in self.language_model.named_modules():
            if isinstance(modules, nn.Linear) and\
               'query' in name or 'key' in name or 'value' in name:
                self.text_generator_lora[name.replace('.', '')] = LoRA(modules, 768, 64, alpha=32)
                self.text_discriminator_lora[name.replace('.', '')] = LoRA(modules, 768, 64, alpha=32)
                self.text_classifier_lora[name.replace('.', '')] = LoRA(modules, 768, 64, alpha=32)

        self.lora_mode('discriminator', requires_grad=False)

        self.generator_audio_affine = nn.Linear(768, 768)
        self.generator_text_affine = nn.Linear(768, 768)
        self.audio_disriminator_head = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.text_disriminator_head = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_layer_1 = nn.Linear(768 * 2, output_size)
        self.classifier = nn.Linear(output_size, num_class)

        self._lora_mode = None
        self.tik_tok = False
        self.counter = 0

        self.g_loss = torch.empty(0)
        self.real_d_loss = torch.empty(0)
        self.fake_d_loss = torch.empty(0)

    def pretext_forward(self, x):
        self.tik_tok = not self.tik_tok
        self.counter += 1
        if self.tik_tok:
            # Discriminator Training
            text = x["text"]
            sep_pos = (text == 3).nonzero()
            text_max_len = sep_pos[:,1].max()
            generated_token_pos = sep_pos.repeat_interleave(self.predict_length, dim=0)
            generated_token_pos += torch.cat((torch.zeros_like(generated_token_pos[:, :1]), torch.arange(self.predict_length).repeat(sep_pos.shape[0]).to(text.device).unsqueeze(1)), dim=1)

            generated_audio_token_pos = torch.cat((torch.arange(sep_pos.shape[0]).repeat_interleave(24, dim=0).unsqueeze(1),
                                                   torch.arange(74 + 24)[74:].repeat(sep_pos.shape[0]).unsqueeze(1)), dim=-1).to(x["audio"].device)

            real_audio_tokens, real_text_tokens = self.real_tokens(x)
            # real_text_tokens = real_text_tokens[:, :text_max_len]
            fake_audio_tokens, fake_text_tokens = self.generate(x)

            real_audio_pred, real_text_pred = self.discriminate(real_audio_tokens, real_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)
            a_to_t_audio_pred, a_to_t_text_pred = self.discriminate(real_audio_tokens, fake_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)
            t_to_a_audio_pred, t_to_a_text_pred = self.discriminate(fake_audio_tokens, real_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)
            fake_audio_pred, fake_text_pred = self.discriminate(fake_audio_tokens, fake_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)

            real_d_loss = - a_to_t_audio_pred.log().mean() - t_to_a_text_pred.log().mean() - real_audio_pred.log().mean() - real_text_pred.log().mean()
            # real_d_loss = - real_audio_pred.log().mean() - real_text_pred.log().mean() - a_to_t_audio_pred.log().mean() - t_to_a_text_pred.log().mean()
            # fake_d_loss = - (1 - a_to_t_text_pred).log().mean() - (1 - t_to_a_audio_pred).log().mean() - (1 - fake_audio_pred).log().mean() - (1 - fake_text_pred).log().mean()
            fake_d_loss = - (1 - a_to_t_text_pred).log().mean() - (1 - t_to_a_audio_pred).log().mean() - (1 - fake_audio_pred).log().mean() - (1 - fake_text_pred).log().mean()
            # Make a log
            self.real_d_loss = torch.cat((self.real_d_loss, real_d_loss.unsqueeze(0).detach().cpu()))
            self.fake_d_loss = torch.cat((self.fake_d_loss, fake_d_loss.unsqueeze(0).detach().cpu()))

            d_loss = real_d_loss + fake_d_loss
            # print(f"RA : {real_audio_pred.mean():.6f}, RT : {real_text_pred.mean():.6f}, FA : {fake_audio_pred.mean():.6f}, FT : {fake_text_pred.mean():.6f}", end=' ')
            print(f"RA : {a_to_t_audio_pred.mean():.6f}, RT : {t_to_a_text_pred.mean():.6f}, FA : {t_to_a_audio_pred.mean():.6f}, FT : {a_to_t_text_pred.mean():.6f}", end=' ')
            print(f"D Loss: {d_loss:.6f}", end=' ')
            return d_loss
        else:
            # Generator Training
            sep_pos = (x["text"] == 3).nonzero()
            max_len = sep_pos[:,1].max()
            original_text_tokens = sep_pos.repeat_interleave(max_len, dim=0)
            original_text_tokens -= torch.cat((torch.zeros_like(original_text_tokens[:,0:1]), torch.arange(max_len).repeat(len(sep_pos)).unsqueeze(1).to(x["text"].device)), dim=1)
            original_text_tokens = original_text_tokens[original_text_tokens[:,1] > 0]
            
            generated_token_pos = sep_pos.repeat_interleave(self.predict_length, dim=0)
            generated_token_pos += torch.cat((torch.zeros_like(generated_token_pos[:, :1]), torch.arange(self.predict_length).repeat(sep_pos.shape[0]).to(x["text"].device).unsqueeze(1)), dim=1)

            generated_audio_token_pos = torch.cat((torch.arange(sep_pos.shape[0]).repeat_interleave(24, dim=0).unsqueeze(1),
                                                   torch.arange(74 + 24)[74:].repeat(sep_pos.shape[0]).unsqueeze(1)), dim=-1).to(x["audio"].device)

            real_audio_tokens, real_text_tokens = self.real_tokens(x)
            fake_audio_tokens, fake_text_tokens = self.generate(x)
            a_to_t_audio_pred, a_to_t_text_pred = self.discriminate(real_audio_tokens, fake_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)
            t_to_a_audio_pred, t_to_a_text_pred = self.discriminate(fake_audio_tokens, real_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)
            fake_audio_pred, fake_text_pred = self.discriminate(fake_audio_tokens, fake_text_tokens, audio_pos=generated_audio_token_pos, text_pos=generated_token_pos)

            # Focal Loss
            # g_loss = - ((1 - fake_audio_pred)** 2 * fake_audio_pred.log()).mean() - ((1 - fake_text_pred) ** 2 * fake_text_pred.log()).mean()
            # Normal Loss
            # g_loss = - fake_audio_pred.log().mean() - fake_text_pred.log().mean() - a_to_t_text_pred.log().mean() - t_to_a_audio_pred.log().mean()
            g_loss = - a_to_t_text_pred.log().mean() - t_to_a_audio_pred.log().mean() - fake_audio_pred.log().mean() - fake_text_pred.log().mean()
            self.g_loss = torch.cat((self.g_loss, g_loss.unsqueeze(0).detach().cpu()))

            decoded_text = self.decode_text(fake_text_tokens)
            total_text = self.tokenizer.batch_decode(torch.cat((x["text"], x["target_text"]), dim=1), skip_special_tokens=True)
            print(f"G Loss: {g_loss:.6f}, Text: {total_text[0]} -> {decoded_text[0]}")
            plt.plot(self.g_loss.numpy(), label='Generator Loss')
            plt.plot(self.real_d_loss.numpy(), label='Real Discriminator Loss')
            plt.plot(self.fake_d_loss.numpy(), label='Fake Discriminator Loss')
            total_data = torch.cat((self.g_loss, self.real_d_loss, self.fake_d_loss))
            std = total_data.std()
            mean = total_data.mean()
            ax = plt.gca()
            ax.set_ylim([None, mean + std * 3])
            plt.legend()
            plt.savefig(f"{self.path}/loss.png")
            plt.clf()

            if self.counter % 1000 == 0:
                decoded_real_audio = self.decode_audio(fake_audio_tokens[:1,:74], len(x["audio"][0,0,:]))
                decoded_fake_audio = self.decode_audio(fake_audio_tokens[:1,74:], len(x["target_audio"][0,0,:]))
                decoded_audio = torch.cat((decoded_real_audio, decoded_fake_audio), dim=1)
                torchaudio.save(f"{self.path}/audio_{self.counter}.wav", decoded_audio.detach().cpu(), 16000)
            return g_loss

    def generate(self, x : dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_text_tokens = x["text"].clone()
        real_audio_tokens, real_text_tokens = self.real_tokens(x)
        sep_pos = (original_text_tokens == 3).nonzero()
        max_len = sep_pos[:,1].max()
        original_text_token_pos = sep_pos.repeat_interleave(max_len, dim=0)
        original_text_token_pos -= torch.cat((torch.zeros_like(original_text_token_pos[:,0:1]), torch.arange(max_len).repeat(len(sep_pos)).unsqueeze(1).to(x["text"].device)), dim=1)
        original_text_token_pos = original_text_token_pos[original_text_token_pos[:,1] > 0]

        original_text_tokens = self.pad_mask_token(original_text_tokens)
        mask_token_pos = (original_text_tokens == 4).nonzero()
        original_text_tokens = self.language_model.embeddings(original_text_tokens)
        original_text_tokens[mask_token_pos[:,0], mask_token_pos[:,1]] = torch.randn(mask_token_pos.shape[0], 768, device=original_text_tokens.device)

        original_audio_tokens = x["audio"][:, 0, :]
        original_audio_tokens = self.audio_model.model.feature_extractor(original_audio_tokens)
        original_audio_tokens = self.audio_model.model.feature_projection(original_audio_tokens.transpose(1,2))
        original_audio_tokens = torch.cat((original_audio_tokens, torch.randn(original_audio_tokens.shape[0], 24, 768,device=original_audio_tokens.device)), dim=1)

        audio_tokens, text_tokens = self.unified_encoder(original_audio_tokens, original_text_tokens)
        audio_tokens = self.generator_audio_affine(audio_tokens)
        text_tokens = self.generator_text_affine(text_tokens) + original_text_tokens

        audio_tokens[:, :74] = 0
        text_tokens[original_text_token_pos[:,0], original_text_token_pos[:,1]] = 0

        audio_tokens = audio_tokens + original_audio_tokens
        text_tokens = text_tokens + original_text_tokens

        return audio_tokens, text_tokens
    
    def discriminate(self, audio_tokens : torch.Tensor, text_tokens : torch.Tensor, audio_pos=None, text_pos=None) -> Tuple[torch.Tensor, torch.Tensor]:
        #74, 788
        audio_tokens, text_tokens = self.unified_encoder(audio_tokens, text_tokens)

        # audio_pred = audio_tokens.mean(dim=1)
        # text_pred = text_tokens.mean(dim=1)

        audio_max_len = audio_tokens.shape[1]
        text_max_len = text_tokens.shape[1]

        if audio_pos is not None:
            audio_pos = audio_pos[audio_pos[:,1] < audio_max_len]
            audio_tokens = audio_tokens[audio_pos[:, 0], audio_pos[:, 1]]
        if text_pos is not None:
            text_pos = text_pos[text_pos[:,1] < text_max_len]
            text_tokens = text_tokens[text_pos[:, 0], text_pos[:, 1]]

        # audio_tokens = audio_tokens.mean(1)
        # text_tokens = text_tokens.mean(1)

        audio_pred = self.audio_disriminator_head(audio_tokens)
        text_pred = self.text_disriminator_head(text_tokens)

        return audio_pred, text_pred

    def real_tokens(self, x : dict) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = x["audio"][:, 0, :].clone()
        text = x["text"].clone()
        
        target_text = x["target_text"].clone()
        target_audio = x["target_audio"][:, 0, :].clone()

        text = self.concat_text_token(text, target_text)
        text[text==3] = 1
        text_tokens = self.language_model.embeddings(text)
        audio_tokens = self.audio_model.model.feature_extractor(audio)
        audio_tokens = self.audio_model.model.feature_projection(audio_tokens.transpose(1,2))
        # 74, 768
        target_audio_tokens = self.audio_model.model.feature_extractor(target_audio)
        target_audio_tokens = self.audio_model.model.feature_projection(target_audio_tokens.transpose(1,2))
        # 24, 768
        audio_tokens = torch.cat((audio_tokens, target_audio_tokens), dim=1)
        # self.unified_encoder(audio_tokens, text_tokens)
        return audio_tokens, text_tokens

    def lora_mode(self, mode : str, requires_grad:bool=True) -> None:
        if mode == 'generator':
            dictionary = self.audio_generator_lora
        elif mode == 'discriminator':
            dictionary = self.audio_discriminator_lora
        elif mode == 'classifier':
            dictionary = self.audio_classifier_lora
        for name, modules in self.audio_model.named_modules():
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
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
            if 'query' in name or 'key' in name or 'value' in name:
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
        optimizer = torch.optim.AdamW([synthtic_audio], lr=0.1)
        # print("\nDecoding Audio")
        for i in range(1000):
            optimizer.zero_grad()
            audio = self.audio_model.model.feature_extractor(synthtic_audio)
            audio = self.audio_model.model.feature_projection(audio.transpose(1,2))
            loss = F.mse_loss(audio, audio_tokens)
            loss.backward(retain_graph=True)
            optimizer.step()
            # print(f"{i}/{1000} loss: {loss:.6f}", end='\r')
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
                _audio_tokens = self.a2tca[layer](audio_tokens, text_tokens)
                _text_tokens = self.t2aca[layer](text_tokens, audio_tokens)
                audio_tokens = _audio_tokens
                text_tokens = _text_tokens
        return audio_tokens, text_tokens

    def forward(self, x):
        y = {}
        self.lora_mode('generator', requires_grad=False)
        self.generator_text_affine.requires_grad_(False)
        self.generator_audio_affine.requires_grad_(False)
        self.generator_audio_to_text_attention.requires_grad_(False)
        self.generator_text_to_audio_attention.requires_grad_(False)

        audio = x["audio"][:, 0, :]
        text  = x["text"]
        B, L = text.shape

        gen_audio, gen_text = self.generate(x)
        
        self.lora_mode('classifier', requires_grad=True)
        audio,text = self.unified_encoder(audio, text)

        AB, AL, _ = audio.shape
        TB, TL, _ = text.shape

        # if self.mode != "text_only":
        #     audio = self.audio_model(audio)
        # if self.mode != "audio_only":
        #     text, text2 = self.language_model(text)

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