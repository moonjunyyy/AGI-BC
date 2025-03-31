import os
import sys
import time
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from itertools import permutations
from utils.utils import get_audio_model, get_language_model, all_gather
from utils.kmeans import KMeans
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from diffusers import StableDiffusion3Pipeline
from sklearn.manifold import TSNE
import logging
import matplotlib.pyplot as plt

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Diffused_Backchannel(nn.Module):
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

        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16)

        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor

        self.lora_change_list = ['to_q', 'to_k', 'to_v', 'to_out', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
        self.loras = nn.ModuleDict({
             'audio': nn.ModuleDict(),
             'text': nn.ModuleDict(),
             'video': nn.ModuleDict(),
             'audio_encoder': nn.ModuleDict(),
             'audio_decoder': nn.ModuleDict(),
             'text_encoder': nn.ModuleDict(),
             'text_decoder': nn.ModuleDict(),
             'video_encoder': nn.ModuleDict(),
             'video_decoder': nn.ModuleDict()
             })
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear) and any([change in name for change in self.lora_change_list]):
                for mod in ['audio', 'text', 'video']:
                    self.loras[mod][name.replace('.', '_')] = LoRA(module, 32, alpha=64)
        self.mode = mode
        self.num_classes = num_class
        
        for param in self.transformer.parameters():
            param.requires_grad = False
        # for param in self.vae.parameters():
        #     param.requires_grad = False
        
        if language_model is not None:
            self.register_module("language_model", language_model)
            # if bert and vocab are not provided, raise an error
            assert self.language_model is not None, "bert and vocab must be provided"
            for param in self.language_model.parameters():
                param.requires_grad = False
        if audio_model is not None:
            self.register_module("audio_model", audio_model)
            self.audio_feature_size = audio_model.get_feature_size()
            for param in self.audio_model.parameters():
                param.requires_grad = False
        if video_model is not None:
            self.register_module("video_model", video_model)
            self.video_feature_size = video_model.get_feature_size()
            for param in self.video_model.parameters():
                param.requires_grad = False

        self.audio_emb_proj_1 = nn.Linear(768, 768)
        self.text_emb_proj_1 = nn.Linear(768, 768)
        self.video_emb_proj_1 = nn.Linear(384, 768)
        self.pooled_emb_proj_1 = nn.Linear(4096, 2048)
        self.nonlinearity = Swish()
        self.audio_emb_proj_2 = nn.Linear(768, 4096)
        self.text_emb_proj_2 = nn.Linear(768, 4096)
        self.video_emb_proj_2 = nn.Linear(768, 4096)
        self.pooled_emb_proj_2 = nn.Linear(2048, 2048)

        self.betas = nn.Parameter(torch.linspace(0.0, 0.1, 1000), requires_grad=False)
        self.alphas = nn.Parameter(1 - self.betas, requires_grad=False)
        self.alpha_bars = nn.Parameter(torch.cumprod(self.alphas, 0), requires_grad=False)

        for name, module in self.audio_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                self.loras['audio_encoder'][name.replace('.', '_')] = LoRA(module, 32, alpha=64)
                self.loras['audio_decoder'][name.replace('.', '_')] = LoRA(module, 32, alpha=64)

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                self.loras['text_encoder'][name.replace('.', '_')] = LoRA(module, 32, alpha=64)
                self.loras['text_decoder'][name.replace('.', '_')] = LoRA(module, 32, alpha=64)

        for name, module in self.video_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                self.loras['video_encoder'][name.replace('.', '_')] = LoRA(module, 32, alpha=64)
                self.loras['video_decoder'][name.replace('.', '_')] = LoRA(module, 32, alpha=64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768 + self.audio_model.get_feature_size(), num_class)
        self.internal_counter = 1000
        self.size = None
        self.to(torch.bfloat16)

    def lora_on(self, mode='encoder'):
        if mode == 'encoder':
            for name, module in self.audio_model.named_modules():
                if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                    _name = name.split('.')
                    _module = self.audio_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras['audio_encoder'][name.replace('.', '_')])
            for name, module in self.language_model.named_modules():
                if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                    _name = name.split('.')
                    _module = self.language_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras['text_encoder'][name.replace('.', '_')])
            for name, module in self.video_model.named_modules():
                if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                    _name = name.split('.')
                    _module = self.video_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras['video_encoder'][name.replace('.', '_')])
        elif mode == 'decoder':
            for name, module in self.audio_model.named_modules():
                if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                    _name = name.split('.')
                    _module = self.audio_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras['audio_decoder'][name.replace('.', '_')])
            for name, module in self.language_model.named_modules():
                if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                    _name = name.split('.')
                    _module = self.language_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras['text_decoder'][name.replace('.', '_')])
            for name, module in self.video_model.named_modules():                                                                                                                                                                                                                                                                     
                if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                    _name = name.split('.')
                    _module = self.video_model
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras['video_decoder'][name.replace('.', '_')])
        else:
            for name, module in self.transformer.named_modules():
                if isinstance(module, nn.Linear) and any([change in name for change in self.lora_change_list]):
                    _name = name.split('.')
                    _module = self.transformer
                    for i in range(len(_name)-1):
                        _module = _module.__getattr__(_name[i])
                    _module.__setattr__(_name[-1], self.loras[mode][name.replace('.', '_')])

    def ddim_sampling_step(self, x_t, noise_pred, t_0, t_1):
        dimension = len(x_t.size())
        # Get alpha_bar values for t_0 and t_1
        alpha_bar_t0 = self.alpha_bars[t_0]
        for _ in range(dimension - 1):
            alpha_bar_t0 = alpha_bar_t0.unsqueeze(-1)
        alpha_bar_t1 = self.alpha_bars[t_1]
        for _ in range(dimension - 1):
            alpha_bar_t1 = alpha_bar_t1.unsqueeze(-1)
        # Estimate x_0 (the original clean image)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t0) * noise_pred) / torch.sqrt(alpha_bar_t0)
        # Compute x_{t_1} using DDIM update rule
        x_t1 = torch.sqrt(alpha_bar_t1) * x0_pred + torch.sqrt(1 - alpha_bar_t1) * noise_pred
        return x_t1

    def pretext_task(self, _):
        from dataset.ETRI_Dataset import ETRI_All_Dialog_Video_Dataset
        from utils.utils import get_language_model
        from torch.utils.data import DataLoader

        tokenizer, _ = get_language_model("koBert")
        dataset = ETRI_All_Dialog_Video_Dataset(path = "/local_datasets", train=True, tokenizer=tokenizer, length=1.5)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
                                                                  rank=dist.get_rank() if dist.is_initialized() else 0)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=8, sampler=sampler)

        self.train()
        self.lora_on('encoder')
        self.lora_on('audio')
        device = self.parameters().__next__().device
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        step_passed = 0
        time_start = time.time()
        total_steps = 10 * len(dataloader)
        for epoch in range(10):
            for i, x in enumerate(dataloader):
                step_passed += 1
                target_audio = x["target_audio"].to(device)[:, 0, :]

                optimizer.zero_grad()
                audio = x["audio"].to(torch.bfloat16).to(device)[:, 0, :]
                text = x["text"].to(device)
                video = x["video"].to(torch.bfloat16).to(device)
                # target_text = x["target_text"].to(device)
                # target_video = x["target_video"].to(device)

                B, L = audio.shape

                with torch.no_grad():
                    audio = self.audio_model.model.feature_extractor(audio)
                    audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                    audio_embedding = self.audio_model.model.encoder.pos_conv_embed(audio_embedding)
                    audio_embedding = self.audio_model.model.encoder.layer_norm(audio_embedding)
                    audio_embedding = self.audio_model.model.encoder.dropout(audio_embedding)
                    audio_embedding = self.audio_model.model.encoder(audio_embedding)[0].mean(dim=1, keepdim=True)
                    
                    text_embedding = self.language_model.embeddings(text)
                    text_embedding = self.language_model.encoder(text_embedding)[0][:, :1, :]

                    video_embedding = self.video_model.model.embeddings(video, None)
                    video_embedding = self.video_model.model.encoder(video_embedding)[0].mean(dim=1, keepdim=True)

                    target_audio = torch.stft(target_audio, n_fft=4096, hop_length=256, win_length=4096, return_complex=False).to(torch.bfloat16)
                    target_audio = (target_audio - target_audio.min()) / (target_audio.max() - target_audio.min() + 1e-6) * torch.exp(torch.tensor(1).to(target_audio.device))
                    target_audio = (target_audio + 1e-6).log()
                    target_audio = F.pad(target_audio, (0, 1), 'constant', 0)
                    target_audio = target_audio.permute(0,3,1,2)

                audio_embedding = self.audio_emb_proj_2(self.nonlinearity(self.audio_emb_proj_1(audio_embedding)))
                text_embedding = self.text_emb_proj_2(self.nonlinearity(self.text_emb_proj_1(text_embedding)))
                video_embedding = self.video_emb_proj_2(self.nonlinearity(self.video_emb_proj_1(video_embedding)))

                embedding = torch.cat((audio_embedding, text_embedding, video_embedding), dim=1)
                pooled_embedding = self.pooled_emb_proj_2(self.nonlinearity(self.pooled_emb_proj_1(embedding.mean(dim=1))))

                timesteps = torch.randint(0, 1000, (B,)).to(device)

                latent = self.vae.tiled_encode(target_audio).latent_dist.sample()
                #_target_audio = self.vae.tiled_decode(latent).sample
                noise = torch.randn_like(latent)
                noise_added_latent = torch.sqrt(1 - self.alpha_bars[timesteps].unsqueeze(1).unsqueeze(2).unsqueeze(3)) * noise + torch.sqrt(self.alpha_bars[timesteps].unsqueeze(1).unsqueeze(2).unsqueeze(3)) * latent
                noise_pred = self.transformer(noise_added_latent, embedding, pooled_embedding, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)# + F.mse_loss(_target_audio, target_audio[:, :_target_audio.size(1):, :_target_audio.size(2):, :_target_audio.size(3)])
                loss.backward()
                optimizer.step()
                time_passed = time.time() - time_start
                print(f"Epoch: {epoch}, Step: {i}/{len(dataloader)}, Loss: {loss.item()}, ETA: {time_passed / step_passed * (total_steps - step_passed)}")
                sys.stdout.flush()

    def forward(self, x):
        if self.size is None:
            with torch.no_grad():
                target_audio = x["target_audio"]
                target_audio = target_audio[:, 0, :]
                print(target_audio.shape)
                target_audio = torch.stft(target_audio, n_fft=4096, hop_length=256, win_length=4096, return_complex=False).to(torch.bfloat16)
                ftarget_audio = F.pad(ftarget_audio, (0, 1), 'constant', 0)
                print(ftarget_audio.shape)
                # make heatmap of the spectrogram
                faudio_normalized = torch.log(ftarget_audio.abs() + 1e-6).permute(0,3,1,2)[0]
                faudio_normalized = (faudio_normalized - faudio_normalized.min()) / (faudio_normalized.max() - faudio_normalized.min())
                plt.plot(target_audio[0].cpu().float())
                plt.savefig("target_audio.png")
                plt.close()

                plt.pcolor(faudio_normalized.cpu().float()[0])
                plt.savefig("ftarget_audio.png")
                plt.close()
                ftarget_audio = ftarget_audio.permute(0,3,1,2)
                self.size = self.vae.tiled_encode(ftarget_audio).latent_dist.sample().shape[1:]
                self.f_size = ftarget_audio.shape[1:]

        # Extract the features from the audio and text
        device = self.parameters().__next__().device
        audio = x["audio"].to(torch.bfloat16)
        text  = x["text"]
        video = x["video"].to(torch.bfloat16)
        y = {}
        # get audio only one channel
        audio = audio[:, 0, :]
        AB, AL = audio.shape
        TB, TL = text.shape

        with torch.no_grad():
            self.lora_on('encoder')
            audio = self.audio_model.model.feature_extractor(audio)
            audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
            text_embedding = self.language_model.embeddings(text)
            video_embedding = self.video_model.model.embeddings(video, None)

            audio_embedding = self.audio_model.model.encoder.pos_conv_embed(audio_embedding)
            audio_embedding = self.audio_model.model.encoder.layer_norm(audio_embedding)
            audio_embedding = self.audio_model.model.encoder.dropout(audio_embedding)
            audio_embedding = self.audio_model.model.encoder(audio_embedding)[0].mean(dim=1, keepdim=True)
            text_embedding = self.language_model.encoder(text_embedding)[0][:, :1, :]
            video_embedding = self.video_model.model.encoder(video_embedding)[0].mean(dim=1, keepdim=True)

            audio_embedding = self.audio_emb_proj_2(self.nonlinearity(self.audio_emb_proj_1(audio_embedding)))
            text_embedding = self.text_emb_proj_2(self.nonlinearity(self.text_emb_proj_1(text_embedding)))
            video_embedding = self.video_emb_proj_2(self.nonlinearity(self.video_emb_proj_1(video_embedding)))

            embedding = torch.cat((audio_embedding, text_embedding, video_embedding), dim=1)
            pooled_embedding = self.pooled_emb_proj_2(self.nonlinearity(self.pooled_emb_proj_1(embedding.mean(dim=1))))

            timesteps = torch.randint(1, 1000, (AB, 100)).to(device)
            timesteps = timesteps.sort(dim=1, descending=True).values
            target_audio = torch.randn(AB, *self.size).to(device).to(torch.bfloat16)
            # DDIM Sampling
            for i in range(100):
                timestep_0 = timesteps[:, i]
                timestep_1 = timesteps[:, i + 1] if i < 99 else torch.zeros_like(timestep_0)
                pred_noise = self.transformer(target_audio, embedding, pooled_embedding, timesteps[:, i]).sample
                target_audio = self.ddim_sampling_step(target_audio, pred_noise, timestep_0, timestep_1)
            target_audio = self.vae.tiled_decode(target_audio).sample
            # target_audio_normalized = torch.log(target_audio.abs() + 1e-6)[0,0]
            # make heatmap of the spectrogram
            # plt.pcolor(target_audio_normalized.cpu().float())
            print(target_audio.shape)
            plt.pcolor(target_audio.cpu().float()[0][0])
            plt.savefig("fgenerated_audio.png")
            plt.close()
            # Change target_audio into the complex number
            # target_audio = F.pad(target_audio, (0, 0, 1, 0), 'constant', 0)
            target_audio = target_audio.exp()
            target_audio = (target_audio - target_audio.max() * 0.5)
            target_audio = target_audio.to(torch.float32)
            target_real = target_audio[:, 0]
            target_imag = target_audio[:, 1]
            target_audio = torch.complex(target_real, target_imag)
            target_audio = torch.istft(target_audio, n_fft=4096, hop_length=256, win_length=4096, return_complex=False)
            print(target_audio.shape)
            plt.plot(target_audio[0].cpu().float())
            plt.savefig("generated_audio.png")
            plt.close()
            exit()

        if self.training:
            with torch.no_grad():
                gt_audio = torch.cat((audio_embedding, target_audio + self.target_audio_embedding), dim=1)
                gt_audio = self.audio_model.model.encoder.pos_conv_embed(gt_audio)
                gt_audio = self.audio_model.model.encoder.layer_norm(gt_audio)
                gt_audio = self.audio_model.model.encoder.dropout(gt_audio)
                for i, layer in enumerate(self.audio_model.model.encoder.layers):
                    # gt_audio = self.audio_time_cross_attn(gt_audio, self.audio_time_embed(torch.zeros_like(random_steps).unsqueeze(1)))
                    gt_audio = self.audio_adaptors[i](gt_audio)
                    gt_audio = layer(gt_audio)[0]

            target_audio = torch.sqrt(1 - self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * torch.randn_like(target_audio) + \
                        torch.sqrt(self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * target_audio + self.target_audio_embedding
            
            audio = torch.cat((audio_embedding, target_audio), dim=1)
            audio = self.audio_model.model.encoder.pos_conv_embed(audio)
            audio = self.audio_model.model.encoder.layer_norm(audio)
            audio = self.audio_model.model.encoder.dropout(audio)
            for i, layer in enumerate(self.audio_model.model.encoder.layers):
                # audio = self.audio_time_cross_attn(audio, self.audio_time_embed(random_steps.unsqueeze(1)))
                audio = self.audio_adaptors[i](audio)
                audio = layer(audio)[0]

            with torch.no_grad():
                gt_text = torch.cat((text_embedding, target_text_embedding + self.target_text_embedding), dim=1)
                for i, layer in enumerate(self.language_model.encoder.layer):
                    # gt_text = self.text_time_cross_attn(gt_text, self.text_time_embed(torch.zeros_like(random_steps).unsqueeze(1)))
                    gt_text = self.text_adaptors[i](gt_text)
                    gt_text = layer(gt_text)[0]

            target_text_embedding = torch.sqrt(1 - self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * torch.randn_like(target_text_embedding) + \
                        torch.sqrt(self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * target_text_embedding + self.target_text_embedding

            # with torch.no_grad():
            text = torch.cat((text_embedding, target_text_embedding), dim=1)
            for i, layer in enumerate(self.language_model.encoder.layer):
                # text = self.text_time_cross_attn(text, self.text_time_embed(random_steps.unsqueeze(1)))
                text = self.text_adaptors[i](text)
                text = layer(text)[0]

            y["audio"] = F.mse_loss(audio, gt_audio, reduction='mean')
            y["text"] = F.mse_loss(text, gt_text, reduction='mean')

            concat = torch.cat((audio.mean(dim=1), text[:, 0, :]), dim=1)
            y["logit"] = self.classifier(self.dropout(concat))
 
        else:
            audio = torch.cat((audio_embedding, torch.randn_like(target_audio) + self.target_audio_embedding), dim=1)
            for i, layer in enumerate(self.audio_model.model.encoder.layers):
                # audio = self.audio_time_cross_attn(audio, self.audio_time_embed(torch.ones_like(random_steps).unsqueeze(1) * 999))
                audio = self.audio_adaptors[i](audio)
                audio = layer(audio)[0]

            text = torch.cat((text_embedding, torch.randn_like(target_text_embedding) + self.target_text_embedding), dim=1)
            for i, layer in enumerate(self.language_model.encoder.layer):
                # text = self.text_time_cross_attn(text, self.text_time_embed(torch.ones_like(random_steps).unsqueeze(1) * 999))
                text = self.text_adaptors[i](text)
                text = layer(text)[0]

            audio = audio.mean(dim=1)
            text = text[:, 0, :]

            if self.mode == "audio_only" or self.mode == "text_only":
                concat = audio if self.mode == "audio_only" else text
            else :
                concat = torch.cat((audio, text), dim=1)
            y["logit"] = self.classifier(concat)

        if self.internal_counter != 1000:
            self.internal_counter += 1
        return y