import sys
import math
import time
import torch
import torch.amp
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from utils.contrastive_loss import NormSoftmaxLoss
from M00NNY_Utils.warmup_cosine_anneling import WarmUpCosineAnnelingScheduler
from M00NNY_Utils.warmup_constant import WarmUpConstantScheduler
from M00NNY_Utils.progress_bar import ProgressBar
from M00NNY_Utils.lora import apply_lora

class ResidualSequantial(nn.Module):
    def __init__(self, *args):
        super(ResidualSequantial, self).__init__()
        self.layers = nn.Sequential(*args)
    def forward(self, x):
        return x + self.layers(x)

class Diffused_Backchannel(nn.Module):
    def __init__(self, language_model=None, audio_model=None, video_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Diffused_Backchannel, self).__init__()
        
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")
        self.register_module("proj_model", pipe.projection_model)
        self.register_module("diffusion", pipe.transformer)
        self.register_module("vae", pipe.vae)
        self.scheduler = pipe.scheduler
        # FrozenDict([('sigma_min', 0.3), ('sigma_max', 500), ('sigma_data', 1.0), ('sigma_schedule', 'exponential'),
        #             ('num_train_timesteps', 1000), ('solver_order', 2), ('prediction_type', 'v_prediction'), ('rho', 7.0), 
        #             ('solver_type', 'midpoint'), ('lower_order_final', True), ('euler_at_final', False), ('final_sigmas_type', 'zero'),
        #             ('_class_name', 'CosineDPMSolverMultistepScheduler'), ('_diffusers_version', '0.30.0.dev0')])

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

        # Freeze the parameters of the models
        for name, param in self.audio_model.named_parameters():    param.requires_grad = False
        for name, param in self.language_model.named_parameters(): param.requires_grad = False
        for name, param in self.video_model.named_parameters():    param.requires_grad = False
        for name, param in self.diffusion.named_parameters():      param.requires_grad = False
        for name, param in self.proj_model.named_parameters():     param.requires_grad = False
        for name, param in self.vae.named_parameters():            param.requires_grad = False
        
        # Define the cross-attention layers and classifiers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.contrastive_loss = NormSoftmaxLoss(temperature=0.05)
        self.fc_layer_1 = nn.Linear(192*3, output_size)
        self.classifier = nn.Linear(output_size, num_class)

        self.cross_attn0 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(12)])
        self.cross_attn1 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(12)])
        self.cross_attn2 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(12)])
        self.audio_downproject = nn.Linear(768, 192)
        self.text_downproject  = nn.Linear(768, 192)
        self.video_downproject = nn.Linear(384, 192)

        # Define the LoRA layers
        self.audio_proj = ResidualSequantial(nn.Linear(768, 1536), nn.SiLU(), nn.Linear(1536, 768), nn.LayerNorm(768))
        self.text_proj  = ResidualSequantial(nn.Linear(768, 1536), nn.SiLU(), nn.Linear(1536, 768), nn.LayerNorm(768))
        self.video_proj = nn.Sequential(nn.Linear(384,768), ResidualSequantial(nn.Linear(768, 1536), nn.SiLU(), nn.Linear(1536, 768), nn.LayerNorm(768)))

        self.audio_lora     = apply_lora(self.audio_model,    rank=32, alpha=64, module_names=['q_proj', 'k_proj', 'v_proj', 'out_proj'], lora_names=['encoder', 'decoder'])
        self.language_lora  = apply_lora(self.language_model, rank=32, alpha=64, module_names=['query', 'key', 'value', 'output.dense'],  lora_names=['encoder', 'decoder'])
        self.video_lora     = apply_lora(self.video_model,    rank=32, alpha=64, module_names=['query', 'key', 'value', 'output.dense'],  lora_names=['encoder', 'decoder'])
        self.diffusion_lora = apply_lora(self.diffusion,      rank=32, alpha=64, module_names=['to_q', 'to_k', 'to_v', 'to_out.0'])

        self.pretext_epochs = 10
        self.sample_steps = 32

        self.cfg_scale   = 3.0

        self.empty_text  = None
        self.empty_audio = None
        self.empty_video = None
        self.empty_text_attention_mask = None

        self.empty_audio_embedding = None
        self.empty_text_embedding  = None
        self.empty_video_embedding = None

    @ torch.no_grad()
    def _attention_mask_from_2d_to_4d(self, attention_mask:torch.Tensor, num_heads:int, dtype:torch.device=torch.float32):
        _B, _L = attention_mask.shape
        _fmin = torch.finfo(dtype).min
        attention_mask = torch.where(attention_mask==0, _fmin, 0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, num_heads, _L, -1).to(dtype) # B, H, L, L, mask the attention value of 0
        return attention_mask

    def lora_mode(self, mode):
        self.audio_lora.   activate(mode)
        self.language_lora.activate(mode)
        self.video_lora.   activate(mode)

    def pretext_task(self, _):
        self.lora_mode('encoder')
        from dataset.ETRI_Dataset import ETRI_All_Dialog_Video_Dataset
        from utils.utils import get_language_model
        from torch.utils.data import DataLoader

        tokenizer, _ = get_language_model("koBert")
        dataset = ETRI_All_Dialog_Video_Dataset(path = "/local_datasets", train=True, tokenizer=tokenizer, length=3, predict_length=3)
        if dist.is_initialized(): sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        else: sampler = None
        dataloader = DataLoader(dataset, batch_size=12, num_workers=8, sampler=sampler)
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        # lrscheduler = WarmUpConstantScheduler(optimizer, warmup_steps=len(dataloader) / 2)
        lrscheduler = WarmUpCosineAnnelingScheduler(optimizer, warmup_steps=len(dataloader), t_total=len(dataloader) * self.pretext_epochs)

        sigma_min = self.scheduler.config.sigma_min
        sigma_max = self.scheduler.config.sigma_max
        t_min = math.atan(sigma_min) / math.pi * 2
        t_max = math.atan(sigma_max) / math.pi * 2
        self.scheduler.config.sigma_min = math.tan(t_min * math.pi / 2) + 1e-6 # To avoid numerical instability at the boundaries for the BrownianSampling.
        self.scheduler.config.sigma_max = math.tan(t_max * math.pi / 2) - 1e-6 # To avoid numerical instability at the boundaries for the BrownianSampling.

        start_time = time.time()
        total_steps = len(dataloader) * self.pretext_epochs        
        scaler = torch.amp.GradScaler('cuda')
        for e in range(self.pretext_epochs):
            if dist.is_initialized(): sampler.set_epoch(e)
            processed = 0
            total_loss = 0
            pbar = ProgressBar(dataloader)
            for i, data in enumerate(pbar):
                for k in data.keys():
                    data[k] = data[k].cuda()
                with torch.no_grad():
                    audio = data["audio"].clone()
                    text  = data["text"].clone()
                    video = data["video"].clone()
                    target_audio = data["target_audio"]
                    text_attention_mask = data["text_attention_mask"].clone()
                    audio = audio[:, 0, :]
                    AB, AL = audio.shape
                    TB, TL = text.shape

                    audio = self.audio_model.model.feature_extractor(audio)                          # B, 512,  74
                    audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))         # B,  74, 768
                    text  = self.language_model.embeddings(text)                                     # B, L, 768
                    video = self.video_model.model.embeddings(video, None)                           # B, L, 768

                    target_audio = torchaudio.transforms.Resample(16000, 44100)(target_audio.cpu() ).cuda().repeat(1, 2, 1)
                    prior_audio  = torchaudio.transforms.Resample(16000, 44100)(data["audio"].cpu()).cuda().repeat(1, 2, 1)
                    total_audio = torch.cat((prior_audio, target_audio), dim=-1)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    with torch.no_grad():
                        target_audio_latent = self.vae.encode(total_audio).latent_dist.sample()
                        if  self.empty_text is None: 
                            self.empty_text = torch.zeros_like(text[:1])
                            self.empty_text[:, 0] = 2
                            self.empty_text[:, 1] = 3
                            self.empty_text_attention_mask = torch.zeros_like(text_attention_mask[:1])
                            self.empty_text_attention_mask[:, :2] = 1
                        if  self.empty_audio is None:
                            self.empty_audio = torch.zeros_like(audio[:1])
                        if  self.empty_video is None:
                            self.empty_video = torch.zeros_like(video[:1])

                        random_masking = torch.randint(0, 8, (AB,)).to(audio.device)
                         # 0: all,        1: mask audio,           2: mask text,           3: mask audio and text,
                         # 4: mask video, 5: mask audio and video, 6: mask text and video, 7: mask all
                        mask_audio     = (random_masking // 4)      == 1
                        mask_text      = (random_masking %  4) // 2 == 1
                        mask_video     = (random_masking %  2)      == 1

                        audio[mask_audio] = self.empty_audio
                        video[mask_video] = self.empty_video
                        text[mask_text]   = self.empty_text
                        text_attention_mask[mask_text] = self.empty_text_attention_mask
                        text_attention_mask = self._attention_mask_from_2d_to_4d(text_attention_mask, 12, audio.dtype)

                        _timesteps = torch.rand(AB) * (t_max - t_min) + t_min
                        _sigmas    = torch.tan(_timesteps * math.pi / 2)
                        _epsilon   = torch.randn_like(target_audio_latent)
                        _sigmas    = _sigmas.to(target_audio_latent.device).unsqueeze(1).unsqueeze(2)
                        # sigma_data = 1
                        # sigma_t = tan(t * pi / 2)
                        # t = 2 * atan(sigma_t) / pi

                        # x_t    = sigma_data * x_0 + sigma_t * epsilon
                        # c_in   = sigma_data ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                        # c_out  =    sigma_t ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                        # c_skip = sigma_data ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                        c_in   =       1 / (1 + _sigmas ** 2) ** 0.5
                        c_out  = _sigmas / (1 + _sigmas ** 2) ** 0.5 # sigma_data = 1
                        c_skip =       1 / (1 + _sigmas ** 2) ** 0.5 # sigma_data = 1
                        #            x_0 =  c_skip * x_t - c_out * v_pred
                        # c_out * v_pred =  c_skip * x_t - x_0
                        #         v_pred = (c_skip * x_t - x_0) / c_out
                        latent             = (target_audio_latent + _epsilon * _sigmas)
                        velocity           = (c_skip * latent - target_audio_latent) / c_out
                        model_input_latent = latent * c_in

                        projection_output = self.proj_model(
                            start_seconds=torch.zeros(AB, 1).to(audio.device),
                            end_seconds  =torch.ones (AB, 1).to(audio.device) * 6,
                        )
                        seconds_start_hidden_states = projection_output.seconds_start_hidden_states
                        seconds_end_hidden_states   = projection_output.seconds_end_hidden_states

                    audio = self.audio_model.model.encoder(audio)[0]                                 # B,  74, 768
                    text  = self.language_model.encoder(text, attention_mask=text_attention_mask)[0] # B, L, 768
                    video = self.video_model.model.encoder(video)[0]

                    audio_feature = self.audio_proj(audio)
                    text_feature  = self.text_proj(text)
                    video_feature = self.video_proj(video)
                    features = torch.cat((audio_feature, text_feature, video_feature), dim=1)

                    text_audio_duration_embeds = torch.cat((features, seconds_start_hidden_states, seconds_end_hidden_states), dim=1)
                    audio_duration_embeds      = torch.cat((seconds_start_hidden_states, seconds_end_hidden_states), dim=2)
                    rotary_embedding = get_1d_rotary_pos_embed(
                        self.diffusion.config.attention_head_dim // 2,
                        target_audio_latent.shape[2] + audio_duration_embeds.shape[1],
                        use_real=True,
                        repeat_interleave_real=False
                        )
                    velocity_pred = self.diffusion(
                        hidden_states=model_input_latent,
                        timestep=_timesteps.to(model_input_latent.device),
                        encoder_hidden_states=text_audio_duration_embeds,
                        global_hidden_states=audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                        )[0]
                    loss = F.mse_loss(velocity_pred, velocity)
                total_loss += loss.item()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() 
                lrscheduler.step()
                processed += 1
                eta = (time.time() - start_time) / processed * (total_steps - processed)
                pbar.set_descriptions({"Loss": f"{total_loss / (i + 1):.4f} ({loss.item():.4f})", "ETA": f"{pbar.seconds_to_hms(eta)}"})
            with torch.no_grad():
                self.scheduler.set_timesteps(self.sample_steps)
                # Conditions and Empty Conditions for CFG
                audio  = data["audio"]
                video  = data["video"]
                text   = data["text"]
                text_attention_mask = data["text_attention_mask"]
                audio  = audio[:, 0, :]
                AB, AL = audio.shape
                TB, TL = text.shape

                empty_audio = torch.zeros_like(audio)
                empty_video = torch.zeros_like(video)
                empty_text  = torch.zeros_like(text)
                empty_text_attention_mask = torch.zeros_like(text_attention_mask)
                empty_text[:, 0] = 2; empty_text[:, 1] = 3
                empty_text_attention_mask[:, :2] = 1

                text_attention_mask = self._attention_mask_from_2d_to_4d(text_attention_mask, 12, audio.dtype)
                empty_text_attention_mask = self._attention_mask_from_2d_to_4d(empty_text_attention_mask, 12, audio.dtype)

                target_audio = torchaudio.transforms.Resample(16000, 44100)(data["target_audio"].cpu()).cuda().repeat(1, 2, 1)
                prior_audio  = torchaudio.transforms.Resample(16000, 44100)(data["audio"].cpu()       ).cuda().repeat(1, 2, 1)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    audio = self.audio_model.model.feature_extractor(audio)                                            # B, 512,  74
                    audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))                           # B,  74, 768
                    audio = self.audio_model.model.encoder(audio)[0]                                                   # B,   1, 768 

                    text  = self.language_model.embeddings(text)                                                       # B, 74, 768
                    text  = self.language_model.encoder(text, attention_mask=text_attention_mask)[0]                   # B,  1, 768

                    video = self.video_model.model.embeddings(video, None)                                             # B, 74, 768
                    video = self.video_model.model.encoder(video)[0]                                                   # B,  1, 768

                    empty_audio = self.audio_model.model.feature_extractor(empty_audio)                                # B, 512,  74
                    empty_audio = self.audio_model.model.feature_projection(empty_audio.transpose(1, 2))               # B,  74, 768
                    empty_audio = self.audio_model.model.encoder(empty_audio)[0]                                       # B,   1, 768
                    
                    empty_text  = self.language_model.embeddings(empty_text)                                           # B, 74, 768
                    empty_text  = self.language_model.encoder(empty_text, attention_mask=empty_text_attention_mask)[0] # B,  1, 768

                    empty_video = self.video_model.model.embeddings(empty_video, None)                                 # B, 74, 768
                    empty_video = self.video_model.model.encoder(empty_video)[0]                    

                    audio_feature = self.audio_proj(audio)
                    text_feature  = self.text_proj(text)
                    video_feature = self.video_proj(video)
                    features = torch.cat((audio_feature, text_feature, video_feature), dim=1)

                    empty_audio_feature = self.audio_proj(empty_audio)
                    empty_text_feature  = self.text_proj(empty_text)
                    empty_video_feature = self.video_proj(empty_video)
                    empty_features = torch.cat((empty_audio_feature, empty_text_feature, empty_video_feature), dim=1)
                    
                    # Embedding
                    start_time_in_seconds = torch.zeros(AB, 1).to(audio.device)
                    end_time_in_seconds   = torch.ones (AB, 1).to(audio.device) * 6
                    projection_output = self.proj_model(
                        start_seconds=start_time_in_seconds,
                        end_seconds=end_time_in_seconds,
                    )
                    seconds_start_hidden_states = projection_output.seconds_start_hidden_states
                    seconds_end_hidden_states   = projection_output.seconds_end_hidden_states
                    text_audio_duration_embeds  = torch.cat((features, seconds_start_hidden_states, seconds_end_hidden_states), dim=1)
                    audio_duration_embeds       = torch.cat((seconds_start_hidden_states, seconds_end_hidden_states), dim=2)
                    empty_text_audio_duration_embeds = torch.cat((empty_features, seconds_start_hidden_states, seconds_end_hidden_states), dim=1)
                    empty_audio_duration_embeds      = torch.cat((seconds_start_hidden_states, seconds_end_hidden_states), dim=2)
                    text_audio_duration_embeds = torch.cat((text_audio_duration_embeds, empty_text_audio_duration_embeds), dim=0)
                    audio_duration_embeds      = torch.cat((audio_duration_embeds, empty_audio_duration_embeds), dim=0)
                    
                    # Prepare Latents
                    timesteps = self.scheduler.timesteps
                    total_audio = torch.cat((prior_audio, target_audio), dim=-1)
                    target_audio_latent = self.vae.encode(total_audio).latent_dist.sample() # Ground Truth Latent
                    latent_length = int(prior_audio.shape[2] / total_audio.shape[2] * target_audio_latent.shape[2])
                    # Random Latent, x_t = sigma_data * x_0 + sigma_t * epsilon
                    latents = torch.rand_like(target_audio_latent) * (self.scheduler.config.sigma_max ** 2 + 1) ** 0.5 
                    for i, t in enumerate(timesteps):
                        # Get the real velocity from the latents
                        #   sigma_data = 1, sigma_t = tan(t * pi / 2), t = 2 * atan(sigma_t) / pi
                        #   x_t = sigma_data * x_0 + sigma_t * epsilon
                        #   c_out = sigma_t ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                        #   c_skip = sigma_data ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                        #   x_0 = c_skip * x_t - c_out * v_pred
                        #   c_out * v_pred = c_skip * x_t - x_0
                        #   v_pred = (c_skip * x_t - x_0) / c_out
                        _sigmas  = self.scheduler.sigmas[i]
                        c_in     =       1 / (1 + _sigmas ** 2) ** 0.5
                        c_out    = _sigmas / (1 + _sigmas ** 2) ** 0.5 # sigma_data = 1
                        c_skip   =       1 / (1 + _sigmas ** 2) ** 0.5 # sigma_data = 1
                        velocity = (c_skip * latents - target_audio_latent) / c_out

                        # Scaling
                        model_intput_latents = self.scheduler.scale_model_input(latents, t)
                        # Augmentation for CFG scaling
                        model_intput_latents = torch.cat((model_intput_latents, model_intput_latents), dim=0)

                        # Positional Encoding
                        rotary_embedding = get_1d_rotary_pos_embed(
                            self.diffusion.config.attention_head_dim // 2,
                            model_intput_latents.shape[2] + audio_duration_embeds.shape[1],
                            use_real=True,
                            repeat_interleave_real=False,
                        )

                        # Prediction
                        velocity_pred = self.diffusion(
                            hidden_states=model_intput_latents,
                            timestep=torch.full((AB*2,), t, device=model_intput_latents.device),
                            encoder_hidden_states=text_audio_duration_embeds,
                            global_hidden_states=audio_duration_embeds,
                            rotary_embedding=rotary_embedding,
                            return_dict=False,
                        )[0]

                        # CFG scaling
                        velocity_pred, empty_velocity_pred = velocity_pred[:AB], velocity_pred[AB:]
                        velocity_pred = empty_velocity_pred + self.cfg_scale * (velocity_pred - empty_velocity_pred)

                        # Modify the prediction with the real velocity for inpainting
                        velocity_pred[:, :, :latent_length] = velocity[:, :, :latent_length]

                        # Sampling
                        latents = self.scheduler.step(velocity_pred, t, latents).prev_sample
                audio_pred   = self.vae.decode(latents).sample
                total_audio  = self.vae.decode(target_audio_latent).sample
                audio_pred   = torchaudio.transforms.Resample(44100, 16000)(audio_pred.cpu().float()).cuda()
                target_audio = torchaudio.transforms.Resample(44100, 16000)(total_audio.cpu().float()).cuda()

                target_audio = target_audio[:, :1, :]
                audio_pred = audio_pred[:, :1, :]

                import matplotlib.pyplot as plt
                for i, (t_a, p_a) in enumerate(zip(target_audio, audio_pred)):
                    torchaudio.save(f"audio_{e}_{i}.wav", p_a.cpu().float(), 16000)
                    torchaudio.save(f"audio_{e}_{i}_target.wav", t_a.cpu().float(), 16000)
                    plt.figure(figsize=(10, 5))
                    plt.subplot(2, 1, 1)
                    plt.plot(t_a[0].cpu().numpy())
                    plt.subplot(2, 1, 2)
                    plt.plot(p_a[0].cpu().numpy())
                    plt.savefig(f"audio_{e}_{i}.png")
        with torch.no_grad():
            empty_audio = self.empty_audio
            empty_text  = self.empty_text
            empty_video = self.empty_video
            empty_text_attention_mask = self.empty_text_attention_mask

            audio = self.audio_model.model.feature_extractor(empty_audio)                          # B, 512,  74
            audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))               # B,  74, 768
            audio = self.audio_model.model.encoder(audio)[0]                                       # B,   1, 768

            text  = self.language_model.embeddings(empty_text)                                     # B,  74, 768
            text  = self.language_model.encoder(text, attention_mask=empty_text_attention_mask)[0] # B,   1, 768

            video = self.video_model.model.embeddings(empty_video, None)                           # B,  74, 768
            video = self.video_model.model.encoder(video)[0]

            self.empty_audio_embedding = self.audio_proj(audio)
            self.empty_text_embedding  = self.text_proj(text)
            self.empty_video_embedding = self.video_proj(video)

    def forward(self, x, train="train"):
        self.lora_mode('encoder')
        y = {}
        with torch.no_grad():
            self.scheduler.set_timesteps(self.sample_steps)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                # Conditions and Empty Conditions for CFG
                audio  = x["audio"]
                text   = x["text"]
                video  = x["video"]
                AB, AL = audio.shape
                TB, TL = text.shape
                audio  = audio[:, 0, :]
                text_attention_mask = x["text_attention_mask"]
                text_attention_mask = self._attention_mask_from_2d_to_4d(text_attention_mask, 12, audio.dtype)

                target_audio = x["target_audio"]
                target_audio = torchaudio.transforms.Resample(16000, 44100)(target_audio.cpu()).cuda().repeat(1, 2, 1)
                prior_audio  = torchaudio.transforms.Resample(16000, 44100)(x["audio"].cpu()).cuda().repeat(1, 2, 1)

            audio = self.audio_model.model.feature_extractor(audio)                                            # B, 512,  74
            audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))                           # B,  74, 768
            audio = self.audio_model.model.encoder(audio)[0]                                                   # B,   1, 768 

            text  = self.language_model.embeddings(text)                                                       # B,  74, 768
            text  = self.language_model.encoder(text, attention_mask=text_attention_mask)[0]                   # B,   1, 768

            video = self.video_model.model.embeddings(video, None)                                             # B,  74, 768
            video = self.video_model.model.encoder(video)[0]                                                   # B,   1, 768

            audio_feature = self.audio_proj(audio)
            text_feature  = self.text_proj(text)
            video_feature = self.video_proj(video)
            features = torch.cat((audio_feature, text_feature, video_feature), dim=1)
            empty_features = torch.cat((self.empty_audio_embedding, self.empty_text_embedding, self.empty_video_embedding), dim=1)

            # Embedding
            start_time_in_seconds = torch.zeros(AB, 1).to(audio.device)
            end_time_in_seconds = torch.ones(AB, 1).to(audio.device) * 3.5
            projection_output = self.proj_model(
                start_seconds=start_time_in_seconds,
                end_seconds=end_time_in_seconds,
            )
            seconds_start_hidden_states = projection_output.seconds_start_hidden_states
            seconds_end_hidden_states = projection_output.seconds_end_hidden_states
            text_audio_duration_embeds = torch.cat((features, seconds_start_hidden_states, seconds_end_hidden_states), dim=1)
            audio_duration_embeds = torch.cat((seconds_start_hidden_states, seconds_end_hidden_states), dim=2)
            empty_text_audio_duration_embeds = torch.cat((empty_features, seconds_start_hidden_states, seconds_end_hidden_states), dim=1)
            empty_audio_duration_embeds = torch.cat((seconds_start_hidden_states, seconds_end_hidden_states), dim=2)
            text_audio_duration_embeds = torch.cat((text_audio_duration_embeds, empty_text_audio_duration_embeds), dim=0)
            audio_duration_embeds = torch.cat((audio_duration_embeds, empty_audio_duration_embeds), dim=0)
                    
            # Prepare Latents
            timesteps = self.scheduler.timesteps
            total_audio = torch.cat((prior_audio, target_audio), dim=-1)
            target_audio_latent = self.vae.encode(total_audio).latent_dist.sample() # Ground Truth Latent
            latent_length = int(prior_audio.shape[2] / total_audio.shape[2] * target_audio_latent.shape[2])
            # Random Latent, x_t = sigma_data * x_0 + sigma_t * epsilon
            latents = torch.rand_like(target_audio_latent) * (self.scheduler.config.sigma_max ** 2 + 1) ** 0.5 
            for i, t in enumerate(timesteps):
                # Get the real velocity from the latents
                #   sigma_data = 1, sigma_t = tan(t * pi / 2), t = 2 * atan(sigma_t) / pi
                #   x_t = sigma_data * x_0 + sigma_t * epsilon
                #   c_out = sigma_t ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                #   c_skip = sigma_data ** 2 / (sigma_data ** 2 + sigma_t ** 2) ** 0.5
                #   x_0 = c_skip * x_t - c_out * v_pred
                #   c_out * v_pred = c_skip * x_t - x_0
                #   v_pred = (c_skip * x_t - x_0) / c_out
                c_out = self.scheduler.sigmas[i] ** 2 / (1 + self.scheduler.sigmas[i] ** 2) ** 0.5
                c_skip = 1 / (1 + self.scheduler.sigmas[i] ** 2) ** 0.5
                velocity = (c_skip * latents - target_audio_latent) / c_out

                # Scaling
                model_intput_latents = self.scheduler.scale_model_input(latents, t)
                # Augmentation for CFG scaling
                model_intput_latents = torch.cat((model_intput_latents, model_intput_latents), dim=0)

                # Positional Encoding
                rotary_embedding = get_1d_rotary_pos_embed(
                    self.diffusion.config.attention_head_dim // 2,
                    model_intput_latents.shape[2] + audio_duration_embeds.shape[1],
                    use_real=True,
                    repeat_interleave_real=False,
                )

                # Prediction
                velocity_pred = self.diffusion(
                    hidden_states=model_intput_latents,
                    timestep=torch.full((AB*2,), t, device=model_intput_latents.device),
                    encoder_hidden_states=text_audio_duration_embeds,
                    global_hidden_states=audio_duration_embeds,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                )[0]

                # CFG scaling
                velocity_pred, empty_velocity_pred = velocity_pred[:AB], velocity_pred[AB:]
                velocity_pred = empty_velocity_pred + self.cfg_scale * (velocity_pred - empty_velocity_pred)

                # Modify the prediction with the real velocity for inpainting
                velocity_pred[:, :, :latent_length] = velocity[:, :, :latent_length]

                # Sampling
                latents = self.scheduler.step(velocity_pred, t, latents).prev_sample
            audio_pred = self.vae.decode(latents).sample
            audio_pred = torchaudio.transforms.Resample(44100, 16000)(audio_pred.cpu().float()).cuda().half()
            audio_pred = audio_pred.mean(dim=1)

            # Audio Presentation
            total_audio = self.vae.decode(target_audio_latent).sample
            target_audio = torchaudio.transforms.Resample(44100, 16000)(total_audio.cpu().float()).cuda().half()
            target_audio = target_audio.mean(dim=1)
            audio_pred = torch.cat((target_audio[:, :AL], audio_pred[:, AL:]), dim=1)
            import matplotlib.pyplot as plt
            for i, (t_a, p_a) in enumerate(zip(target_audio, audio_pred)):
                torchaudio.save(f"audio_{i}.wav", p_a.cpu().float().unsqueeze(0), 16000)
                torchaudio.save(f"audio_{i}_target.wav", t_a.cpu().float().unsqueeze(0), 16000)
                plt.figure(figsize=(10, 5))
                plt.subplot(2, 1, 1)
                plt.plot(t_a.cpu().numpy())
                plt.subplot(2, 1, 2)
                plt.plot(p_a.cpu().numpy())
                plt.savefig(f"audio_{i}.png")
            exit()

            audio = x["audio"][:, 0, :]
            text  = x["text"]
            video = x["video"]
            audio = torch.cat((audio, audio_pred[:, AL:]), dim=1)
            audio = self.audio_model.model.feature_extractor(audio)                    # B, 512,  74
            audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))   # B,  74, 768
            audio = self.audio_model.model.encoder.pos_conv_embed(audio)               # B.  74, 768        
            audio = self.audio_model.model.encoder.layer_norm(audio)
            audio = self.audio_model.model.encoder.dropout(audio)
            text  = self.language_model.embeddings(text)                               # B, 74, 768
            video = self.video_model.model.embeddings(video, None)                     # B, 74, 768
        self.lora_mode('decoder')

        a_layers = self.audio_model.model.encoder.layers
        t_layers = self.language_model.encoder.layer
        v_layers = self.video_model.model.encoder.layer

        # y["InfoNCE"] = 0
        for l, (a_layer, t_layer, v_layer) in enumerate(zip(a_layers, t_layers, v_layers)):
            audio = a_layer(audio)[0]
            text  = t_layer(text)[0]
            video = v_layer(video)[0]
            # if l > 8:
            #     a_feature = self.audio_downproject(audio.mean(dim=1)) 
            #     t_feature = self.text_downproject(text[:, 0])
            #     v_feature = self.video_downproject(video.mean(dim=1))
            #     a_t = self.cross_attn0[0](a_feature, t_feature)
            #     a_v = self.cross_attn1[0](a_feature, v_feature)
            #     t_v = self.cross_attn2[0](t_feature, v_feature)
            #     y["InfoNCE"] += ( self.contrastive_loss(self.sim_matrix(a_feature, t_feature))
            #                       + self.contrastive_loss(self.sim_matrix(a_feature, v_feature))
            #                       + self.contrastive_loss(self.sim_matrix(t_feature, v_feature))
            #                       + self.contrastive_loss(self.sim_matrix(a_t, v_feature))
            #                       + self.contrastive_loss(self.sim_matrix(a_v, t_feature))
            #                       + self.contrastive_loss(self.sim_matrix(t_v, a_feature)))
        a_feature = self.audio_downproject(audio.mean(dim=1))
        t_feature = self.text_downproject(text[:, 0])
        v_feature = self.video_downproject(video.mean(dim=1))

        concat = torch.cat((a_feature, t_feature, v_feature), dim=1)
        y["logit"] = self.fc_layer_1(self.dropout(concat))
        y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(y["logit"]))
            
        return y
    
    def normalize_embeddings(self, a, eps=1e-8):
        a_n = a.norm(dim=-1, keepdim=True)
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        return a_norm

    def sim_matrix(self, a, b, eps=1e-8):
        a = self.normalize_embeddings(a, eps)
        b = self.normalize_embeddings(b, eps)

        sim_mt = torch.mm(a, b.transpose(0, 1))
        return sim_mt

# class Diffused_Backchannel(nn.Module):
#     BATCH_SIZE = 2 ** 14

#     def __init__(self,
#                  language_model=None,
#                  audio_model=None,
#                  video_model=None,
#                  sentiment_dict = None,
#                  output_size=128,
#                  num_class=4,
#                  sentiment_output_size=64,
#                  dropout=0.3,
#                  mode="cross_entropy"):
#         super(Diffused_Backchannel, self).__init__()

#         pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")
#         self.diffusion = pipe.transformer
#         self.vae = pipe.vae

#         self.multi_modal = False
#         self.class_wise = False
#         self.cross_attn = False
#         self.consistency = False

#         self.mode = mode
#         self.num_classes = num_class

#         self.register_module("language_model", language_model)
#         # if bert and vocab are not provided, raise an error
#         assert self.language_model is not None, "bert and vocab must be provided"

#         self.sentiment_dict = sentiment_dict
#         self.is_MT = self.sentiment_dict is not None
#         self.register_module("audio_model", audio_model)
#         self.register_module("video_model", video_model)

#         self.loras = nn.ModuleDict()
#         self.loras["encoder"] = nn.ModuleDict()
#         self.loras["decoder"] = nn.ModuleDict()
#         self.loras["diffusion"] = nn.ModuleDict()
#         self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

#         for name, module in self.audio_model.named_modules():
#             if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
#                 self.audio_lora[name.replace('.', '_')] = LoRA(module, 32, alpha=64
#                 self.audio_linear[name.replace('.', '_')] = module

#         for name, module in self.language_model.named_modules():
#             if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
#                 self.language_lora[name.replace('.', '_')] = LoRA(module, 32, alpha=64
#                 self.language_linear[name.replace('.', '_')] = module

#         self.lora_on()
        
#         self.register_buffer("betas", torch.arange(0, 1, 1/2000).to(torch.float32))
#         self.register_buffer("alphas", 1 - self.betas)
#         self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

#         print("Betas: ", self.betas)
#         print("Alphas: ", self.alphas)
#         print("Alpha Bars: ", self.alpha_bars)

#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Linear(768 + self.audio_model.get_feature_size(), num_class)
#         self.internal_counter = 1

#     def lora_on(self):
#         for name, module in self.audio_model.named_modules():
#             # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
#             if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
#                 _name = name.split('.')
#                 _module = self.audio_model
#                 for i in range(len(_name)-1):
#                     _module = _module.__getattr__(_name[i])
#                 _module.__setattr__(_name[-1], self.audio_lora[name.replace('.', '_')])

#         for name, module in self.language_model.named_modules():
#             if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
#                 _name = name.split('.')
#                 _module = self.language_model
#                 for i in range(len(_name)-1):
#                     _module = _module.__getattr__(_name[i])
#                 _module.__setattr__(_name[-1], self.language_lora[name.replace('.', '_')])

#     def lora_off(self):
#         for name, module in self.audio_model.named_modules():
#             # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
#             if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
#                 _name = name.split('.')
#                 _module = self.audio_model
#                 for i in range(len(_name)-1):
#                     _module = _module.__getattr__(_name[i])
#                 _module.__setattr__(_name[-1], self.audio_linear[name.replace('.', '_')])

#         for name, module in self.language_model.named_modules():
#             if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
#                 _name = name.split('.')
#                 _module = self.language_model
#                 for i in range(len(_name)-1):
#                     _module = _module.__getattr__(_name[i])
#                 _module.__setattr__(_name[-1], self.language_linear[name.replace('.', '_')])

#     def forward(self, x):
#         self.lora_on()
#         # Extract the features from the audio and text
#         device = self.parameters().__next__().device
#         audio = x["audio"]
#         text  = x["text"]
#         target_audio = x["target_audio"]
#         target_text = x["target_text"]
#         y = {}
#         # get audio only one channel
#         audio = audio[:, 0, :]
#         target_audio = target_audio[:, 0, :]
#         AB, AL = audio.shape
#         TB, TL = text.shape
#         random_steps = torch.randint(0, self.internal_counter, (AB,)).to(device)
        
#         audio = self.audio_model.model.feature_extractor(audio)
#         audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        
#         target_audio = self.audio_model.model.feature_extractor(target_audio)
#         target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))

#         text_embedding = self.language_model.embeddings(text)

#         target_text_embedding = self.language_model.embeddings(target_text)

#         if self.training:
#             noise_audio = torch.sqrt(1 - self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * torch.randn_like(target_audio_embedding) + \
#                         torch.sqrt(self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * target_audio_embedding

#             audio = torch.cat((audio_embedding, target_audio_embedding), dim=1)
#             noise_audio = torch.cat((audio_embedding, noise_audio), dim=1)

#             audio = self.audio_model.model.encoder(audio)[0]
#             noise_audio = self.audio_model.model.encoder(noise_audio)[0]

#             y['audio'] = F.mse_loss(audio, noise_audio)

#             noise_text = torch.sqrt(1 - self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * torch.randn_like(target_text_embedding) + \
#                         torch.sqrt(self.alpha_bars[random_steps].unsqueeze(1).unsqueeze(2)) * target_text_embedding

#             text = torch.cat((text_embedding, target_text_embedding), dim=1)
#             noise_text = torch.cat((text_embedding, noise_text), dim=1)

#             text = self.language_model.encoder(text)[0]
#             noise_text = self.language_model.encoder(noise_text)[0]

#             y['text'] = F.mse_loss(text, noise_text)

#             audio = (audio.mean(dim=1) + noise_audio.mean(dim=1)) / 2
#             text = (text[:, 0, :] + noise_text[:, 0, :]) / 2
        
#         else:
#             audio = torch.cat((audio_embedding, torch.rand_like(target_audio_embedding)), dim=1)
#             audio = self.audio_model.model.encoder(audio)[0]

#             text = torch.cat((text_embedding, torch.rand_like(target_text_embedding)), dim=1)
#             text = self.language_model.encoder(text)[0]

#             audio = audio.mean(dim=1)
#             text = text[:, 0, :]

#         if self.mode == "audio_only" or self.mode == "text_only":
#             concat = audio if self.mode == "audio_only" else text
#         else :
#             concat = torch.cat((audio, text), dim=1)
#         y["logit"] = self.classifier(self.dropout(concat))
#         if self.internal_counter != 2000:
#             self.internal_counter += 1
#         return y
    
#     def post_epoch(self, dataloader):
#         with torch.no_grad():
#             acc = 0
#             total = 0
#             for i, x in enumerate(dataloader):
#                 self.lora_on()
#                 # Extract the features from the audio and text
#                 device = self.parameters().__next__().device
#                 audio = x["audio"].to(device)
#                 text  = x["text"].to(device)
#                 target_audio = x["target_audio"].to(device)
#                 target_text = x["target_text"].to(device)
#                 y = {}
#                 # get audio only one channel
#                 audio = audio[:, 0, :]
#                 target_audio = target_audio[:, 0, :]
#                 AB, AL = audio.shape
#                 TB, TL = text.shape
#                 random_steps = torch.randint(1, self.internal_counter+1, (AB,)).to(device)
                
#                 audio = self.audio_model.model.feature_extractor(audio)
#                 audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                
#                 target_audio = self.audio_model.model.feature_extractor(target_audio)
#                 target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))

#                 text_embedding = self.language_model.embeddings(text)

#                 target_text_embedding = self.language_model.embeddings(target_text)

#                 audio = torch.cat((audio_embedding, target_audio_embedding), dim=1)
#                 audio = self.audio_model.model.encoder(audio)[0]

#                 text = torch.cat((text_embedding, target_text_embedding), dim=1)
#                 text = self.language_model.encoder(text)[0]

#                 audio = audio.mean(dim=1)
#                 text = text[:, 0, :]

#                 if self.mode == "audio_only" or self.mode == "text_only":
#                     concat = audio if self.mode == "audio_only" else text
#                 else :
#                     concat = torch.cat((audio, text), dim=1)
#                 y["logit"] = self.classifier(self.dropout(concat))
#                 if self.internal_counter != 2000:
#                     self.internal_counter += 1

#                 acc += (y["logit"].argmax(dim=1) == x["label"].to(device)).sum().item()
#                 total += x["label"].shape[0]
#                 print(f"Accuracy: {acc/total}", end='\r')
#             print()