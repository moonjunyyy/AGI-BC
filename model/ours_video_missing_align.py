import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.contrastive_loss import NormSoftmaxLoss
from layer.cross_attention_layer import CrossAttentionLayer
from m00nny_utils.torch_util.lr_scheduler.warmup_cosine_anneling import WarmUpCosineAnnelingScheduler
from m00nny_utils.torch_util.lr_scheduler.warmup_constant import WarmUpConstantScheduler
from m00nny_utils.torch_util.parallel.sharded_modules import all_gather
from m00nny_utils.torch_util.parallel.parameter_hook import ParameterHook
from m00nny_utils.util.progress_bar import ProgressBar
from m00nny_utils.torch_util.layer.lora import apply_lora, LoRA

# MODE = ["TA", "VA", "VT", "AT", "AV", "TV"]
MODE = ["TA", "VA", "VT"] # By default

class Ours_Video_Missing_Align(nn.Module):
    def __init__(self, language_model=None, audio_model=None, video_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours_Video_Missing_Align, self).__init__()

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
        self.audio_lora     = apply_lora(self.audio_model,    rank=32, alpha=64, module_names=['q_proj', 'k_proj', 'v_proj', 'out_proj'], lora_names=['encoder', 'decoder'])
        self.language_lora  = apply_lora(self.language_model, rank=32, alpha=64, module_names=['query', 'key', 'value', 'output.dense'],  lora_names=['encoder', 'decoder'])
        self.video_lora     = apply_lora(self.video_model,    rank=32, alpha=64, module_names=['query', 'key', 'value', 'output.dense'],  lora_names=['encoder', 'decoder'])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_layer_1 = nn.Linear(192*3, output_size)
        self.classifier = nn.Linear(output_size, num_class)
        self.contrastive_loss = NormSoftmaxLoss(temperature=0.05)
        # self.contrastive_loss = nn.MSELoss()

        self.cross_attn0 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(1)])
        self.cross_attn1 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(1)])
        self.cross_attn2 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(1)])
        self.audio_downproject = nn.Linear(768, 192)
        self.text_downproject  = nn.Linear(768, 192)
        self.video_downproject = nn.Linear(384, 192)

        self.pretext_epochs = 10

        self._audio_features = None
        self._text_features  = None
        self._video_features = None
        self._generated_text = json.load(open("backchannel_generated_text.json"))
        self._generated_text = pd.DataFrame(self._generated_text["samples"])
        # Assign a integer value to the backchannel category
        _assign = {"NoBC": 0, "Continuer": 1, "Understanding": 2, "EmpathicResponse": 3}
        self._generated_text["backchannel_category"] = self._generated_text["backchannel_category"].apply(lambda x: _assign[x])
    
    def lora_mode(self, mode:str):
        self.audio_lora.activate(mode)
        self.language_lora.activate(mode)
        self.video_lora.activate(mode)

    @ torch.no_grad()
    def _attention_mask_from_2d_to_4d(self, attention_mask:torch.Tensor, num_heads:int, dtype:torch.device=torch.float32):
        _B, _L = attention_mask.shape
        _fmin = torch.finfo(dtype).min
        attention_mask = torch.where(attention_mask==0, _fmin, 0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, num_heads, _L, -1).to(dtype) # B, H, L, L, mask the attention value of 0
        return attention_mask

    def pretext_task(self, trainloader):
        self.lora_mode("encoder")
        length = trainloader.dataset.length
        from dataset.ETRI_Dataset import ETRI_All_Dialog_Video_Dataset
        from utils.utils import get_language_model
        from torch.utils.data import DataLoader

        tokenizer, _ = get_language_model("koBert")
        dataset = ETRI_All_Dialog_Video_Dataset(path = "/local_datasets", train=True, tokenizer=tokenizer, length=length)
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=8, sampler=sampler)
        scaler = torch.amp.GradScaler('cuda')
        bert_params = []
        other_params = []
        classifier_params = []
        for name, param in self.named_parameters():
            # print(name)
            if 'language_model' in name or 'audio_model' in name or 'video_model' in name:
                bert_params.append(param)
            elif 'fc_layer' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': bert_params,  'lr': 5e-6},
            {'params': other_params, 'lr': 5e-5},
        ])
        scheduler = WarmUpCosineAnnelingScheduler(optimizer, warmup_steps=len(dataloader), t_total=10*len(dataloader))
        scaler = torch.amp.GradScaler('cuda')
        for e in range(self.pretext_epochs):
            if dist.is_initialized(): sampler.set_epoch(e)
            total_loss = 0
            pbar = ProgressBar(dataloader)
            for i, data in enumerate(pbar):
                for k in data.keys():
                    data[k] = data[k].cuda()
                audio = data["audio"]
                text  = data["text"]
                video = data["video"]
                text_attention_mask = data["text_attention_mask"]
                audio = audio[:, 0, :]
                AB, AL = audio.shape
                TB, TL = text.shape
                text_attention_mask = self._attention_mask_from_2d_to_4d(text_attention_mask, 12, dtype=torch.float32)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
                    audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
                    audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
                    audio = self.audio_model.model.encoder.layer_norm(audio)
                    audio = self.audio_model.model.encoder.dropout(audio)    
                    
                    text  = self.language_model.embeddings(text)
                    text  = self.language_model.encoder(text, attention_mask=text_attention_mask)[0]

                    video = self.video_model.model.embeddings(video, None)

                    a_layers = self.audio_model.model.encoder.layers
                    t_layers = self.language_model.encoder.layer
                    v_layers = self.video_model.model.encoder.layer

                    align_loss = 0
                    for l, (a_layer, t_layer, v_layer) in enumerate(zip(a_layers, t_layers, v_layers)):
                        audio = a_layer(audio)[0]
                        text  = t_layer(text)[0]
                        video = v_layer(video)[0]
                        if l > 8:
                            a_feature = self.audio_downproject(audio) 
                            t_feature = self.text_downproject(text)
                            v_feature = self.video_downproject(video)

                            if "AT" in MODE:
                                a_t = self.cross_attn0[-1](a_feature, t_feature).mean(dim=1)
                                all_a_t = torch.cat(all_gather(a_t), dim=0)
                            if "AV" in MODE:
                                a_v = self.cross_attn1[-1](a_feature, v_feature).mean(dim=1)
                                all_a_v = torch.cat(all_gather(a_v), dim=0)
                            if "TV" in MODE:
                                t_v = self.cross_attn2[-1](t_feature, v_feature).mean(dim=1)
                                all_t_v = torch.cat(all_gather(t_v), dim=0)
                            if "TA" in MODE:
                                t_a = self.cross_attn0[-1](t_feature, a_feature).mean(dim=1)
                                all_t_a = torch.cat(all_gather(t_a), dim=0)
                            if "VA" in MODE:
                                v_a = self.cross_attn1[-1](v_feature, a_feature).mean(dim=1)
                                all_v_a = torch.cat(all_gather(v_a), dim=0)
                            if "VT" in MODE:
                                v_t = self.cross_attn2[-1](v_feature, t_feature).mean(dim=1)
                                all_v_t = torch.cat(all_gather(v_t), dim=0)
                            
                            # Mean pool the feature
                            a_feature = a_feature.mean(dim=1)
                            t_feature = t_feature.mean(dim=1)
                            v_feature = v_feature.mean(dim=1)

                            # Concatenate the features from other ranks
                            all_a_feature = torch.cat(all_gather(a_feature), dim=0)
                            all_t_feature = torch.cat(all_gather(t_feature), dim=0)
                            all_v_feature = torch.cat(all_gather(v_feature), dim=0)

                            # Calculate the contrastive loss
                            align_loss = align_loss + (self.contrastive_loss(self.sim_matrix(all_a_feature, all_t_feature))
                                                    +  self.contrastive_loss(self.sim_matrix(all_a_feature, all_v_feature))
                                                    +  self.contrastive_loss(self.sim_matrix(all_t_feature, all_v_feature)))
                            if "AT" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_a_t, all_v_feature))
                            if "AV" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_a_v, all_t_feature))
                            if "TV" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_t_v, all_a_feature))
                            if "TA" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_t_a, all_v_feature))
                            if "VA" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_v_a, all_t_feature))
                            if "VT" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_v_t, all_a_feature))
                total_loss += align_loss.item()
                optimizer.zero_grad()
                scaler.scale(align_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.set_descriptions({"Loss": f"{total_loss/(i+1):.4f} ({align_loss.item():.4f})",
                                       "LR": [f"{group['lr']:.2E}" for group in optimizer.param_groups]})
                scheduler.step()

    def _generate_features(self):
        from dataset.ETRI_Dataset import ETRI_All_Dialog_Video_Dataset
        from utils.utils import get_language_model
        from torch.utils.data import DataLoader

        tokenizer, _ = get_language_model("koBert")
        dataset = ETRI_All_Dialog_Video_Dataset(path = "/local_datasets", train=True, tokenizer=tokenizer, length=3)
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
            indices = [i for i in range(len(dataset))][dist.get_rank()::dist.get_world_size()]
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
            indices = [i for i in range(len(dataset))]
        dataloader = DataLoader(dataset, batch_size=32, num_workers=8, sampler=sampler)
        
        _indices        = torch.as_tensor(indices, device="cuda")
        _audio_features = torch.empty(0, device="cuda")
        _video_features = torch.empty(0, device="cuda")
        pbar = ProgressBar(dataloader)
        with torch.no_grad():
            for i, data in enumerate(pbar):
                for k in data.keys():
                    data[k] = data[k].cuda()
                audio = data["audio"]
                video = data["video"]
                audio = audio[:, 0, :]
                AB, AL = audio.shape

                audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
                audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
                audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
                audio = self.audio_model.model.encoder.layer_norm(audio)
                audio = self.audio_model.model.encoder.dropout(audio)    
                
                video = self.video_model.model.embeddings(video, None)

                a_layers = self.audio_model.model.encoder.layers
                v_layers = self.video_model.model.encoder.layer

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    for l, (a_layer, v_layer) in enumerate(zip(a_layers, v_layers)):
                        audio = a_layer(audio)[0]
                        video = v_layer(video)[0]
                        # if l == 8:
                        #     self._audio_intermediate_features = torch.cat((self._audio_intermediate_features.cpu(), audio), dim=0)
                        #     self._video_intermediate_features = torch.cat((self._video_intermediate_features.cpu(), video), dim=0)
                    a_feature = self.audio_downproject(audio).mean(dim=1)
                    v_feature = self.video_downproject(video).mean(dim=1)
                    a_feature = torch.cat(all_gather(a_feature), dim=0)
                    v_feature = torch.cat(all_gather(v_feature), dim=0)
                    _audio_features = torch.cat((_audio_features, a_feature), dim=0)
                    _video_features = torch.cat((_video_features, v_feature), dim=0)
            torch.save(_audio_features, "audio_features.pt")
            torch.save(_video_features, "video_features.pt")
        return _audio_features, _video_features

    def _load_features(self):
        from dataset.ETRI_Dataset import ETRI_All_Dialog_Video_Dataset, ETRI_2022_Dialog_Video_Dataset, ETRI_2023_Dialog_Video_Dataset
        from utils.utils import get_language_model
        from torch.utils.data import DataLoader, Subset
        import os
        if os.path.exists("audio_features.pt") and os.path.exists("video_features.pt"):
            self._audio_features = torch.load("audio_features.pt")
            self._video_features = torch.load("video_features.pt")
        else:
            self._audio_features, self._video_features = self._generate_features()

        tokenizer, _ = get_language_model("koBert")
        text = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        text = text.cuda()
        text = self.language_model.embeddings(text)
        text = self.language_model.encoder(text)[0]
        t_feature = self.text_downproject(text).mean(dim=1)
        if self._text_features is None: self._text_features = t_feature
        else: self._text_features = torch.cat((self._text_features, t_feature), dim=0)

        t_a_sim = self.sim_matrix(self._text_features, self._audio_features).cpu()
        t_v_sim = self.sim_matrix(self._text_features, self._video_features).cpu()

        text_audio_retrival = t_a_sim.argmax(dim=1)
        text_video_retrival = t_v_sim.argmax(dim=1)

        _dataset = ETRI_All_Dialog_Video_Dataset(path = "/local_datasets", train=True, tokenizer=tokenizer, length=3)
        _dataset_2022 = _dataset.dataset_2022
        _dataset_2023 = _dataset.dataset_2023
        _df_2022 = _dataset_2022.dataframe
        _df_2023 = _dataset_2023.dataframe
        _len2022 = len(_dataset_2022)
        _indices_2022 = text_audio_retrival[text_audio_retrival <  _len2022]
        _indices_2023 = text_audio_retrival[text_audio_retrival >= _len2022]
        _df_2022 = _df_2022.iloc[_indices_2022]
        _df_2023 = _df_2023.iloc[_indices_2023]

        _dataset.dataset_2022.dataframe = _df_2022
        _dataset.dataset_2023.dataframe = _df_2023

        self.dataloader_2022 = DataLoader(_dataset_2022, batch_size=4, num_workers=8, shuffle=True)
        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt

        # # _audio_sim_matrix = self.sim_matrix(_audio_features, _audio_features).cpu()
        # # _video_sim_matrix = self.sim_matrix(_video_features, _video_features).cpu()

        # _audio_tsne = TSNE(n_components=2).fit_transform(_audio_features.cpu())
        # _video_tsne = TSNE(n_components=2).fit_transform(_video_features.cpu())

        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(1, 2, 1)
        # ax.scatter(_audio_tsne[:, 0], _audio_tsne[:, 1])
        # ax.set_title("Audio")
        # ax = fig.add_subplot(1, 2, 2)
        # ax.scatter(_video_tsne[:, 0], _video_tsne[:, 1])
        # ax.set_title("Video")
        # fig.savefig("audio_video_tsne.png")
        # fig.clf()

        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(1, 2, 1)
        # ax.scatter(_audio_tsne[:, 0], _audio_tsne[:, 1], _audio_tsne[:, 2])
        # ax.set_title("Audio")
        # ax = fig.add_subplot(1, 2, 2)
        # ax.scatter(_video_tsne[:, 0], _video_tsne[:, 1], _video_tsne[:, 2])
        # ax.set_title("Video")
        # fig.savefig("audio_video_tsne_3d.png")
        # fig.clf()

    def forward(self, x, train="train"):
        self.lora_mode("encoder")
        if self._audio_features is None: self._load_features()

        y = {}
        audio = x["audio"]
        audio = audio[:, 0, :]
        text  = x["text"]
        video = x["video"]
        AB, AL = audio.shape
        TB, TL = text.shape
        # Make the mask into 4D shape
        text_attention_mask = x["text_attention_mask"]
        text_attention_mask = self._attention_mask_from_2d_to_4d(text_attention_mask, 12, dtype=audio.dtype)

        # video = self.video_model(video)
             
        audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)    
        
        text = self.language_model.embeddings(text)

        video = self.video_model.model.embeddings(video, None)

        a_layers = self.audio_model.model.encoder.layers
        t_layers = self.language_model.encoder.layer
        v_layers = self.video_model.model.encoder.layer

        align_loss = 0
        for l, (a_layer, t_layer, v_layer) in enumerate(zip(a_layers, t_layers, v_layers)):
            audio = a_layer(audio)[0]
            text  = t_layer(text, attention_mask=text_attention_mask)[0]
            video = v_layer(video)[0]
            if l > 8:
                a_feature = self.audio_downproject(audio) 
                t_feature = self.text_downproject(text)
                v_feature = self.video_downproject(video)
                
                if "AT" in MODE:
                    a_t = self.cross_attn0[-1](a_feature, t_feature).mean(dim=1)
                    all_a_t = torch.cat(all_gather(a_t), dim=0)
                if "AV" in MODE:
                    a_v = self.cross_attn1[-1](a_feature, v_feature).mean(dim=1)
                    all_a_v = torch.cat(all_gather(a_v), dim=0)
                if "TV" in MODE:
                    t_v = self.cross_attn2[-1](t_feature, v_feature).mean(dim=1)
                    all_t_v = torch.cat(all_gather(t_v), dim=0)
                if "TA" in MODE:
                    t_a = self.cross_attn0[-1](t_feature, a_feature).mean(dim=1)
                    all_t_a = torch.cat(all_gather(t_a), dim=0)
                if "VA" in MODE:
                    v_a = self.cross_attn1[-1](v_feature, a_feature).mean(dim=1)
                    all_v_a = torch.cat(all_gather(v_a), dim=0)
                if "VT" in MODE:
                    v_t = self.cross_attn2[-1](v_feature, t_feature).mean(dim=1)
                    all_v_t = torch.cat(all_gather(v_t), dim=0)
                
                # Mean pool the feature
                a_feature = a_feature.mean(dim=1)
                t_feature = t_feature.mean(dim=1)
                v_feature = v_feature.mean(dim=1)

                # Concatenate the features from other ranks
                all_a_feature = torch.cat(all_gather(a_feature), dim=0)
                all_t_feature = torch.cat(all_gather(t_feature), dim=0)
                all_v_feature = torch.cat(all_gather(v_feature), dim=0)

                # Calculate the contrastive loss
                align_loss = align_loss + (self.contrastive_loss(self.sim_matrix(all_a_feature, all_t_feature))
                                        +  self.contrastive_loss(self.sim_matrix(all_a_feature, all_v_feature))
                                        +  self.contrastive_loss(self.sim_matrix(all_t_feature, all_v_feature)))
                
                if "AT" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_a_t, all_v_feature))
                if "AV" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_a_v, all_t_feature))
                if "TV" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_t_v, all_a_feature))
                if "TA" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_t_a, all_v_feature))
                if "VA" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_v_a, all_t_feature))
                if "VT" in MODE: align_loss = align_loss + self.contrastive_loss(self.sim_matrix(all_v_t, all_a_feature))
        y["InfoNCE"] = align_loss
        # a_feature = self.audio_downproject(audio.mean(dim=1))
        # t_feature = self.text_downproject(text[:, 0])
        # v_feature = self.video_downproject(video.mean(dim=1))

        # concat = torch.cat((a_feature, t_feature, v_feature), dim=1)
        concat = torch.cat((a_feature, t_a, v_a), dim=1)
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
