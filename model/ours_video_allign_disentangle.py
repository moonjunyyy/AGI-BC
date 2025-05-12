import time
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.contrastive_loss import NormSoftmaxLoss
from layer.cross_attention_layer import CrossAttentionLayer
from layer.cross_attention_layer import CrossAttentionLayer
from m00nny_utils.torch_util.lr_scheduler.warmup_cosine_anneling import WarmUpCosineAnnelingScheduler
from m00nny_utils.torch_util.lr_scheduler.warmup_constant import WarmUpConstantScheduler
from m00nny_utils.torch_util.parallel.sharded_modules import all_gather
from m00nny_utils.torch_util.parallel.parameter_hook import ParameterHook
from m00nny_utils.util.progress_bar import ProgressBar
from m00nny_utils.torch_util.layer.lora import apply_lora, LoRA

MODE = "ALL"
class Ours_Video_Align_Disentangle(nn.Module):
    def __init__(self, language_model=None, audio_model=None, video_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours_Video_Align_Disentangle, self).__init__()

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
        self.contrastive_loss = NormSoftmaxLoss(temperature=0.05)

        self.fc_layer_1 = nn.Linear(192*6, output_size)
        # self.contrastive_loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

        self.cross_attn0 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(12)])
        self.cross_attn1 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(12)])
        self.cross_attn2 = nn.ModuleList([CrossAttentionLayer(d_query=192, d_kv=192, nhead=12) for _ in range(12)])
        self.audio_downproject = nn.Linear(768, 192)
        self.text_downproject  = nn.Linear(768, 192)
        self.video_downproject = nn.Linear(384, 192)
        
    def pretext_task(self, _):
        from dataset.ETRI_Dataset import ETRI_All_Dialog_Video_Dataset
        from utils.utils import get_language_model
        from torch.utils.data import DataLoader

        tokenizer, _ = get_language_model("koBert")
        dataset = ETRI_All_Dialog_Video_Dataset(path = "/local_datasets", train=True, tokenizer=tokenizer, length=3)
        if dist.is_initialized(): sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        else: sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=8, sampler=sampler)
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
        optimizer = torch.optim.AdamW([
            {'params': bert_params,  'lr': 5e-6},
            {'params': other_params, 'lr': 5e-5},
        ])
        scheduler = WarmUpCosineAnnelingScheduler(optimizer, warmup_steps=len(dataloader), t_total=10*len(dataloader))
        start_time = time.time()
        for e in range(10):
            sampler.set_epoch(e)
            for i, data in enumerate(dataloader):
                for k in data.keys():
                    data[k] = data[k].cuda()
                audio = data["audio"]
                text  = data["text"]
                audio = audio[:, 0, :]
                video = data["video"]
                AB, AL = audio.shape
                TB, TL = text.shape

                with torch.amp.autocast('cuda', dtype=torch.float16):

                    audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
                    audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
                    audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
                    audio = self.audio_model.model.encoder.layer_norm(audio)
                    audio = self.audio_model.model.encoder.dropout(audio)   

                    global_audio = audio
                    local_audio  = audio.clone()
                    local_audio_mask = torch.arange(local_audio.size(1), device=local_audio.device).expand(local_audio.size(0), local_audio.size(1))
                    local_audio_mask = (local_audio_mask.unsqueeze(1) > local_audio_mask.unsqueeze(2)).long()
                    local_audio_mask = local_audio_mask.unsqueeze(1).expand(-1, 12, -1, -1)
                    
                    text = self.language_model.embeddings(text)
                    text = self.language_model.encoder(text)[0]
                    
                    global_text = text
                    local_text  = text.clone()
                    local_text_mask = torch.arange(local_text.size(1), device=local_text.device).expand(local_text.size(0), local_text.size(1))
                    local_text_mask = (local_text_mask.unsqueeze(1) > local_text_mask.unsqueeze(2)).long()
                    local_text_mask = local_text_mask.unsqueeze(1).expand(-1, 12, -1, -1)

                    video = self.video_model.model.embeddings(video, None)

                    global_video = video
                    local_video  = video.clone()
                    local_video_mask = torch.arange(local_video.size(1), device=local_video.device).expand(local_video.size(0), local_video.size(1))
                    local_video_mask = (local_video_mask.unsqueeze(1) > local_video_mask.unsqueeze(2)).long()
                    local_video_mask = local_video_mask.unsqueeze(1).expand(-1, 12, -1, -1)

                    a_layers = self.audio_model.model.encoder.layers
                    t_layers = self.language_model.encoder.layer
                    v_layers = self.video_model.model.encoder.layer

                    align_loss = 0
                    for l, (a_layer, t_layer, v_layer) in enumerate(zip(a_layers, t_layers, v_layers)):
                        # audio = a_layer(audio)[0]
                        # text  = t_layer(text)[0]
                        # video = v_layer(video)[0]

                        global_audio = a_layer(global_audio)[0]
                        global_text  = t_layer(global_text) [0]
                        global_video = v_layer(global_video)[0]

                        local_audio = a_layer(local_audio, attention_mask=local_audio_mask)[0]
                        local_text  = t_layer(local_text,  attention_mask=local_text_mask) [0]
                        local_video = v_layer(local_video, attention_mask=local_video_mask)[0]

                        if l > 8:
                            a_feature = self.audio_downproject(global_audio) 
                            t_feature = self.text_downproject(global_text)
                            v_feature = self.video_downproject(global_video)

                            la_feature = self.audio_downproject(local_audio)
                            lt_feature = self.text_downproject(local_text)
                            lv_feature = self.video_downproject(local_video)

                            # if MODE == "AT":
                            #     a_t = self.cross_attn0[-1](a_feature, t_feature).mean(dim=1)
                            # if MODE == "AV":
                            #     a_v = self.cross_attn1[-1](a_feature, v_feature).mean(dim=1)
                            # if MODE == "TV":
                            #     t_v = self.cross_attn2[-1](t_feature, v_feature).mean(dim=1)
                            # if MODE == "ALL" or MODE == "TA":
                            #     t_a = self.cross_attn0[-1](a_feature, t_feature).mean(dim=1)
                            # if MODE == "ALL" or MODE == "VA":
                            #     v_a = self.cross_attn1[-1](v_feature, a_feature).mean(dim=1)
                            # if MODE == "ALL" or MODE == "VT":
                            #     v_t = self.cross_attn2[-1](v_feature, t_feature).mean(dim=1)
                            
                            a_feature = a_feature.mean(dim=1)
                            t_feature = t_feature.mean(dim=1)
                            v_feature = v_feature.mean(dim=1)

                            la_feature = la_feature[:, -1, :].mean(dim=1)
                            lt_feature = lt_feature[:, -1, :].mean(dim=1)
                            lv_feature = lv_feature[:, -1, :].mean(dim=1)

                            align_loss += (self.contrastive_loss(self.sim_matrix(a_feature, t_feature))
                                         + self.contrastive_loss(self.sim_matrix(a_feature, v_feature))
                                         + self.contrastive_loss(self.sim_matrix(t_feature, v_feature)))
                            
                            align_loss += (self.contrastive_loss(self.sim_matrix(la_feature, lt_feature))
                                         + self.contrastive_loss(self.sim_matrix(la_feature, lv_feature))
                                         + self.contrastive_loss(self.sim_matrix(lt_feature, lv_feature)))

                            # if MODE == "AT":
                            #     align_loss += self.contrastive_loss(self.sim_matrix(a_t, v_feature))
                            # if MODE == "AV":
                            #     align_loss += self.contrastive_loss(self.sim_matrix(a_v, t_feature))
                            # if MODE == "TV":
                            #     align_loss += self.contrastive_loss(self.sim_matrix(t_v, a_feature))
                            # if MODE == "ALL" or MODE == "TA":
                            #     align_loss += self.contrastive_loss(self.sim_matrix(t_a, v_feature))
                            # if MODE == "ALL" or MODE == "VA":
                            #     align_loss += self.contrastive_loss(self.sim_matrix(v_a, t_feature))
                            # if MODE == "ALL" or MODE == "VT":
                            #     align_loss += self.contrastive_loss(self.sim_matrix(v_t, a_feature))

                optimizer.zero_grad()
                scaler.scale(align_loss).backward()
                if dist.is_initialized():
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None: p.grad = torch.zeros_like(p)
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                scaler.step(optimizer)
                scaler.update()
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / (i + 1 + e * len(dataloader))) * (10 * len(dataloader) - i - 1 - e * len(dataloader))
                print(f"Epoch {e}, Iter : {i} / {len(dataloader)} : {align_loss.item()}, LR : {[group['lr'] for group in optimizer.param_groups]}, ETA : {eta}", end='\r')
                scheduler.step()
            print()

    def forward(self, x, train="train"):
        y = {}
        audio = x["audio"]
        text  = x["text"]
        audio = audio[:, 0, :]
        video = x["video"]
        AB, AL = audio.shape
        TB, TL = text.shape
        
        audio = self.audio_model.model.feature_extractor(audio)                     # B, 512,  74
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))    # B,  74, 768
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)                # B.  74, 768        
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)   

        global_audio = audio
        local_audio  = audio.clone()
        local_audio_mask = torch.arange(local_audio.size(1), device=local_audio.device).expand(local_audio.size(0), local_audio.size(1))
        local_audio_mask = (local_audio_mask.unsqueeze(1) > local_audio_mask.unsqueeze(2)).long()
        local_audio_mask = local_audio_mask.unsqueeze(1).expand(-1, 12, -1, -1)
        
        text = self.language_model.embeddings(text)
        text = self.language_model.encoder(text)[0]
        
        global_text = text
        local_text  = text.clone()
        local_text_mask = torch.arange(local_text.size(1), device=local_text.device).expand(local_text.size(0), local_text.size(1))
        local_text_mask = (local_text_mask.unsqueeze(1) > local_text_mask.unsqueeze(2)).long()
        local_text_mask = local_text_mask.unsqueeze(1).expand(-1, 12, -1, -1)

        video = self.video_model.model.embeddings(video, None)

        global_video = video
        local_video  = video.clone()
        local_video_mask = torch.arange(local_video.size(1), device=local_video.device).expand(local_video.size(0), local_video.size(1))
        local_video_mask = (local_video_mask.unsqueeze(1) > local_video_mask.unsqueeze(2)).long()
        local_video_mask = local_video_mask.unsqueeze(1).expand(-1, 12, -1, -1)

        a_layers = self.audio_model.model.encoder.layers
        t_layers = self.language_model.encoder.layer
        v_layers = self.video_model.model.encoder.layer

        align_loss = 0
        for l, (a_layer, t_layer, v_layer) in enumerate(zip(a_layers, t_layers, v_layers)):
            # audio = a_layer(audio)[0]
            # text  = t_layer(text)[0]
            # video = v_layer(video)[0]

            global_audio = a_layer(global_audio)[0]
            global_text  = t_layer(global_text) [0]
            global_video = v_layer(global_video)[0]

            local_audio = a_layer(local_audio, attention_mask=local_audio_mask)[0]
            local_text  = t_layer(local_text,  attention_mask=local_text_mask) [0]
            local_video = v_layer(local_video, attention_mask=local_video_mask)[0]

            if l > 8:
                a_feature = self.audio_downproject(global_audio) 
                t_feature = self.text_downproject(global_text)
                v_feature = self.video_downproject(global_video)

                la_feature = self.audio_downproject(local_audio)
                lt_feature = self.text_downproject(local_text)
                lv_feature = self.video_downproject(local_video)

                # if MODE == "AT":
                #     a_t = self.cross_attn0[-1](a_feature, t_feature).mean(dim=1)
                # if MODE == "AV":
                #     a_v = self.cross_attn1[-1](a_feature, v_feature).mean(dim=1)
                # if MODE == "TV":
                #     t_v = self.cross_attn2[-1](t_feature, v_feature).mean(dim=1)
                # if MODE == "ALL" or MODE == "TA":
                #     t_a = self.cross_attn0[-1](a_feature, t_feature).mean(dim=1)
                # if MODE == "ALL" or MODE == "VA":
                #     v_a = self.cross_attn1[-1](v_feature, a_feature).mean(dim=1)
                # if MODE == "ALL" or MODE == "VT":
                #     v_t = self.cross_attn2[-1](v_feature, t_feature).mean(dim=1)
                
                a_feature = a_feature.mean(dim=1)
                t_feature = t_feature.mean(dim=1)
                v_feature = v_feature.mean(dim=1)

                la_feature = la_feature[:, -1, :].mean(dim=1)
                lt_feature = lt_feature[:, -1, :].mean(dim=1)
                lv_feature = lv_feature[:, -1, :].mean(dim=1)

                align_loss += (self.contrastive_loss(self.sim_matrix(a_feature, t_feature))
                                + self.contrastive_loss(self.sim_matrix(a_feature, v_feature))
                                + self.contrastive_loss(self.sim_matrix(t_feature, v_feature)))
                
                align_loss += (self.contrastive_loss(self.sim_matrix(la_feature, lt_feature))
                                + self.contrastive_loss(self.sim_matrix(la_feature, lv_feature))
                                + self.contrastive_loss(self.sim_matrix(lt_feature, lv_feature)))
        y["InfoNCE"] = align_loss
        concat = torch.cat([a_feature, t_feature, v_feature,
                            la_feature, lt_feature, lv_feature], dim=-1)
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
