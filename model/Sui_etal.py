import time
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.contrastive_loss import NormSoftmaxLoss
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from m00nny_utils.torch_util.lr_scheduler.warmup_cosine_anneling import WarmUpCosineAnnelingScheduler
from m00nny_utils.torch_util.lr_scheduler.warmup_constant import WarmUpConstantScheduler

MODE = "ALL"
class Sui_etal(nn.Module):
    def __init__(self, language_model=None, audio_model=None, video_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Sui_etal, self).__init__()

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

        self.audio_downsample = nn.Linear(768, 192)
        self.video_downsample = nn.Linear(384, 192)
        self.text_downsample  = nn.Linear(768, 192)

        self.at_cross_attention = CrossAttentionLayer(192, 192, 12)
        self.ta_cross_attention = CrossAttentionLayer(192, 192, 12)
        self.av_cross_attention = CrossAttentionLayer(192, 192, 12)
        self.va_cross_attention = CrossAttentionLayer(192, 192, 12)
        self.tv_cross_attention = CrossAttentionLayer(192, 192, 12)
        self.vt_cross_attention = CrossAttentionLayer(192, 192, 12)

        self.audio_self_attention = SelfAttentionLayer(192, 12)
        self.video_self_attention = SelfAttentionLayer(192, 12)
        self.text_self_attention  = SelfAttentionLayer(192, 12)

        self.fc_layer_1 = nn.Linear(192*9, output_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

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
            {'params': bert_params, 'lr': 5e-6},
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

                    audio = self.audio_model(audio)     # B,  74, 768
                    text = self.language_model(text)    # B,  20, 768
                    video = self.video_model(video)     # B,  16, 384

                    audio = self.audio_downsample(audio)     # B,  74, 192
                    text  = self.text_downsample(text)
                    video = self.video_downsample(video)

                    at = self.at_cross_attention(audio, text) # B, 20, 192
                    ta = self.ta_cross_attention(text, audio)
                    av = self.av_cross_attention(audio, video)
                    va = self.va_cross_attention(video, audio)
                    tv = self.tv_cross_attention(text, video)
                    vt = self.vt_cross_attention(video, text)

                    audio = self.audio_self_attention(audio)     # B,  20, 192
                    video = self.video_self_attention(video)
                    text  = self.text_self_attention(text)

                    align_loss = 0
                    align_loss += self.contrastive_loss(self.sim_matrix(audio, text))
                    align_loss += self.contrastive_loss(self.sim_matrix(audio, video))
                    align_loss += self.contrastive_loss(self.sim_matrix(text, video))

                    align_loss += self.contrastive_loss(self.sim_matrix(at, video))
                    align_loss += self.contrastive_loss(self.sim_matrix(ta, video))
                    align_loss += self.contrastive_loss(self.sim_matrix(av, text))
                    align_loss += self.contrastive_loss(self.sim_matrix(va, text))
                    align_loss += self.contrastive_loss(self.sim_matrix(tv, audio))
                    align_loss += self.contrastive_loss(self.sim_matrix(vt, audio))

                optimizer.zero_grad()
                # align_loss.backward()
                # self.align_optimizer.step()
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
        
        # video = self.video_model(video)
             
        audio = self.audio_model(audio)     # B,  74, 768
        text = self.language_model(text)    # B,  20, 768
        video = self.video_model(video)     # B,  16, 384

        audio = self.audio_downsample(audio)     # B,  74, 192
        text  = self.text_downsample(text)
        video = self.video_downsample(video)

        at = self.at_cross_attention(audio, text) # B, 20, 192
        ta = self.ta_cross_attention(text, audio)
        av = self.av_cross_attention(audio, video)
        va = self.va_cross_attention(video, audio)
        tv = self.tv_cross_attention(text, video)
        vt = self.vt_cross_attention(video, text)

        audio = self.audio_self_attention(audio)     # B,  20, 192
        video = self.video_self_attention(video)
        text  = self.text_self_attention(text)

        align_loss = 0
        align_loss += self.contrastive_loss(self.sim_matrix(audio, text))
        align_loss += self.contrastive_loss(self.sim_matrix(audio, video))
        align_loss += self.contrastive_loss(self.sim_matrix(text, video))

        align_loss += self.contrastive_loss(self.sim_matrix(at, video))
        align_loss += self.contrastive_loss(self.sim_matrix(ta, video))
        align_loss += self.contrastive_loss(self.sim_matrix(av, text))
        align_loss += self.contrastive_loss(self.sim_matrix(va, text))
        align_loss += self.contrastive_loss(self.sim_matrix(tv, audio))
        align_loss += self.contrastive_loss(self.sim_matrix(vt, audio))

        y["InfoNCE"] = align_loss
        # concat = torch.cat((a_feature, t_feature, v_feature), dim=1)

        audio = audio.mean(dim=1)
        text  = text.mean(dim=1)
        video = video.mean(dim=1)
        concat = torch.cat((audio, text, video), dim=1)
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
