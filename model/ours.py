import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from util.utils import get_audio_model, get_language_model
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE

class Ours(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours, self).__init__()

        self.multi_modal = False
        self.class_wise = False
        self.cross_attn = False
        self.consistency = False

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
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name):
                self.audio_lora[name.replace('.', '_')] = LoRA(module, 64, alpha=16)
                self.audio_linear[name.replace('.', '_')] = module

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name):
                self.language_lora[name.replace('.', '_')] = LoRA(module, 64, alpha=16)
                self.language_linear[name.replace('.', '_')] = module

        self.num_cluster = 32
        self.num_inner_cluster = 32
        self.mem_size = 60000

        self.dropout = nn.Dropout(dropout)
        if self.mode == "audio_only" or self.mode == "text_only":
            self.fc_layer_1 = nn.Linear(768, output_size)
        elif self.mode == "flatten":
            self.fc_layer_1 = nn.Linear(768 * 65, output_size)
        else:
            # self.fc_layer_1 = nn.Linear(794, output_size)
            self.fc_layer_1 = nn.Linear(768 + self.audio_model.get_feature_size(), output_size)
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
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.audio_lora[name.replace('.', '_')])

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name):
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.language_lora[name.replace('.', '_')])

    def lora_off(self):
        for name, module in self.audio_model.named_modules():
            # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.audio_linear[name.replace('.', '_')])

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name):
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.language_linear[name.replace('.', '_')])

    def k_means(self, x, k, max_iter=200, batch_size=64, init="kmeans++"):
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

    def pretext_forward(self, dataloader):
        self.audio_model.eval()
        self.language_model.eval()

        # self.audio_centroids = torch.randn(self.num_cluster, 768).to(self.parameters().__next__().device)
        # self.audio_target_centriods = torch.randn(self.num_cluster, self.num_inner_cluster, 768).to(self.parameters().__next__().device)
        # self.audio_text_centriods = torch.randn(self.num_cluster, self.num_inner_cluster, 768).to(self.parameters().__next__().device)
        # self.audio_cluster = torch.arange(self.num_cluster).repeat(1,self.num_inner_cluster).to(self.parameters().__next__().device)
        # self.text_centroids = torch.randn(self.num_cluster, 768).to(self.parameters().__next__().device)
        # self.text_target_centriods = torch.randn(self.num_cluster, self.num_inner_cluster, 768).to(self.parameters().__next__().device)
        # self.text_audio_centriods = torch.randn(self.num_cluster, self.num_inner_cluster, 768).to(self.parameters().__next__().device)
        # self.text_cluster = torch.arange(self.num_cluster).repeat(1,self.num_inner_cluster).to(self.parameters().__next__().device)
        # return

        if os.path.exists("features.pt"):
            self.lora_off()
            with torch.no_grad():
                features = torch.load("features.pt")
                self.audios = features["audios"]
                self.target_audios = features["target_audios"]
                self.texts = features["texts"]
                self.target_texts = features["target_texts"]
                self.labels = features["labels"]
        else:
            self.lora_off()
            with torch.no_grad():
                device = self.parameters().__next__().device
                self.audios = torch.empty(0, device=device)
                self.target_audios = torch.empty(0, device=device)
                self.texts = torch.empty(0, device=device)
                self.target_texts = torch.empty(0, device=device)
                self.labels = torch.empty(0, device=device)

                for n, x in enumerate(dataloader):
                    print(f"{n+1}/{len(dataloader)}", end='\r')

                    audio = x["audio"]
                    target_audio = x["target_audio"]

                    
                    audio = audio.to(device)
                    target_audio = target_audio.to(device)

                    audio = audio[:, 0, :]
                    audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                    audio = self.audio_model.model.feature_extractor(audio)
                    audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                    audio = self.audio_model.model.encoder(audio)[0]
                    audio = audio.mean(dim=1)
                    self.audios = torch.cat((self.audios, audio), dim=0)

                    target_audio = target_audio[:, 0, :]
                    target_audio = self.audio_model.processor(target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                    target_audio = self.audio_model.model.feature_extractor(target_audio)
                    target_audio = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))
                    # target_audio = self.audio_model.model.encoder.pos_conv_embed(target_audio)
                    # target_audio = self.audio_model.model.encoder.layer_norm(target_audio)
                    # target_audio = self.audio_model.model.encoder.dropout(target_audio)
                    self.target_audios = torch.cat((self.target_audios, target_audio), dim=0)
                    text = x["text"]
                    target_text = x["target_text"]

                    device = self.parameters().__next__().device
                    text = text.to(device)
                    target_text = target_text.to(device)

                    text = self.language_model.embeddings(text)
                    text = self.language_model.encoder(text)[0]
                    text = text[:, 0, :]
                    self.texts = torch.cat((self.texts, text), dim=0)
                    
                    target_text = self.language_model.embeddings(target_text)
                    target_text = self.language_model.encoder(target_text)[0]
                    self.target_texts = torch.cat((self.target_texts, target_text), dim=0)

                    self.labels = torch.cat((self.labels, x["label"].to(device)), dim=0)

                    if len(self.audios) > self.mem_size:
                        # break
                        index = torch.randperm(self.audios.shape[0])[:self.mem_size]
                        self.audios = self.audios[index]
                        self.target_audios = self.target_audios[index]
                        self.texts = self.texts[index]
                        self.target_texts = self.target_texts[index]
                        self.labels = self.labels[index]
                print()
                torch.save({"audios": self.audios, "target_audios": self.target_audios, "texts": self.texts, "target_texts": self.target_texts, "labels": self.labels}, "features.pt")
            # self.audios = torch.cat(self.audios, dim=0)
            # self.target_audios = torch.cat(self.target_audios, dim=0)
            # self.texts = torch.cat(self.texts, dim=0)
            # self.target_texts = torch.cat(self.target_texts, dim=0)
            # self.labels = torch.cat(self.labels, dim=0)

            # make a prototype for each cluster of audio
        self.audio_cluster = []
        self.audio_centroids = []
        self.audio_target_centriods = []
        self.audio_text_centriods = []
        if self.class_wise:
            for c in range(self.num_classes):
                centroids, cluster = self.k_means(self.audios[self.labels==c], self.num_cluster//self.num_classes)
                self.audio_centroids.append(centroids)
                for i, _ in enumerate(centroids):
                    _, _N, _D = self.target_audios[self.labels==c][cluster==i].shape
                    _centroids, _ = self.k_means(self.target_audios[self.labels==c][cluster==i].flatten(1), self.num_inner_cluster)
                    self.audio_target_centriods.append(_centroids.reshape(-1, _N, _D))
                    _, _N, _D = self.target_texts[self.labels==c][cluster==i].shape
                    _centroids, _ = self.k_means(self.target_texts[self.labels==c][cluster==i].flatten(1), self.num_inner_cluster)
                    self.audio_text_centriods.append(_centroids.reshape(-1, _N, _D))
                    self.audio_cluster.append(torch.tensor([i] * self.num_inner_cluster, device=cluster.device) + self.num_cluster // self.num_classes * c)
        else:
            centroids, cluster = self.k_means(self.audios, self.num_cluster)
            self.audio_centroids.append(centroids)
            for i, _ in enumerate(centroids):
                _, _N, _D = self.target_audios[cluster==i].shape
                _centroids, _ = self.k_means(self.target_audios[cluster==i].flatten(1), self.num_inner_cluster)
                self.audio_target_centriods.append(_centroids.reshape(-1, _N, _D))
                _, _N, _D = self.target_texts[cluster==i].shape
                _centroids, _ = self.k_means(self.target_texts[cluster==i].flatten(1), self.num_inner_cluster)
                self.audio_text_centriods.append(_centroids.reshape(-1, _N, _D))
                self.audio_cluster.append(torch.tensor([i] * self.num_inner_cluster, device=cluster.device))
        self.audio_centroids = torch.cat(self.audio_centroids, dim=0)
        self.audio_target_centriods = torch.cat(self.audio_target_centriods, dim=0)
        self.audio_text_centriods = torch.cat(self.audio_text_centriods, dim=0)
        self.audio_cluster = torch.cat(self.audio_cluster, dim=0)

        # make a prototype for each cluster of text
        self.text_cluster = []
        self.text_centroids = []
        self.text_target_centriods = []
        self.text_audio_centriods = []
        if self.class_wise:
            for c in range(self.num_classes):
                centroids, cluster = self.k_means(self.texts[self.labels==c], self.num_cluster//self.num_classes)
                self.text_centroids.append(centroids)
                for i, _ in enumerate(centroids):
                    _, _N, _D = self.target_texts[self.labels==c][cluster==i].shape
                    _centroids, _ = self.k_means(self.target_texts[self.labels==c][cluster==i].flatten(1), self.num_inner_cluster)
                    self.text_target_centriods.append(_centroids.reshape(-1, _N, _D))
                    _, _N, _D = self.target_audios[self.labels==c][cluster==i].shape
                    _centroids, _ = self.k_means(self.target_audios[self.labels==c][cluster==i].flatten(1), self.num_inner_cluster)
                    self.text_audio_centriods.append(_centroids.reshape(-1, _N, _D))
                    self.text_cluster.append(torch.tensor([i] * self.num_inner_cluster, device=cluster.device) + self.num_cluster // self.num_classes * c)
        else:
            centroids, cluster = self.k_means(self.texts, self.num_cluster)
            self.text_centroids.append(centroids)
            for i, _ in enumerate(centroids):
                _, _N, _D = self.target_texts[cluster==i].shape
                _centroids, _ = self.k_means(self.target_texts[cluster==i].flatten(1), self.num_inner_cluster)
                self.text_target_centriods.append(_centroids.reshape(-1, _N, _D))
                _, _N, _D = self.target_audios[cluster==i].shape
                _centroids, _ = self.k_means(self.target_audios[cluster==i].flatten(1), self.num_inner_cluster)
                self.text_audio_centriods.append(_centroids.reshape(-1, _N, _D))
                self.text_cluster.append(torch.tensor([i] * self.num_inner_cluster, device=cluster.device))
        self.text_centroids = torch.cat(self.text_centroids, dim=0)
        self.text_target_centriods = torch.cat(self.text_target_centriods, dim=0)
        self.text_audio_centriods = torch.cat(self.text_audio_centriods, dim=0)
        self.text_cluster = torch.cat(self.text_cluster, dim=0)

        # print("audio_centroids", self.audio_centroids.shape)
        # print("audio_target_centriods", self.audio_target_centriods.shape)
        # print("audio_text_centriods", self.audio_text_centriods.shape)
        # print("text_centroids", self.text_centroids.shape)
        # print("text_target_centriods", self.text_target_centriods.shape)
        # print("text_audio_centriods", self.text_audio_centriods.shape)
        # print("audio_cluster", self.audio_cluster.shape)
        # print("text_cluster", self.text_cluster.shape)

        self.audio_centroids.requires_grad = False
        self.audio_target_centriods.requires_grad = False
        self.text_centroids.requires_grad = False
        self.text_target_centriods.requires_grad = False
        self.audio_text_centriods.requires_grad = False
        self.text_audio_centriods.requires_grad = False


    def forward(self, x):
        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape
        
        device = self.parameters().__next__().device
        self.lora_off()
        # original_audio = audio.clone()
        # original_text = text.clone()
        with torch.no_grad():
            audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
            audio = self.audio_model.model.feature_extractor(audio)
            audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
            _audio = audio.clone()
            _audio = self.audio_model.model.encoder(_audio)[0]
            _audio = _audio.mean(dim=1)

        similarty = torch.cdist(_audio, self.audio_centroids.to(_audio.device),  p=2)
        # similarty = 1 - torch.cosine_similarity(__audio.unsqueeze(1), self.audio_keys.to(__audio.device).unsqueeze(0), dim=2)
        _, cluster = torch.min(similarty, dim=1)

        # print(cluster.shape)
        # print(self.audio_cluster.shape)
        # print((self.audio_centroids.unsqueeze(0) == self.audio_centroids.unsqueeze(1)).shape)
        # print(self.audio_target_centriods.repeat(AB, 1, 1, 1)[self.audio_cluster.unsqueeze(0) == cluster.unsqueeze(1)].shape)
        
        _in = self.audio_target_centriods.repeat(AB, 1, 1, 1)[self.audio_cluster.unsqueeze(0) == cluster.unsqueeze(1)].reshape(AB, self.num_inner_cluster, -1, 768)
        target_audio = _in[:, torch.randint(0, _in.shape[1], (1,))].squeeze(1)
        if self.multi_modal:
            _in = self.audio_text_centriods[cluster]
            target_audio_text = _in[:, torch.randint(0, _in.shape[1], (1,))]

        with torch.no_grad():
            text = self.language_model.embeddings(text)
            _text = text.clone()
            _text = self.language_model.encoder(_text)[0]
            _text = _text[:, 0, :]

        similarty = torch.cdist(_text, self.text_centroids.to(_text.device), p=2)
        _, cluster = torch.min(similarty, dim=1)
        
        _in = self.text_target_centriods.repeat(TB, 1, 1, 1)[self.audio_cluster.unsqueeze(0) == cluster.unsqueeze(1)].reshape(TB, self.num_inner_cluster, -1, 768)
        target_text = _in[:, torch.randint(0, _in.shape[1], (1,))].squeeze(1)
        if self.multi_modal:
            _in = self.text_audio_centriods[cluster]
            target_text_audio = _in[:, torch.randint(0, _in.shape[1], (1,))]
            
        if self.consistency:
            with torch.no_grad():
                __audio = audio.clone()
                __text = text.clone()

                _target_audio = x["target_audio"]
                _target_text = x["target_text"]

                _target_audio = _target_audio[:, 0, :]
                _target_audio = self.audio_model.processor(_target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                _target_audio = self.audio_model.model.feature_extractor(_target_audio)
                _target_audio = self.audio_model.model.feature_projection(_target_audio.transpose(1, 2))
                # _target_audio = self.audio_model.model.encoder.pos_conv_embed(_target_audio)
                # _target_audio = self.audio_model.model.encoder.layer_norm(_target_audio)
                # _target_audio = self.audio_model.model.encoder.dropout(_target_audio)

                _target_text = self.language_model.embeddings(_target_text)
                # _target_text = self.language_model.encoder(_target_text)[0]
                # _target_text = _target_text.unsqueeze(1)

                _audio = torch.cat((audio, _target_audio), dim=1)
                _text = torch.cat((text, _target_text), dim=1)

                _audio = self.audio_model.model.encoder.pos_conv_embed(_audio)
                _audio = self.audio_model.model.encoder.layer_norm(_audio)
                _audio = self.audio_model.model.encoder.dropout(_audio)
                for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
                    _audio = a_layer(_audio)[0]
                    _text = t_layer(_text)[0]

                _audio = _audio.reshape(AB, -1, 768).mean(dim=1)
                _text = _text.reshape(TB, -1, 768)[:, 0, :]

        self.lora_on()
        if self.multi_modal:
            audio = torch.cat((audio, target_audio, target_text_audio), dim=1)
            text = torch.cat((text, target_text, target_audio_text), dim=1)
        else:
            audio = torch.cat((audio, target_audio), dim=1)
            text = torch.cat((text, target_text), dim=1)

        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)
        for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            audio = a_layer(audio)[0]
            text = t_layer(text)[0]
            if self.cross_attn:
                if l > 8:
                    __audio = self.cross_attention_layer[l](audio, text)
                    __text = self.cross_attention_layer[l](text, audio)
                    audio = __audio
                    text = __text

        audio = audio.reshape(AB, -1, 768).mean(dim=1)
        text = text.reshape(TB, -1, 768)[:, 0, :]

        if self.consistency:
            consistency_loss = F.mse_loss(audio, _audio) + F.mse_loss(text, _text)
            y["consistency_loss"] = consistency_loss

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        y["logit"] = self.fc_layer_1(self.dropout(concat))
        y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(y["logit"]))
        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))

        return y