import os
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from utils.utils import get_audio_model, get_language_model, all_gather
from utils.kmeans import KMeans
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE
import logging
import matplotlib.pyplot as plt

class Proxy_Prototype(nn.Module):
    BATCH_SIZE = 2 ** 14

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
        super(Proxy_Prototype, self).__init__()

        self.multi_modal = False
        self.class_wise = False
        self.cross_attn = False
        self.consistency = False

        self.mode = mode
        self.num_classes = num_class

        if language_model is not None:
            self.register_module("language_model", language_model)
            # if bert and vocab are not provided, raise an error
            assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        if audio_model is not None:
            self.register_module("audio_model", audio_model)
            # define the LSTM layer, 4 of layers
            self.audio_feature_size = audio_model.get_feature_size()

        if video_model is not None:
            self.register_module("video_model", video_model)
            self.video_feature_size = video_model.get_feature_size()

        self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

        self.language_linear = nn.ModuleDict()
        self.language_lora = nn.ModuleDict()
        self.audio_linear = nn.ModuleDict()
        self.audio_lora = nn.ModuleDict()

        for name, module in self.audio_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
                self.audio_lora[name.replace('.', '_')] = LoRA(module, 4, alpha=8)
                self.audio_linear[name.replace('.', '_')] = module

        for name, module in self.language_model.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
                self.language_lora[name.replace('.', '_')] = LoRA(module, 4, alpha=8)
                self.language_linear[name.replace('.', '_')] = module

        self.internal_counter = 0
        self.num_cluster = 2 ** 8
        self.mem_size = 2 ** 12

        class ResMLP(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, (out_dim + in_dim) * 2)
                self.fc2 = nn.Linear((out_dim + in_dim)  * 2, out_dim)
                self.relu = nn.GELU()
                self.dropout = nn.Dropout(0.1)
            def forward(self, x):
                return x + self.relu(self.fc2(self.dropout(self.relu(self.fc1(self.dropout(x))))))

        self.audio_key_matcher = nn.Sequential(
            nn.Linear(768, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            nn.Linear(768 * 2, self.num_cluster),
        ).to(self.parameters().__next__().device)
        self.text_key_matcher = nn.Sequential(
            nn.Linear(768, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            ResMLP(768 * 2, 768 * 2),
            nn.Linear(768 * 2, self.num_cluster),
        ).to(self.parameters().__next__().device)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear((768 + self.audio_model.get_feature_size())*2, num_class)
        self.internal_counter = 0

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

    def pretext_task(self, dataloader):
        self.lora_on()
        device = self.parameters().__next__().device
        tmp_head = nn.Linear(768 * 2, self.num_classes).to(device)
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        optim.add_param_group({'params': tmp_head.parameters()})
        for epoch in range(0):
            for i, data in enumerate(dataloader):
                batch_size = data["audio"].shape[0]
                audio = data["audio"].to(device)
                text = data["text"].to(device)
                target_audio = data["target_audio"].to(device)
                target_text = data["target_text"].to(device)
                audio = audio[:, 0, :]
                target_audio = target_audio[:, 0, :]
                AB, AL = audio.shape
                TB, TL = text.shape

                audio = self.audio_model.model.feature_extractor(audio)
                audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                audio = self.audio_model.model.encoder(audio_embedding)[0]
                audio = audio.mean(dim=1)

                text_embedding = self.language_model.embeddings(text)
                text = self.language_model.encoder(text_embedding)[0]
                text = text[:, 0, :]

                concat = torch.cat((audio, text), dim=1)
                y = tmp_head(concat)
                loss = F.cross_entropy(y, data["label"].to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(f"pretext_task: {i}/{len(dataloader)}", end='\r')
            print()
    
    def pre_epoch(self, dataloader):
        self.lora_on()
        device = self.parameters().__next__().device
        # optimizer = torch.optim.Adam([p for n, p in self.named_parameters() if 'lora' not in n], lr=1e-3)

        self.audio_features = torch.empty(0).to(device)
        self.text_features = torch.empty(0).to(device)
        self.target_audio_features = torch.empty(0).to(device)
        self.target_text_features = torch.empty(0).to(device)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                batch_size = data["audio"].shape[0]
                audio = data["audio"].to(device)
                text = data["text"].to(device)
                target_audio = data["target_audio"].to(device)
                target_text = data["target_text"].to(device)
                audio = audio[:, 0, :]
                target_audio = target_audio[:, 0, :]
                AB, AL = audio.shape
                TB, TL = text.shape

                # audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                audio = self.audio_model.model.feature_extractor(audio)
                audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                audio = self.audio_model.model.encoder(audio_embedding.detach())[0]
                audio = audio.mean(dim=1)
                self.audio_features = torch.cat((self.audio_features, audio.detach()), dim=0)

                # target_audio = self.audio_model.processor(target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                target_audio = self.audio_model.model.feature_extractor(target_audio)
                target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))
                target_audio = self.audio_model.model.encoder(target_audio_embedding)[0]
                target_audio = target_audio.mean(dim=1)
                self.target_audio_features = torch.cat((self.target_audio_features, target_audio.detach()), dim=0)

                text_embedding = self.language_model.embeddings(text)
                text = self.language_model.encoder(text_embedding)[0]
                text = text[:, 0, :]
                self.text_features = torch.cat((self.text_features, text.detach()), dim=0)

                target_text_embedding = self.language_model.embeddings(target_text)
                target_text = self.language_model.encoder(target_text_embedding)[0]
                target_text = target_text[:, 0, :]
                self.target_text_features = torch.cat((self.target_text_features, target_text.detach()), dim=0)

                print(f"pre_epoch: {i}/{len(dataloader)}", end='\r')
            print()

            device = self.parameters().__next__().device
            if torch.distributed.is_initialized():
                gathered = all_gather(self.audio_features, device)
                self.audio_features = torch.cat(gathered, dim=0)
                gathered = all_gather(self.text_features, device)
                self.text_features = torch.cat(gathered, dim=0)
                gathered = all_gather(self.target_audio_features, device)
                self.target_audio_features = torch.cat(gathered, dim=0)
                gathered = all_gather(self.target_text_features, device)
                self.target_text_features = torch.cat(gathered, dim=0)

            print("clustering")
            self.target_audio_clustering = KMeans(n_clusters=self.num_cluster, max_iter=1000, batchsize=self.BATCH_SIZE, mode='cosine', init='kmeans++', seed=None)
            self.audio_clusters = self.target_audio_clustering.fit_predict(self.target_audio_features).clone()
            self.target_audio_centroids = self.target_audio_clustering.get_centroids().clone().reshape(self.num_cluster, 768)

            self.target_text_clustering = KMeans(n_clusters=self.num_cluster, max_iter=1000, batchsize=self.BATCH_SIZE, mode='cosine', init='kmeans++', seed=None)
            self.text_clusters = self.target_text_clustering.fit_predict(self.target_text_features).clone()
            self.target_text_centriods = self.target_text_clustering.get_centroids().clone().reshape(self.num_cluster, 768)
            print("clustering done")

        for param in self.audio_key_matcher.parameters():
            param.requires_grad = True
        for param in self.text_key_matcher.parameters():
            param.requires_grad = True
        self.audio_key_matcher.train()
        self.text_key_matcher.train()

        self.audio_key_matcher_optimizer = torch.optim.Adam(self.audio_key_matcher.parameters(), lr=1e-3)
        self.text_key_matcher_optimizer = torch.optim.Adam(self.text_key_matcher.parameters(), lr=1e-3)

        audio_acc = 0
        indices = torch.tensor(list(range(torch.distributed.get_rank(), len(self.text_features), torch.distributed.get_world_size())))
        while audio_acc < 0.95:
            acc = 0; count = 0
            indices = torch.randperm(len(indices))
            x = self.audio_features[indices]
            y = self.audio_clusters[indices]
            for i in range(0, len(indices), self.BATCH_SIZE):
                p = self.audio_key_matcher(x[i:i+self.BATCH_SIZE])
                l = F.cross_entropy(p, y[i:i+self.BATCH_SIZE], reduction='mean')
                self.audio_key_matcher_optimizer.zero_grad()
                l.backward()
                with torch.no_grad():
                    acc += (p.argmax(dim=1) == y[i:i+self.BATCH_SIZE]).float().sum()
                    count += torch.tensor(len(indices[i:i+self.BATCH_SIZE])).to(device)
                    audio_acc = acc / count
                print(f"audio_loss: {l.item()} audio_acc: {audio_acc.item()}", end='\r')
                if torch.distributed.is_initialized():
                    for param in self.audio_key_matcher.parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        torch.distributed.all_reduce(param.grad)
                self.audio_key_matcher_optimizer.step()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(acc)
                torch.distributed.all_reduce(count)
            audio_acc = acc / count
            print(f"audio_loss: {l.item()} audio_acc: {audio_acc.item()}", end='\r')
        print()
        text_acc = 0
        indices = torch.tensor(list(range(torch.distributed.get_rank(), len(self.text_features), torch.distributed.get_world_size())))
        x = self.text_features[indices]
        y = self.text_clusters[indices]
        while text_acc < 0.95:
            acc = 0; count = 0
            indices = torch.randperm(len(indices))
            x = self.text_features[indices]
            for i in range(0, len(indices), self.BATCH_SIZE):
                p = self.text_key_matcher(x[i:i+self.BATCH_SIZE])
                l = F.cross_entropy(p, y[i:i+self.BATCH_SIZE], reduction='mean')
                self.text_key_matcher_optimizer.zero_grad()
                l.backward()
                with torch.no_grad():
                    acc += (p.argmax(dim=1) == self.text_clusters[indices[i:i+self.BATCH_SIZE]]).float().sum()
                    count += torch.tensor(len(indices[i:i+self.BATCH_SIZE])).to(device)
                    audio_acc = acc / count
                print(f"audio_loss: {l.item()} audio_acc: {audio_acc.item()}", end='\r')
                if torch.distributed.is_initialized():
                    for param in self.text_key_matcher.parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        torch.distributed.all_reduce(param.grad)
                self.text_key_matcher_optimizer.step()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(acc)
                torch.distributed.all_reduce(count)
            text_acc = acc / count
            print(f"text_loss: {l.item()} text_acc: {text_acc.item()}", end='\r')
        print()
        for param in self.audio_key_matcher.parameters():
            param.requires_grad = False
        for param in self.text_key_matcher.parameters():
            param.requires_grad = False
        self.audio_key_matcher.eval()
        self.text_key_matcher.eval()

    def forward(self, x):
        self.lora_on()
        # Extract the features from the audio and text
        device = self.parameters().__next__().device
        audio = x["audio"].to(device)
        text  = x["text"].to(device)
        target_audio = x["target_audio"].to(device)
        target_text = x["target_text"].to(device)
        y = {}
        # get audio only one channel
        audio = audio[:, 0, :]
        target_audio = target_audio[:, 0, :]
        AB, AL = audio.shape
        TB, TL = text.shape
        
        audio = self.audio_model.model.feature_extractor(audio)
        audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.audio_model.model.encoder(audio_embedding)[0]
        audio = audio.mean(dim=1)
        text_embedding = self.language_model.embeddings(text)
        text = self.language_model.encoder(text_embedding)[0]
        text = text[:, 0, :]

        with torch.no_grad():
            audio_weight = self.audio_key_matcher(audio)
            text_weight = self.text_key_matcher(text)
        target_audio = self.target_audio_centroids[audio_weight.argmax(dim=1)]
        target_text = self.target_text_centriods[text_weight.argmax(dim=1)]
        
        # audio_positive_map = (audio_label.unsqueeze(1) == audio_label.unsqueeze(0)).to(torch.float32)
        # text_positive_map = (text_label.unsqueeze(1) == text_label.unsqueeze(0)).to(torch.float32)
        # audio_distance_map = 1 - F.cosine_similarity(target_audio_pred.unsqueeze(1), target_audio_pred.unsqueeze(0), dim=2)
        # text_distance_map = 1 - F.cosine_similarity(target_text_pred.unsqueeze(1), target_text_pred.unsqueeze(0), dim=2)
        # audio_loss = (audio_positive_map * audio_distance_map).sum() / (audio_distance_map.sum() + 1e-8)
        # text_loss = (text_positive_map * text_distance_map).sum() / (text_distance_map.sum() + 1e-8)

        # y["contrastive_loss"] = audio_loss + text_loss
        # audio = audio + target_audio
        # text = text + target_text

        audio = torch.cat((audio, target_audio), dim=1)
        text = torch.cat((text, target_text), dim=1)

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        # y["logit"] = self.fc_layer_1(self.dropout(concat))
        # y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(concat))
        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))
        self.internal_counter += 1
        return y

    def post_epoch(self, dataloader):
        with torch.no_grad():
            self.lora_on()
            test_text_accuracy = 0
            test_audio_accuracy = 0
            count = 0
            device = self.parameters().__next__().device
            for i, data in enumerate(dataloader):
                audio = data["audio"].to(device)
                text = data["text"].to(device)
                target_audio = data["target_audio"].to(device)
                target_text = data["target_text"].to(device)
                audio = audio[:, 0, :]
                target_audio = target_audio[:, 0, :]
                AB, AL = audio.shape
                TB, TL = text.shape

                # audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                audio = self.audio_model.model.feature_extractor(audio)
                audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                audio = self.audio_model.model.encoder(audio_embedding)[0]
                audio_feature = audio.mean(dim=1) 

                text_embedding = self.language_model.embeddings(text)
                text = self.language_model.encoder(text_embedding)[0]
                text_feature = text[:, 0, :]

                # target_audio = self.audio_model.processor(target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                target_audio = self.audio_model.model.feature_extractor(target_audio)
                target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))
                target_audio = self.audio_model.model.encoder(target_audio_embedding)[0]
                target_audio_feature = target_audio.mean(dim=1)

                target_text_embedding = self.language_model.embeddings(target_text)
                target_text = self.language_model.encoder(target_text_embedding)[0]
                target_text_feature = target_text[:, 0, :]

                audio_pred = self.audio_key_matcher(audio_feature)
                text_pred = self.text_key_matcher(text_feature)

                audio_label = F.cosine_similarity(audio_feature.unsqueeze(1), self.target_audio_centroids.unsqueeze(0), dim=2).argmax(dim=1)
                text_label = F.cosine_similarity(text_feature.unsqueeze(1), self.target_text_centriods.unsqueeze(0), dim=2).argmax(dim=1)

                test_audio_accuracy += (audio_pred.argmax(dim=1) == audio_label).float().sum() 
                test_text_accuracy += (text_pred.argmax(dim=1) == text_label).float().sum()
                count += AB
                print(f"post_epoch: {i}/{len(dataloader)}", end='\r')
            test_audio_accuracy /= count
            test_text_accuracy /= count
            print(f"audio accuracy: {test_audio_accuracy} text accuracy: {test_text_accuracy}")