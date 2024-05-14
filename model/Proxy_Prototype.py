import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from utils.utils import get_audio_model, get_language_model
from utils.kmeans import KMeans
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE
import logging
import matplotlib.pyplot as plt

class Proxy_Prototype(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Proxy_Prototype, self).__init__()

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

        self.internal_counter = 0
        self.num_cluster = 2 ** 8
        # self.num_inner_cluster = 32
        self.mem_size = 2 ** 12

        class ResMLP(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, out_dim + in_dim)
                self.fc2 = nn.Linear(out_dim + in_dim, out_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
            def forward(self, x):
                return x + self.dropout(self.relu(self.fc2(self.relu(self.fc1(x)))))
        self.audio_key_matcher = nn.Sequential(
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            nn.Linear(768, self.num_cluster)
        ).to(self.parameters().__next__().device)
        self.audio_key_matcher.train()
        self.text_key_matcher = nn.Sequential(
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            ResMLP(768, 768),
            nn.Linear(768, self.num_cluster)
        ).to(self.parameters().__next__().device)

        self.audio_key_matcher_optimizer = torch.optim.Adam(self.audio_key_matcher.parameters(), lr=1e-3)
        self.text_key_matcher_optimizer = torch.optim.Adam(self.text_key_matcher.parameters(), lr=1e-3)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768 + self.audio_model.get_feature_size(), num_class)
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
    
    def pre_epoch(self, dataloader):
        self.lora_on()
        if self.training:
            device = self.parameters().__next__().device
            # optimizer = torch.optim.Adam([p for n, p in self.named_parameters() if 'lora' not in n], lr=1e-3)

            self.audio_features = torch.empty(0).to(device)
            self.text_features = torch.empty(0).to(device)
            self.target_audio_embeddings = torch.empty(0).to(device)
            self.target_text_embeddings = torch.empty(0).to(device)

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

                    audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                    audio = self.audio_model.model.feature_extractor(audio)
                    audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                    audio = self.audio_model.model.encoder(audio_embedding.detach())[0]
                    audio = audio.mean(dim=1)
                    self.audio_features = torch.cat((self.audio_features, audio.detach()), dim=0)

                    target_audio = self.audio_model.processor(target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
                    target_audio = self.audio_model.model.feature_extractor(target_audio)
                    target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))
                    self.target_audio_embeddings = torch.cat((self.target_audio_embeddings, target_audio_embedding), dim=0)

                    text_embedding = self.language_model.embeddings(text)
                    text = self.language_model.encoder(text_embedding)[0]
                    text = text[:, 0, :]
                    self.text_features = torch.cat((self.text_features, text.detach()), dim=0)

                    target_text_embedding = self.language_model.embeddings(target_text)
                    self.target_text_embeddings = torch.cat((self.target_text_embeddings, target_text_embedding.detach()), dim=0)

                    print(f"pre_epoch: {i}/{len(dataloader)}", end='\r')

                # num_samples = audio_features.shape[0]
                # self.target_audio_cluster = KMeans(n_clusters=self.num_cluster, max_iter=100, batchsize=1024, mode='euclidean', init='kmeans++', seed=None)
                self.target_audio_clustering = KMeans(n_clusters=self.num_cluster, max_iter=100, batchsize=1024, mode='cosine', init='kmeans++', seed=None)
                self.audio_clusters = self.target_audio_clustering.fit_predict(self.target_audio_embeddings.flatten(1)).clone()
                self.audio_target_centriods = self.target_audio_clustering.get_centroids().clone().reshape(self.num_cluster, -1, 768)
                # del self.target_audio_cluster

                # self.target_text_cluster = KMeans(n_clusters=self.num_cluster, max_iter=100, batchsize=1024, mode='euclidean', init='kmeans++', seed=None)
                self.target_text_clustering = KMeans(n_clusters=self.num_cluster, max_iter=100, batchsize=1024, mode='cosine', init='kmeans++', seed=None)
                self.text_clusters = self.target_text_clustering.fit_predict(self.target_text_embeddings.flatten(1)).clone()
                self.text_target_centriods = self.target_text_clustering.get_centroids().clone().reshape(self.num_cluster, -1, 768)
                # del self.target_text_cluster

            # optimizer.zero_grad()
            # audio_positive_map = (audio_cluster.unsqueeze(1) == audio_cluster.unsqueeze(0)).to(torch.float32)
            # text_positive_map = (text_cluster.unsqueeze(1) == text_cluster.unsqueeze(0)).to(torch.float32)

            # audio_distance_map = 1 - F.cosine_similarity(audio.unsqueeze(1), audio.unsqueeze(0), dim=2)
            # text_distance_map = 1 - F.cosine_similarity(text.unsqueeze(1), text.unsqueeze(0), dim=2)

            # audio_loss = (audio_positive_map * audio_distance_map).sum() / audio_positive_map.sum()
            # text_loss = (text_positive_map * text_distance_map).sum() / text_positive_map.sum()

            # loss = audio_loss + text_loss
            # loss.backward()
            # optimizer.step()
            # print(f"loss: {loss.item()}")

            for param in self.audio_key_matcher.parameters():
                param.requires_grad = True
            for param in self.text_key_matcher.parameters():
                param.requires_grad = True
            audio_acc = 0
            while audio_acc < 0.95:
                self.audio_key_matcher_optimizer.zero_grad()
                audio_pred = self.audio_key_matcher(self.audio_features)
                audio_loss = F.cross_entropy(audio_pred, self.audio_clusters)
                audio_loss.backward()
                self.audio_key_matcher_optimizer.step()
                audio_acc = (audio_pred.argmax(dim=1) == self.audio_clusters).float().mean()
                print(f"audio_acc: {audio_acc.item()}", end='\r')
            print()
            text_acc = 0
            while text_acc < 0.95:
                self.text_key_matcher_optimizer.zero_grad()
                text_pred = self.text_key_matcher(self.text_features)
                text_loss = F.cross_entropy(text_pred, self.text_clusters)
                text_loss.backward()
                self.text_key_matcher_optimizer.step()
                text_acc = (text_pred.argmax(dim=1) == self.text_clusters).float().mean()
                print(f"text_acc: {text_acc.item()}", end='\r')
            print()
            for param in self.audio_key_matcher.parameters():
                param.requires_grad = False
            for param in self.text_key_matcher.parameters():
                param.requires_grad = False

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
                    
        audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
        audio = self.audio_model.model.feature_extractor(audio)
        audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.audio_model.model.encoder(audio_embedding)[0]
        audio = audio.mean(dim=1)

        text_embedding = self.language_model.embeddings(text)
        text = self.language_model.encoder(text_embedding)[0]
        text = text[:, 0, :]

        target_audio = self.audio_model.processor(target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
        target_audio = self.audio_model.model.feature_extractor(target_audio)
        target_audio = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))

        target_text = self.language_model.embeddings(target_text)

        audio_weight = self.audio_key_matcher(audio).softmax(dim=1)
        audio_label = audio_weight.argmax(dim=1)
        target_audio = (self.audio_target_centriods.unsqueeze(0) * audio_weight.unsqueeze(2).unsqueeze(3)).sum(dim=1)
        text_weight = self.text_key_matcher(text).softmax(dim=1).to(device)
        text_label = text_weight.argmax(dim=1)
        target_text = (self.text_target_centriods.unsqueeze(0) * text_weight.unsqueeze(2).unsqueeze(3)).sum(dim=1)

        audio_positive_map = (audio_label.unsqueeze(1) == audio_label.unsqueeze(0)).to(torch.float32)
        text_positive_map = (text_label.unsqueeze(1) == text_label.unsqueeze(0)).to(torch.float32)

        audio_distance_map = 1 - F.cosine_similarity(audio_weight.unsqueeze(1), audio_weight.unsqueeze(0), dim=2)
        text_distance_map = 1 - F.cosine_similarity(text_weight.unsqueeze(1), text_weight.unsqueeze(0), dim=2)

        audio_loss = (audio_positive_map * audio_distance_map).sum() / (audio_distance_map.sum() + 1e-8)
        text_loss = (text_positive_map * text_distance_map).sum() / (text_distance_map.sum() + 1e-8)
        y['contrastive_loss'] = audio_loss + text_loss

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
                # _audio = __audio
                # _text = __text

                _audio = self.audio_model.model.encoder.pos_conv_embed(_audio)
                _audio = self.audio_model.model.encoder.layer_norm(_audio)
                _audio = self.audio_model.model.encoder.dropout(_audio)
                for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
                    _audio = a_layer(_audio)[0]
                    _text = t_layer(_text)[0]

                _audio = _audio.reshape(AB, -1, 768).mean(dim=1)
                _text = _text.reshape(TB, -1, 768)[:, 0, :]

            # if self.multi_modal:
            #     audio = torch.cat((audio, target_audio, target_text_audio), dim=1)
            #     text = torch.cat((text, target_text, target_audio_text), dim=1)
            # else:
                
            audio = torch.cat((audio_embedding, target_audio), dim=1)
            text = torch.cat((text_embedding, target_text), dim=1)

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
        # y["logit"] = self.fc_layer_1(self.dropout(concat))
        # y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(concat))
        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))
        self.internal_counter += 1
        return y
    
    def post_epoch(self, dataloader):
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

            audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
            audio = self.audio_model.model.feature_extractor(audio)
            audio_embedding = self.audio_model.model.feature_projection(audio.transpose(1, 2))
            audio = self.audio_model.model.encoder(audio_embedding)[0]
            audio_feature = audio.mean(dim=1)

            text_embedding = self.language_model.embeddings(text)
            text = self.language_model.encoder(text_embedding)[0]
            text_feature = text[:, 0, :]

            target_audio = self.audio_model.processor(target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
            target_audio = self.audio_model.model.feature_extractor(target_audio)

            target_audio_embedding = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))
            target_text_embedding = self.language_model.embeddings(target_text)

            audio_lable = self.audio_key_matcher(target_audio_embedding)
            text_lable = self.text_key_matcher(target_text_embedding)

            audio_target = F.cosine_similarity(target_audio_embedding.unsqueeze(1), self.audio_target_centriods.unsqueeze(0)).argmin(dim=1)
            text_target = F.cosine_similarity(target_text_embedding.unsqueeze(1), self.text_target_centriods.unsqueeze(0)).argmin(dim=1)

            test_audio_accuracy += (audio_lable.argmax(dim=1) == audio_target).float().sum() 
            test_text_accuracy += (text_lable.argmax(dim=1) == text_target).float().sum()
            count += AB
            print(f"post_epoch: {i}/{len(dataloader)}", end='\r')
        test_audio_accuracy /= count
        test_text_accuracy /= count
        print(f"audio accuracy: {test_audio_accuracy} text accuracy: {test_text_accuracy}")