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
        # for p in self.audio_model.parameters():
        #     p.requires_grad = False

        for name, param in self.audio_model.named_parameters():
            param.requires_grad = False
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False

        self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

        self.language_linear = nn.ModuleDict()
        self.language_lora = nn.ModuleDict()
        self.audio_linear = nn.ModuleDict()
        self.audio_lora = nn.ModuleDict()

        for name, modules in self.audio_model.named_modules():
            if isinstance(modules, nn.Linear):
                self.audio_lora[name.replace('.', '')] = LoRA(modules, 32, alpha=8)
                self.audio_linear[name.replace('.', '')] = modules

        for name, modules in self.language_model.named_modules():
            if isinstance(modules, nn.Linear):
                self.language_lora[name.replace('.', '')] = LoRA(modules, 32, alpha=8)
                self.language_linear[name.replace('.', '')] = modules

        self.num_cluster = 512

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

    def lora_on(self, requires_grad=True):
        for name, modules in self.audio_model.named_modules():
            # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            if isinstance(modules, nn.Linear) or isinstance(modules, LoRA):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.audio_lora[name.replace('.', '')])
                self.audio_lora[name.replace('.', '')].requires_grad = requires_grad
        for k, v in self.audio_lora.items():
            v.requires_grad_(requires_grad)

        for name, modules in self.language_model.named_modules():
            if isinstance(modules, nn.Linear) or isinstance(modules, LoRA):
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.language_lora[name.replace('.', '')])
                self.language_lora[name.replace('.', '')].requires_grad = requires_grad
        for k, v in self.language_lora.items():
            v.requires_grad_(requires_grad)

    def lora_off(self, requires_grad=True):
        for name, modules in self.audio_model.named_modules():
            # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            if isinstance(modules, nn.Linear) or isinstance(modules, LoRA):
                _name = name.split('.')
                _module = self.audio_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.audio_linear[name.replace('.', '')])
                self.audio_linear[name.replace('.', '')].requires_grad = requires_grad
        for k, v in self.audio_lora.items():
            v.requires_grad_(requires_grad)

        for name, modules in self.language_model.named_modules():
            if isinstance(modules, nn.Linear) or isinstance(modules, LoRA):
                _name = name.split('.')
                _module = self.language_model
                for i in range(len(_name)-1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], self.language_linear[name.replace('.', '')])
                self.language_linear[name.replace('.', '')].requires_grad = requires_grad
        for k, v in self.language_lora.items():
            v.requires_grad_(requires_grad)

    def k_means(self, x, k, max_iter=200, batch_size=64):
        N, D = x.shape
        c = x[torch.randperm(x.shape[0])[:k]]
        c = c.unsqueeze(1) # (k, 1, D)
        x = x # (1, N, D)
        for i in range(max_iter):
            print(f"{i+1}/{max_iter}", end='\r')
            # print(i)
            # diff = torch.cdist(c, x, p=2)
            _c = c.clone().detach()
            cluster = []
            for n in range(0, x.shape[0], batch_size):
                diff = torch.cdist(c, x[n:n+batch_size], p=2)
                # diff = 1 - torch.cosine_similarity(c, x[n:n+batch_size], dim=2)
                diff = diff.squeeze()
                _, _cluster = torch.min(diff, dim=0)
                cluster.append(_cluster)
            cluster = torch.cat(cluster, dim=0)
            for j in range(k):
                c[j] = x[cluster==j].mean(dim=0).unsqueeze(0)
            # diff = 1 - torch.cosine_similarity(c, x, dim=2)
            # diff = diff.squeeze()
            # _, cluster = torch.min(diff, dim=0)
            # for j in range(k):
                # c[j] = x[cluster==j].mean(dim=0).unsqueeze(0)
            if i > 0 and torch.equal(c, _c):
                break
        print()
        return c.squeeze(), cluster.squeeze()

    def pretext_forward(self, dataloader):
        self.audio_model.eval()
        self.language_model.eval()

        class MLP_Block(nn.Module):
            def __init__(self, input_size, output_size, dropout=0.3):
                super(MLP_Block, self).__init__()
                self.fc_layer_1 = nn.Linear(input_size, output_size + input_size)
                self.relu = nn.ReLU()
                self.fc_layer_2 = nn.Linear(output_size + input_size, output_size)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = x + self.relu(self.fc_layer_2(self.dropout(self.relu(self.fc_layer_1(x)))))
                return x
            
        self.audio_key_projection = nn.Sequential(
            MLP_Block(768, 768),
            MLP_Block(768, 768),
            MLP_Block(768, 768),
            nn.Linear(768, 768)
        ).to(next(self.parameters()).device)
        self.text_key_projection = nn.Sequential(
            MLP_Block(768, 768),
            MLP_Block(768, 768),
            MLP_Block(768, 768),
            nn.Linear(768, 768)
        ).to(next(self.parameters()).device)

        self.audio_keys = torch.randn(self.num_cluster, 768)
        self.text_keys = torch.randn(self.num_cluster, 768)

        audio_optimizer = torch.optim.Adam(self.audio_key_projection.parameters(), lr=1e-3, weight_decay=1e-5)
        audio_optimizer.add_param_group({'params': self.audio_keys})
        text_optimizer = torch.optim.Adam(self.text_key_projection.parameters(), lr=1e-3, weight_decay=1e-5)
        text_optimizer.add_param_group({'params': self.text_keys})
        
        self.lora_off()
        with torch.no_grad():
            self.audios = []
            self.target_audios = []
            self.texts = []
            self.target_texts = []
            self.labels = []
            for n, x in enumerate(dataloader):
                print(f"{n+1}/{len(dataloader)}", end='\r')
                audio = x["audio"]
                target_audio = x["target_audio"]

                device = self.parameters().__next__().device
                audio = audio.to(device)
                target_audio = target_audio.to(device)

                audio = audio[:, 0, :]
                audio = self.audio_model.model.feature_extractor(audio)
                audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                audio = self.audio_model.model.encoder(audio)[0]
                audio = audio.mean(dim=1)
                self.audios.append(audio)

                target_audio = target_audio[:, 0, :]
                target_audio = self.audio_model.model.feature_extractor(target_audio)
                target_audio = self.audio_model.model.feature_projection(target_audio.transpose(1, 2))
                target_audio = self.audio_model.model.encoder.pos_conv_embed(target_audio)
                target_audio = self.audio_model.model.encoder.layer_norm(target_audio)
                target_audio = self.audio_model.model.encoder.dropout(target_audio)
                self.target_audios.append(target_audio)

                text = x["text"]
                target_text = x["target_text"]

                device = self.parameters().__next__().device
                text = text.to(device)
                target_text = target_text.to(device)

                text = self.language_model.embeddings(text)
                text = self.language_model.encoder(text)[0]
                text = text[:, 0, :]
                self.texts.append(text)
                
                target_text = self.language_model.embeddings(target_text)
                self.target_texts.append(target_text)

                self.labels.append(x["label"])
            print()
            self.audios = torch.cat(self.audios, dim=0)
            self.target_audios = torch.cat(self.target_audios, dim=0)
            self.texts = torch.cat(self.texts, dim=0)
            self.target_texts = torch.cat(self.target_texts, dim=0)
            self.labels = torch.cat(self.labels, dim=0)

            # make a prototype for each cluster of audio
            self.audio_centroids = []
            self.audio_target_centriods = []
            self.audio_text_target_centriods = []
            # centroids, cluster = self.k_means(self.audios, 10)
            centroids, cluster = self.k_means(self.target_audios.flatten(1), self.num_cluster)
            self.audio_target_centriods.append(centroids.reshape(self.num_cluster, -1, 768))
            for i, _ in enumerate(centroids):
                self.audio_centroids.append(self.audios[cluster==i])
                self.audio_text_target_centriods.append(self.texts[cluster==i])
            self.audio_centroids = torch.cat(self.audio_centroids, dim=0)
            self.audio_target_centriods = torch.cat(self.audio_target_centriods, dim=0)
            self.audio_text_target_centriods = torch.cat(self.audio_text_target_centriods, dim=0)
            audio_cluster = cluster.clone()

            # make a prototype for each cluster of target audio
            self.text_centroids = []
            self.text_target_centriods = []
            self.text_audio_target_centriods = []
            # centroids, cluster = self.k_means(self.texts, 10)
            centroids, cluster = self.k_means(self.target_texts.flatten(1), self.num_cluster)
            self.text_target_centriods.append(centroids.reshape(self.num_cluster, -1, 768))
            for i, _ in enumerate(centroids):
                self.text_centroids.append(self.target_texts[cluster==i])
                self.text_audio_target_centriods.append(self.audios[cluster==i])
            self.text_centroids = torch.cat(self.text_centroids, dim=0)
            self.text_target_centriods = torch.cat(self.text_target_centriods, dim=0)
            self.text_audio_target_centriods = torch.cat(self.text_audio_target_centriods, dim=0)
            text_cluster = cluster.clone()

        for e in range(100):
            print(f"{e+1}/100", end='\r')
            index = torch.randperm(self.audios.shape[0])
            for i in range(0, self.audios.shape[0], 128):
                x = self.audios[index[i:i+128]]
                y = audio_cluster[index[i:i+128]]
                x = self.audio_key_projection(x)
                similarity = -torch.cdist(x, self.audio_keys.to(x.device), p=2)
                # similarity = torch.cosine_similarity(x.unsqueeze(1), self.audio_keys.to(x.device).unsqueeze(0), dim=2)
                loss = F.cross_entropy(similarity, y)
                audio_optimizer.zero_grad()
                loss.backward()
                audio_optimizer.step()

            index = torch.randperm(self.texts.shape[0])
            for i in range(0, self.texts.shape[0], 128):
                x = self.texts[index[i:i+128]]
                y = text_cluster[index[i:i+128]]
                x = self.text_key_projection(x)
                similarity = -torch.cdist(x, self.text_keys.to(x.device), p=2)
                # similarity = torch.cosine_similarity(x.unsqueeze(1), self.text_keys.to(x.device).unsqueeze(0), dim=2)
                loss = F.cross_entropy(similarity, y)
                text_optimizer.zero_grad()
                loss.backward()
                text_optimizer.step()
        print()
        for param in self.audio_key_projection.parameters():
            param.requires_grad = False
        for param in self.text_key_projection.parameters():
            param.requires_grad = False
        self.audio_keys.requires_grad = False
        self.text_keys.requires_grad = False
            # self.text_target_centriods = torch.stack(self.text_target_centriods, dim=0)
            # self.text_audio_target_centriods = torch.stack(self.text_audio_target_centriods, dim=0)

            # print(self.audio_centroids.shape)
            # print(self.audio_target_centriods.shape)

    def forward(self, x):

        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape

        self.lora_off()
        # original_audio = audio.clone()
        # original_text = text.clone()
        with torch.no_grad():
            audio = self.audio_model.model.feature_extractor(audio)
            audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
            audio = self.audio_model.model.encoder.pos_conv_embed(audio)
            audio = self.audio_model.model.encoder.layer_norm(audio)
            audio = self.audio_model.model.encoder.dropout(audio)
            _audio = audio.clone()
            for l, layer in enumerate(self.audio_model.model.encoder.layers):
                _audio = layer(_audio)[0]
            _audio = _audio.mean(dim=1)
            _audio = self.audio_key_projection(_audio)
            similarty = torch.cdist(_audio, self.audio_keys.to(_audio.device), p=2)
            # similarty = 1 - torch.cosine_similarity(_audio.unsqueeze(1), self.audio_keys.to(_audio.device).unsqueeze(0), dim=2)
            _, cluster = torch.min(similarty, dim=1)
            # similarty = 1 - torch.cosine_similarity(_audio.unsqueeze(1), self.audio_centroids.to(_audio.device).unsqueeze(0), dim=2)
            # _, cluster = torch.min(similarty, dim=1)
            target_audio = self.audio_target_centriods.to(_audio.device)[cluster]
        # target_audio_text = self.audio_text_target_centriods.to(_audio.device)[cluster]
        with torch.no_grad():
            text = self.language_model.embeddings(text)
            _text = text.clone()
            _text = self.language_model.encoder(_text)[0]
            _text = _text[:, 0, :]
            _text = self.text_key_projection(_text)
            similarty = torch.cdist(_text, self.text_keys.to(_text.device), p=2)
            # similarty = 1 - torch.cosine_similarity(_text.unsqueeze(1), self.text_keys.to(_text.device).unsqueeze(0), dim=2)
            _, cluster = torch.min(similarty, dim=1)
            # similarty = 1 - torch.cosine_similarity(_text.unsqueeze(1), self.text_centroids.to(_text.device).unsqueeze(0), dim=2)
            # _, cluster = torch.min(similarty, dim=1)
            target_text = self.text_target_centriods.to(_text.device)[cluster]
        # target_text_audio = self.text_audio_target_centriods.to(_text.device)[cluster]

        # with torch.no_grad():
        #     _audio = audio.clone()
        #     _text = text.clone()

        #     _target_audio = x["target_audio"]
        #     _target_text = x["target_text"]

        #     _target_audio = _target_audio[:, 0, :]
        #     _target_audio = self.audio_model.model.feature_extractor(_target_audio)
        #     _target_audio = self.audio_model.model.feature_projection(_target_audio.transpose(1, 2))
        #     _target_audio = self.audio_model.model.encoder.pos_conv_embed(_target_audio)
        #     _target_audio = self.audio_model.model.encoder.layer_norm(_target_audio)
        #     _target_audio = self.audio_model.model.encoder.dropout(_target_audio)

        #     _target_text = self.language_model.embeddings(_target_text)
        #     _target_text = self.language_model.encoder(_target_text)[0]
        #     _target_text = _target_text[:, 0, :]
        #     _target_text = _target_text.unsqueeze(1)

        #     _audio = torch.cat((_audio, _target_audio), dim=1)
        #     _text = torch.cat((_text, _target_text), dim=1)

        #     for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
        #         _audio = a_layer(_audio)[0]
        #         _text = t_layer(_text)[0]

        #     _audio = _audio.reshape(AB, -1, 768).mean(dim=1)
        #     _text = _text.reshape(TB, -1, 768)[:, 0, :]

        self.lora_on()
        audio = torch.cat((audio, target_audio), dim=1)
        text = torch.cat((text, target_text), dim=1)

        for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            audio = a_layer(audio)[0]
            text = t_layer(text)[0]
            # if l > 8:
            #     _audio = self.cross_attention_layer[l](audio, text)
            #     _text = self.cross_attention_layer[l](text, audio)
            # audio = _audio
            # text = _text

        audio = audio.reshape(AB, -1, 768).mean(dim=1)
        text = text.reshape(TB, -1, 768)[:, 0, :]

        # consistency_loss = F.mse_loss(audio, _audio) + F.mse_loss(text, _text)
        # y["consistency_loss"] = consistency_loss

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        y["logit"] = self.fc_layer_1(self.dropout(concat))
        y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(y["logit"]))
        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))

        return y