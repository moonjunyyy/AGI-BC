import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from util.utils import get_audio_model, get_language_model
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer

class Ours(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours, self).__init__()

        self.mode = mode

        self.register_module("language_model", language_model)
        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        self.register_module("audio_model", audio_model)
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()

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

    def k_means(self, x, k, max_iter=1000):
        N, D = x.shape
        c = x[torch.randperm(x.shape[0])[:k]]
        c = c.unsqueeze(1) # (k, 1, D)
        x = x.unsqueeze(0) # (1, N, D)
        for i in range(max_iter):
            diff = torch.cdist(c, x, p=2)
            diff = diff.squeeze(0)
            _, cluster = torch.min(diff, dim=0)
            for j in range(k):
                c[j] = x[cluster==j].mean(dim=0).unsqueeze(0)
        return c.squeeze(), cluster.squeeze()

    def pretext_forward(self, dataloader):
        with torch.no_grad():
            self.audios = []
            self.target_audios = []
            for x in dataloader:
                audio = x["audio"]
                target_audio = x["target_audio"]

                device = self.parameters().__next__().device
                audio = audio.to(device)
                target_audio = target_audio.to(device)

                audio = audio[:, 0, :]
                audio = self.audio_model.model.feature_extractor(audio)
                audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
                audio = self.audio_model.model.encoder.pos_conv_embed(audio)
                audio = self.audio_model.model.encoder.layer_norm(audio)
                audio = self.audio_model.model.encoder.dropout(audio)
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

            self.audios = torch.cat(self.audios, dim=0)
            self.target_audios = torch.cat(self.target_audios, dim=0)
            self.centriods, self.cluster = self.k_means(self.audios, 128)
            # make a prototype for each cluster of target audio
            self.target_centriods = []
            for i, _ in enumerate(self.centriods):
                self.target_centriods.append(self.target_audios[self.cluster==i].mean(dim=0))
            self.target_centriods = torch.stack(self.target_centriods, dim=0)

    def forward(self, x):
        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape

        original_audio = audio.clone()
        original_text = text.clone()

        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)

        _audio = self.audio_model.model.encoder(audio)[0]
        _audio = _audio.mean(dim=1)
        similarty = torch.cdist(_audio, self.centriods.to(_audio.device), p=2)
        _, cluster = torch.min(similarty, dim=1)
        target_audio = self.target_centriods.to(_audio.device)[cluster]
        audio = torch.cat((audio, target_audio), dim=1)

        audio = self.audio_model.model.encoder(audio)[0]
        text = self.language_model.embeddings(text)
        text = self.language_model.encoder(text)[0]

        audio = audio.reshape(AB, -1, 768).mean(dim=1)
        text = text.reshape(TB, -1, 768)[:, 0, :]

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        y["logit"] = self.fc_layer_1(self.dropout(concat))
        y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(y["logit"]))

        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))
        
        return y