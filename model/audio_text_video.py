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
from model.video_mae import VideoMAE
import logging
import matplotlib.pyplot as plt

class Audio_Text_Video(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Audio_Text_Video, self).__init__()

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
        
        self.video_model = VideoMAE()

        self.dropout = nn.Dropout(dropout)
        self.fc_layer_1 = nn.Linear(768 + self.audio_model.get_feature_size()+384, output_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

    def forward(self, x):
        
        y = {}

        audio = x["audio"]
        text  = x["text"]
        video = x["video"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape
        
        device = self.parameters().__next__().device

        # get the audio feature
        audio = self.audio_model(audio)
        audio = audio.mean(dim=1)
        audio = self.dropout(audio)

        # get the text feature
        text = self.language_model(text)
        text = text[:, 0, :]
        text = self.dropout(text)

        # get the video feature
        video = self.video_model(video)
        video = video[:, 0, :]
        video = self.dropout(video)

        # concatenate the audio and text feature
        x = torch.cat([audio, text, video], dim=1)
        
        x = self.fc_layer_1(x)
        x = self.relu(x)
        x = self.classifier(x)
        y["logits"] = x

        return y