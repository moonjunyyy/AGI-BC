import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer

class BPM_ST(nn.Module):
    def __init__(self, hidden_size=128, num_classes=2, dr_rate=None, params=None):
        super(BPM_ST, self).__init__()
        self.tokenizer = SentencepieceTokenizer(get_tokenizer())
        self.bert, self.vocab = get_pytorch_kobert_model()
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dr_rate = dr_rate

        self.LSTM = nn.LSTM(input_size=13, hidden_size=13, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(781, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        y = {}

        audio = self.LSTM(audio)
        _, text  = self.bert(self.vocab(self.tokenizer(text)))
        x = torch.cat((audio, text), dim=1)
        x = self.fc_layer(x)
        y["logit"] = self.classifier(x)
        return y

class BPM_MT(nn.Module):
    def __init__(self, hidden_size=128, sentiment_hidden_size=64, num_classes=2, dr_rate=None, params=None):
        super(BPM_MT, self).__init__()

        self.tokenizer = SentencepieceTokenizer(get_tokenizer())
        self.model, self.vocab = get_pytorch_kobert_model()
        for param in self.model.parameters():
            param.requires_grad = False
                
        self.fc_layer = nn.Linear(781, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.sentiment_fc_layer = nn.Linear(781, sentiment_hidden_size)
        self.sentiment_classifier = nn.Linear(sentiment_hidden_size, 1)
        
    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        y = {}

        audio = self.LSTM(audio)
        _, text  = self.model(self.vocab(self.tokenizer(text)))
        x = torch.cat((audio, text), dim=1)
        x = self.fc_layer(x)
        y["logit"] = self.classifier(x)

        x = self.sentiment_fc_layer(text)
        y["sentiment"] = self.sentiment_classifier(x)
        return y