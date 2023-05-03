import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class BPM_MT(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3):
        super(BPM_MT, self).__init__()

        # get the bert model and tokenizer from arguments        
        # tokenizer = SentencepieceTokenizer(get_tokenizer())
        # bert, vocab = get_pytorch_kobert_model()
        self.register_module("language_model", language_model)

        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None       

        self.audio_model = audio_model
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()

        self.dropout = nn.Dropout(dropout)
        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        self.fc_layer_1 = nn.Linear(768 + self.audio_feature_size, output_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

        # FC layer that has 64 of nodes which fed the text feature
        # FC layer that has 5 of nodes which fed the sentiment feature
        if self.is_MT:
            self.sentiment_fc_layer_1 = nn.Linear(768, sentiment_output_size)
            self.sentiment_relu = nn.ReLU()
            self.sentiment_classifier = nn.Linear(sentiment_output_size, 5)

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        
        # tokenize the text if tokenizer is provided
        # if self.tokenizer is not None:

        text = self.language_model(text).last_hidden_state
        text = text[:, 0, :]

        # text = self.language_model(text).logits
        y = {}
        
        # extract the MFCC feature from audio
        audio = self.audio_model(audio)
        # audio = audio.reshape(audio.shape[0], -1)

        x = torch.cat((audio, text), dim=1)
        x = self.fc_layer_1(self.dropout(x))
        x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))

        if self.is_MT:
            sentiment = self.sentiment_fc_layer_1(self.dropout(text))
            sentiment = self.sentiment_relu(sentiment)
            y["sentiment"] = self.sentiment_classifier(sentiment)
        
        return y