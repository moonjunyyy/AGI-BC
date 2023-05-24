import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.5):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Linear(d_model, d_model * 4)
        self.ffn_2 = nn.Linear(d_model * 4, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        x2, _ = self.self_attn(x, y, y)
        x = x + self.dropout(x2)
        x = self.norm_1(x)
        x2 = self.ffn_1(x)
        x2 = F.relu(x2)
        x2 = self.ffn_2(x2)
        x = x + self.dropout(x2)
        x = self.norm_2(x)
        return x

class BPM_MT(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3):
        super(BPM_MT, self).__init__()

        self.register_module("language_model", language_model)

        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None       

        self.register_module("audio_model", audio_model)
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()

        self.dropout = nn.Dropout(dropout)
        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        # self.fc_layer_1 = nn.Linear(77568, output_size)
        self.fc_layer_1 = nn.Linear(768*84, output_size * 4)
        # self.fc_layer_1 = nn.Linear(36096, output_size)
        self.fc_layer_2 = nn.Linear(output_size * 4, output_size)
        # self.fc_layer_3 = nn.Linear(output_size * 4, output_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

        # FC layer that has 64 of nodes which fed the text feature
        # FC layer that has 5 of nodes which fed the sentiment feature
        if self.is_MT:
            self.sentiment_fc_layer_1 = nn.Linear(768, sentiment_output_size)
            self.sentiment_relu = nn.ReLU()
            self.sentiment_classifier = nn.Linear(sentiment_output_size, 5)

        self.audio_downproject = nn.Linear(self.audio_feature_size, 256)
        self.text_downproject = nn.Linear(768, 256)

        self.audio_to_text_attention = nn.Sequential(*[CrossAttentionLayer(d_model=768, nhead=8, dropout=dropout) for _ in range(8)])
        self.audio_mask = nn.Parameter(torch.zeros(1, 1, self.audio_feature_size))
        
        self.text_to_audio_attention = nn.Sequential(*[CrossAttentionLayer(d_model=768, nhead=8, dropout=dropout) for _ in range(8)])
        self.text_mask = nn.Parameter(torch.zeros(1, 1, 768))

        self.audio_decoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True) for _ in range(2)])
        self.text_decoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True) for _ in range(2)])

        self.audio_predictor = nn.Linear(256*74, 24000)
        self.text_predictor = nn.Linear(256, 30522)

    def pretext_forward(self, x):
        audio = x["audio"]
        text  = x["text"]

        original_audio = audio.clone()
        original_text = text.clone()

        # tokenize the text if tokenizer is provided
        # if self.tokenizer is not None:

        text = self.language_model(text).last_hidden_state
        # text = text[:, 0, :]

        # text = self.language_model(text).logits
        y = {}
        
        # extract the MFCC feature from audio
        # audio = self.audio_model(audio)[:,:-1,:]
        audio = self.audio_model(audio)

        masked_audio = torch.rand_like(audio.mean(-1))<0.70
        masked_text = torch.rand_like(text.mean(-1))<0.40

        audio[masked_audio] = self.audio_mask
        text[masked_text] = self.text_mask

        _audio = audio.clone()
        _text = text.clone()
        for n, layer in enumerate(self.audio_to_text_attention):
            _audio = layer(_audio, text)
        for n, layer in enumerate(self.text_to_audio_attention):
            _text = layer(_text, audio)

        # for n, layer in enumerate(self.audio_to_text_attention):
        #     if n % 2 == 0:
        #         _audio = layer(_audio, text)
        #     else:
        #         _audio = layer(_audio, _audio)
        # for n, layer in enumerate(self.text_to_audio_attention):
        #     if n % 2 == 0:
        #         _text = layer(_text, audio)
        #     else:
        #         _text = layer(_text, _text)

        audio = self.audio_downproject(_audio)
        text = self.text_downproject(_text)

        audio = self.audio_decoder(audio)
        text = self.text_decoder(text)

        audio = self.audio_predictor(audio.flatten(start_dim=1))
        text = self.text_predictor(text)

        self.pretext_loss = F.mse_loss(audio, original_audio) + F.cross_entropy(text.transpose(-1,-2), original_text)
        
        return self.pretext_loss

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        
        # tokenize the text if tokenizer is provided
        # if self.tokenizer is not None:

        text = self.language_model(text).last_hidden_state
        # text = text[:, 0, :]

        # text = self.language_model(text).logits
        y = {}
        
        # extract the MFCC feature from audio
        # audio = self.audio_model(audio)[:,:-1,:]
        audio = self.audio_model(audio)

        _audio = audio.clone()
        _text = text.clone()
        for n, layer in enumerate(self.audio_to_text_attention):
            _audio = layer(_audio, text)
        for n, layer in enumerate(self.text_to_audio_attention):
            _text = layer(_text, audio)

        # for n, layer in enumerate(self.audio_to_text_attention):
        #     if n % 2 == 0:
        #         _audio = layer(_audio, text)
        #     else:
        #         _audio = layer(_audio, _audio)
        # for n, layer in enumerate(self.text_to_audio_attention):
        #     if n % 2 == 0:
        #         _text = layer(_text, audio)
        #     else:
        #         _text = layer(_text, _text)
        
        x = torch.cat((_audio, _text), dim=1).flatten(start_dim=1)
        # print(x.shape)
        x = self.fc_layer_1(self.dropout(x))
        x = self.relu(x)
        x = self.fc_layer_2(self.dropout(x))
        x = self.relu(x)
        # x = self.fc_layer_3(self.dropout(x))
        # x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))

        if self.is_MT:
            sentiment = self.sentiment_fc_layer_1(self.dropout(_text[:,0]))
            sentiment = self.sentiment_relu(sentiment)
            y["sentiment"] = self.sentiment_classifier(sentiment)
        
        return y