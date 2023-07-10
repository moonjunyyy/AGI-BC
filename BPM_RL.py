import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # self.norm_1_q = nn.LayerNorm(d_model)
        # self.norm_1_k = nn.LayerNorm(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Linear(d_model, d_model * 4)
        self.ffn_2 = nn.Linear(d_model * 4, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, y):

        # x = self.norm_1_q(x)
        # y = self.norm_1_k(y)
        
        x = self.norm_1(x)
        y = self.norm_1(y)

        x = self.dropout(x)
        y = self.dropout(y)
        x2, _ = self.self_attn(x, y, y)
        x = x + x2

        x = self.norm_2(x)
        x2 = self.ffn_1(self.dropout(x))
        x2 = F.relu(x2)
        x2 = self.ffn_2(self.dropout(x2))
        x = x + x2
        
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Linear(d_model, d_model * 4)
        self.ffn_2 = nn.Linear(d_model * 4, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm_1(x)
        x = self.dropout(x)
        x2, _ = self.self_attn(x, x, x)
        x = x + x2

        x = self.norm_2(x)
        x2 = self.ffn_1(self.dropout(x))
        x2 = F.relu(x2)
        x2 = self.ffn_2(self.dropout(x2))
        x = x + x2
        return x

class BPM_RL(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=5, sentiment_output_size=64, dropout=0.3):
        super(BPM_RL, self).__init__()

        self.register_module("language_model", language_model)
        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.register_module("audio_model", audio_model)
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()

        self.encoder = nn.Sequential(*[SelfAttentionLayer(768, 12, dropout=dropout) for _ in range(6)])

        self.before_audio_upproject =  nn.Sequential(*[nn.Linear(768, 768*4) for _ in range(12)])
        self.before_text_upproject =  nn.Sequential(*[nn.Linear(768, 768*4) for _ in range(12)])

        self.before_audio_downproject =  nn.Sequential(*[nn.Linear(768*4, 768) for _ in range(12)])
        self.before_text_downproject =  nn.Sequential(*[nn.Linear(768*4, 768) for _ in range(12)])

        self.audio_to_text_attention = nn.Sequential(*[CrossAttentionLayer(768, 8, dropout=dropout) for _ in range(12)])
        self.text_to_audio_attention = nn.Sequential(*[CrossAttentionLayer(768, 8, dropout=dropout) for _ in range(12)])

        self.after_audio_upproject =  nn.Sequential(*[nn.Linear(768, 768*4) for _ in range(12)])
        self.after_text_upproject =  nn.Sequential(*[nn.Linear(768, 768*4) for _ in range(12)])

        self.after_audio_downproject =  nn.Sequential(*[nn.Linear(768*4, 768) for _ in range(12)])
        self.after_text_downproject =  nn.Sequential(*[nn.Linear(768*4, 768) for _ in range(12)])

        self.dropout = nn.Dropout(dropout)
        self.fc_layer_1 = nn.Linear(768, output_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

        self.text_query = nn.Linear(768, 768)
        self.audio_query = nn.Linear(768, 768)
        self.text_key = nn.Linear(768, 768)
        self.audio_key = nn.Linear(768, 768)

        # FC layer that has 64 of nodes which fed the text feature
        # FC layer that has 5 of nodes which fed the sentiment feature
    
    def forward(self, x):

        y = {}

        audio = x["audio"]
        text  = x["text"]
        label = []

        AB, AN, AC, AL = audio.shape
        TB, TN, TL = text.shape
        
        self.audio_context = torch.zeros(AB, 20, 768).to(audio.device)
        self.text_context = torch.zeros(TB, 20, 768).to(text.device)

        for n in range(AN):
            a = audio[:, n, 0, :]
            t = text[:, n, :]
            
            a = self.audio_model.model.feature_extractor(a.squeeze())
            a = self.audio_model.model.feature_projection(a.transpose(-1,-2))
            a = self.audio_model.model.encoder.pos_conv_embed(a)
            t = self.language_model.embeddings(t)

            a = a.reshape(AB, -1, 768)
            t = t.reshape(TB, -1, 768)

            a = torch.cat((self.audio_context, a), dim=1)
            t = torch.cat((self.text_context, t), dim=1)

            for n, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
                _audio = a_layer(a)[0]
                _text = t_layer(t)[0]

                __audio = self.before_audio_upproject[n](_audio)
                __text = self.before_text_upproject[n](_text)

                __audio = F.gelu(__audio)
                __text = F.gelu(__text)

                _audio = self.before_audio_downproject[n](__audio) + _audio
                _text = self.before_text_downproject[n](__text) + _text

                a = self.audio_to_text_attention[n](_audio, _text)
                t = self.text_to_audio_attention[n](_text, _audio)

                _audio = self.after_audio_upproject[n](a)
                _text = self.after_text_upproject[n](t)

                _audio = F.gelu(_audio)
                _text = F.gelu(_text)

                a = self.after_audio_downproject[n](_audio) + a
                t = self.after_text_downproject[n](_text) + t

            a = a.reshape(AB, -1, 768)
            t = t.reshape(TB, -1, 768)

            aq = self.audio_query(a)
            tq = self.text_query(t)
            ak = self.audio_key(a)
            tk = self.text_key(t)
            av = torch.matmul(aq, ak.transpose(-1,-2)) / (768 ** 0.5)
            tv = torch.matmul(tq, tk.transpose(-1,-2)) / (768 ** 0.5)
            _, ai = av.sum(dim=-1).topk(20, dim=1)
            _, ti = tv.sum(dim=-1).topk(20, dim=1)

            self.audio_context = a[torch.arange(AB).unsqueeze(-1), ai]
            self.text_context = t[torch.arange(TB).unsqueeze(-1), ti]

            l = torch.cat((self.audio_context, self.text_context), dim=1)
            l = self.fc_layer_1(self.dropout(l.mean(dim=1)))
            l = self.relu(l)
            l = self.classifier(l)
            label.append(l)

        pred = torch.stack(label, dim=1)
        label = x["label"]
        # print(pred.shape, label.shape)
        # print(pred.flatten(0,1).shape, label.flatten(0,1).shape)
        # print(label)
        y["loss"] = F.cross_entropy(pred.flatten(0,1), label.flatten(0,1))
        y["pred"] = pred[:, -1]

        return y