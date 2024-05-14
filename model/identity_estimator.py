import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from utils.utils import get_audio_model, get_language_model
from layer.lora import LoRA
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Identity_Estimator(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Identity_Estimator, self).__init__()

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

        # for name, param in self.audio_model.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.language_model.named_parameters():
        #     param.requires_grad = False

        self.cross_attention_layer = nn.ModuleList([CrossAttentionLayer(768, 4, 0.5) for _ in range(12)])

        self.language_linear = nn.ModuleDict()
        self.language_lora = nn.ModuleDict()
        self.audio_linear = nn.ModuleDict()
        self.audio_lora = nn.ModuleDict()

        # for name, module in self.audio_model.named_modules():
        #     if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
        #         self.audio_lora[name.replace('.', '_')] = LoRA(module, 64, alpha=16)
        #         self.audio_linear[name.replace('.', '_')] = module

        # for name, module in self.language_model.named_modules():
        #     if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
        #         self.language_lora[name.replace('.', '_')] = LoRA(module, 64, alpha=16)
        #         self.language_linear[name.replace('.', '_')] = module

        self.num_cluster = 1024
        self.num_inner_cluster = 32
        self.mem_size = 60000

        # self.deidentifier = nn.Sequential(
        #     nn.Linear(768, 384),
        #     nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu'),
        #     nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu'),
        #     nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu'),
        #     nn.Linear(384, 768)
        # )
        # self.id_discriminator = nn.Sequential(
        #     nn.Linear(768, 384),
        #     nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu'),
        #     nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu'),
        #     nn.TransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu'),
        #     nn.Linear(384, 768)
        # )

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
            self.classifier = nn.Linear(768 + self.audio_model.get_feature_size(), num_class)
        self.BC_classifier = nn.Linear(output_size, 2)
        self.identity_classifier = nn.Linear(768 + self.audio_model.get_feature_size(), 174)

    # def lora_on(self):
    #     for name, module in self.audio_model.named_modules():
    #         # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
    #         if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
    #             _name = name.split('.')
    #             _module = self.audio_model
    #             for i in range(len(_name)-1):
    #                 _module = _module.__getattr__(_name[i])
    #             _module.__setattr__(_name[-1], self.audio_lora[name.replace('.', '_')])

    #     for name, module in self.language_model.named_modules():
    #         if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
    #             _name = name.split('.')
    #             _module = self.language_model
    #             for i in range(len(_name)-1):
    #                 _module = _module.__getattr__(_name[i])
    #             _module.__setattr__(_name[-1], self.language_lora[name.replace('.', '_')])

    # def lora_off(self):
    #     for name, module in self.audio_model.named_modules():
    #         # if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
    #         if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'output_dense' in name or 'intermediate_dense' in name):
    #             _name = name.split('.')
    #             _module = self.audio_model
    #             for i in range(len(_name)-1):
    #                 _module = _module.__getattr__(_name[i])
    #             _module.__setattr__(_name[-1], self.audio_linear[name.replace('.', '_')])

    #     for name, module in self.language_model.named_modules():
    #         if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name or 'output.dense' in name or 'intermediate.dense' in name):
    #             _name = name.split('.')
    #             _module = self.language_model
    #             for i in range(len(_name)-1):
    #                 _module = _module.__getattr__(_name[i])
    #             _module.__setattr__(_name[-1], self.language_linear[name.replace('.', '_')])

    def forward(self, x):
        
        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape
        
        device = self.parameters().__next__().device
        # self.lora_off()
        # original_audio = audio.clone()
        # original_text = text.clone()
        
        # self.lora_on()

        audio = self.audio_model.processor(audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        # audio = self.deidentifier(audio)

        text = self.language_model.embeddings(text)

        # if self.consistency:
        #     with torch.no_grad():
        #         __audio = audio.clone()
        #         __text = text.clone()

        #         _target_audio = x["target_audio"]
        #         _target_text = x["target_text"]

        #         _target_audio = _target_audio[:, 0, :]
        #         _target_audio = self.audio_model.processor(_target_audio.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).input_values.squeeze().to(device)
        #         _target_audio = self.audio_model.model.feature_extractor(_target_audio)
        #         _target_audio = self.audio_model.model.feature_projection(_target_audio.transpose(1, 2))
        #         # _target_audio = self.audio_model.model.encoder.pos_conv_embed(_target_audio)
        #         # _target_audio = self.audio_model.model.encoder.layer_norm(_target_audio)
        #         # _target_audio = self.audio_model.model.encoder.dropout(_target_audio)

        #         _target_text = self.language_model.embeddings(_target_text)
        #         # _target_text = self.language_model.encoder(_target_text)[0]
        #         # _target_text = _target_text.unsqueeze(1)

        #         _audio = torch.cat((audio, _target_audio), dim=1)
        #         _text = torch.cat((text, _target_text), dim=1)
        #         # _audio = __audio
        #         # _text = __text

        #         _audio = self.audio_model.model.encoder.pos_conv_embed(_audio)
        #         _audio = self.audio_model.model.encoder.layer_norm(_audio)
        #         _audio = self.audio_model.model.encoder.dropout(_audio)
        #         for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
        #             _audio = a_layer(_audio)[0]
        #             _text = t_layer(_text)[0]

        #         _audio = _audio.reshape(AB, -1, 768).mean(dim=1)
        #         _text = _text.reshape(TB, -1, 768)[:, 0, :]

        # # if self.multi_modal:
        # #     audio = torch.cat((audio, target_audio, target_text_audio), dim=1)
        # #     text = torch.cat((text, target_text, target_audio_text), dim=1)
        # # else:
        # audio = torch.cat((audio, target_audio), dim=1)
        # text = torch.cat((text, target_text), dim=1)

        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)
        for l, (a_layer, t_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            audio = a_layer(audio)[0]
            text = t_layer(text)[0]
            # if self.cross_attn:
            #     if l > 8:
            #         __audio = self.cross_attention_layer[l](audio, text)
            #         __text = self.cross_attention_layer[l](text, audio)
            #         audio = __audio
            #         text = __text

        audio = audio.reshape(AB, -1, 768).mean(dim=1)
        text = text.reshape(TB, -1, 768)[:, 0, :]

        # if self.consistency:
        #     consistency_loss = F.mse_loss(audio, _audio) + F.mse_loss(text, _text)
        #     y["consistency_loss"] = consistency_loss

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        # y["logit"] = self.fc_layer_1(self.dropout(concat))
        # y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(concat))

        y["identity"] = F.cross_entropy(self.identity_classifier(self.dropout(concat)), x["identity"])
        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))

        return y