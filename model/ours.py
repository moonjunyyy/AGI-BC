import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from layer.cross_attention_layer import CrossAttentionLayer
from layer.self_attention_layer import SelfAttentionLayer

class Ours(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(Ours, self).__init__()

        self.register_module("language_model", language_model)
        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        permute = list(permutations([0,1,2,3,4]))
        permute = torch.tensor(permute).long()
        self.register_buffer("permute", permute)

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None

        self.register_module("audio_model", audio_model)
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()

        self.encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True) for _ in range(4)])
        
        self.audio_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        self.text_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        
        self.full_downproject = nn.Linear(768, 192)
        self.audio_downproject = nn.Linear(768, 192)   
        self.text_downproject = nn.Linear(768, 192)

        # self.encoder = nn.Sequential(*[SelfAttentionLayer(768, 8, dropout=dropout) for _ in range(3)])

        self.downproject = nn.Linear(768, 192)

        self.num_prompt = 20
        self.prompt_length = 5

        self.encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True) for _ in range(4)])

        self.audio_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        self.text_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])

        self.mode = mode
        self.dropout = nn.Dropout(dropout)

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

    def pretext_forward(self, x):
        audio = x["audio"][:,0,:]
        text  = x["text"]

        B, L = audio.shape

        original_text = text.clone()
        original_audio = audio.clone()

        generation_target_audio = x["target_audio"][:,0,:]
        generation_target_text = x["target_text"]

        audio = F.pad(audio, (0, L//3), "constant", 0)
        input_ids = []
        attention_mask = []
        for idx, att in zip(text['input_ids'], text['attention_mask']):
            sep = (idx==3).nonzero()
            idx[sep:sep+5]=0; idx[sep+5]=3
            att[sep:sep+5]=0; att[sep+5]=1
            

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

        text = self.language_model.embeddings(text)

        audio = audio.reshape(AB, -1, 768)
        text = text.reshape(TB, -1, 768)

        _audio = audio.clone()
        _text = text.clone()

        for i, (audio_layer, text_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            _audio = audio_layer(_audio)[0]
            _text = text_layer(_text)[0]

        # audio_q = self.audio_q_proj(_audio)
        # audio_k = self.audio_k_proj(_audio)
        # audio_v = self.audio_v_proj(_audio)

        # text_q = self.text_q_proj(_text)
        # text_k = self.text_k_proj(_text)
        # text_v = self.text_v_proj(_text)

        # audio_q = audio_q.reshape(AB, -1, 12, 64).transpose(1,2)
        # audio_k = audio_k.reshape(AB, -1, 12, 64).transpose(1,2)
        # audio_v = audio_v.reshape(AB, -1, 12, 64).transpose(1,2)

        # text_q = text_q.reshape(TB, -1, 12, 64).transpose(1,2)
        # text_k = text_k.reshape(TB, -1, 12, 64).transpose(1,2)
        # text_v = text_v.reshape(TB, -1, 12, 64).transpose(1,2)

        # audio_attention = torch.matmul(audio_q, audio_k.transpose(2,3)) / (768 ** (1/2))
        # audio_attention = torch.softmax(audio_attention, dim=-1)
        # _audio = torch.matmul(audio_attention, audio_v)
        # _audio = _audio.transpose(1,2).reshape(AB, -1, 768)
        # _audio = self.audio_output(_audio)

        # text_attention = torch.matmul(text_q, text_k.transpose(2,3)) / (768 ** (1/2))
        # text_attention = torch.softmax(text_attention, dim=-1)
        # _text = torch.matmul(text_attention, text_v)
        # _text = _text.transpose(1,2).reshape(TB, -1, 768)
        # _text = self.text_output(_text)

        # _audio = _audio + audio
        # _text = _text + text

        audio_mean = _audio.mean(dim=1)
        text_mean = _text[:, 0, :]
        
        audio_audio_similarity = torch.cosine_similarity(audio_mean.unsqueeze(1), self.audio_prompt_keys[0].unsqueeze(0), dim=-1)# / (len(self.audio_prompt_keys[0]) ** (2))
        text_text_similarity = torch.cosine_similarity(text_mean.unsqueeze(1), self.text_prompt_keys[0].unsqueeze(0), dim=-1)# / (len(self.audio_prompt_keys[0]) ** (2))
        # audio_audio_similarity = F.softmax(audio_audio_similarity, dim=-1)
        # text_text_similarity = F.softmax(text_text_similarity, dim=-1)
        audio_text_similarity = torch.cosine_similarity(audio_mean.unsqueeze(1), self.text_prompt_keys[0].unsqueeze(0), dim=-1)# / (len(self.audio_prompt_keys[0]) ** (2))
        text_audio_similarity = torch.cosine_similarity(text_mean.unsqueeze(1), self.audio_prompt_keys[0].unsqueeze(0), dim=-1)# / (len(self.audio_prompt_keys[0]) ** (2))
        # audio_text_similarity = F.softmax(audio_text_similarity, dim=-1)
        # text_audio_similarity = F.softmax(text_audio_similarity, dim=-1)

        # print(audio_audio_similarity.shape)

        # audio_audio_prompt_k = [self.audio_prompt_values_k[i, audio_audio_similarity.argmax(dim=1)].reshape(-1, self.prompt_length, 768) for i in range(12)]
        # audio_audio_prompt_v = [self.audio_prompt_values_v[i, audio_audio_similarity.argmax(dim=1)].reshape(-1, self.prompt_length, 768) for i in range(12)]
        # text_text_prompt_k = [self.text_prompt_values_k[i, text_text_similarity.argmax(dim=1)].reshape(-1, self.prompt_length, 768) for i in range(12)]
        # text_text_prompt_v = [self.text_prompt_values_v[i, text_text_similarity.argmax(dim=1)].reshape(-1, self.prompt_length, 768) for i in range(12)]

        # y["audio_audio_similarity"] = (1 - audio_audio_similarity.max(dim=-1)[0]).sum(dim=-1)
        # y["text_text_similarity"] = (1 - text_text_similarity.max(dim=-1)[0]).sum(dim=-1)

        audio_audio_prompt_k = [torch.matmul(audio_audio_similarity, self.audio_prompt_values_k[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        audio_audio_prompt_v = [torch.matmul(audio_audio_similarity, self.audio_prompt_values_v[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        text_text_prompt_k = [torch.matmul(text_text_similarity, self.text_prompt_values_k[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        text_text_prompt_v = [torch.matmul(text_text_similarity, self.text_prompt_values_v[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]

        audio_text_prompt_k = [torch.matmul(audio_text_similarity, self.text_prompt_values_k[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        audio_text_prompt_v = [torch.matmul(audio_text_similarity, self.text_prompt_values_v[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        text_audio_prompt_k = [torch.matmul(text_audio_similarity, self.audio_prompt_values_k[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        text_audio_prompt_v = [torch.matmul(text_audio_similarity, self.audio_prompt_values_v[i]).reshape(-1, self.prompt_length, 768) for i in range(12)]
        
        for i, (audio_layer, text_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            
            if i < -1:
                audio_query = audio_layer.attention.q_proj(audio)
                # audio_key = audio_layer.attention.k_proj(torch.cat((audio_audio_prompt_k[i], audio), dim=1))
                # audio_value = audio_layer.attention.v_proj(torch.cat((audio_audio_prompt_v[i], audio), dim=1))
                audio_key = audio_layer.attention.k_proj(torch.cat((audio_audio_prompt_k[i], text_audio_prompt_k[i], audio), dim=1))
                audio_value = audio_layer.attention.v_proj(torch.cat((audio_audio_prompt_v[i], text_audio_prompt_v[i], audio), dim=1))

                audio_query = audio_layer.dropout(audio_query).reshape(AB, -1, 12, 64).transpose(1,2)
                audio_key = audio_layer.dropout(audio_key).reshape(AB, -1, 12, 64).transpose(1,2)
                audio_value = audio_layer.dropout(audio_value).reshape(AB, -1, 12, 64).transpose(1,2)

                audio_attention = torch.matmul(audio_query, audio_key.transpose(2,3)) / (768 ** (1/2))
                audio_attention = torch.softmax(audio_attention, dim=-1)
                _audio = torch.matmul(audio_attention, audio_value)
                _audio = _audio.transpose(1,2).reshape(AB, -1, 768)

                _audio = audio_layer.attention.out_proj(_audio)
                _audio = audio_layer.dropout(_audio)
                _audio = audio_layer.layer_norm(_audio)

                audio = audio + _audio

                _audio = audio_layer.feed_forward(audio)
                _audio = audio_layer.final_layer_norm(_audio)

                audio = audio + _audio

                text_query = text_layer.attention.self.query(text)
                # text_key = text_layer.attention.self.key(torch.cat((text_text_prompt_k[i], text), dim=1))
                # text_value = text_layer.attention.self.value(torch.cat((text_text_prompt_v[i], text), dim=1))
                text_key = text_layer.attention.self.key(torch.cat((text_text_prompt_k[i], audio_text_prompt_k[i], text), dim=1))
                text_value = text_layer.attention.self.value(torch.cat((text_text_prompt_v[i], audio_text_prompt_k[i], text), dim=1))

                text_query = text_layer.attention.self.dropout(text_query).reshape(TB, -1, 12, 64).transpose(1,2)
                text_key = text_layer.attention.self.dropout(text_key).reshape(TB, -1, 12, 64).transpose(1,2)
                text_value = text_layer.attention.self.dropout(text_value).reshape(TB, -1, 12, 64).transpose(1,2)

                text_attention = torch.matmul(text_query, text_key.transpose(2,3)) / (768 ** (1/2))
                text_attention = torch.softmax(text_attention, dim=-1)
                _text = torch.matmul(text_attention, text_value)
                _text = _text.transpose(1,2).reshape(TB, -1, 768)

                _text = text_layer.attention.output.dense(_text)
                _text = text_layer.attention.output.LayerNorm(_text)
                _text = text_layer.attention.output.dropout(_text)

                text = text + _text

                _text = text_layer.intermediate(text)
                _text = text_layer.output.dense(_text)
                _text = text_layer.output.dropout(_text)
                _text = text_layer.output.LayerNorm(_text)

                text = text + _text

                audio = audio_layer(audio)[0]
                text = text_layer(text)[0]
            
            elif i < 9:
                audio = audio_layer(audio)[0]
                text = text_layer(text)[0]

            else:
                audio = audio_layer(audio)[0]
                text = text_layer(text)[0]

                _audio = self.before_audio_upproject[i](audio)
                _text = self.before_text_upproject[i](text)

                _audio = F.gelu(_audio)
                _text = F.gelu(_text)

                audio = self.before_audio_downproject[i](_audio) + audio
                text = self.before_text_downproject[i](_text) + text

                # _audio = self.audio_to_text_attention[i](audio, text)
                # _text = self.text_to_audio_attention[i](text, audio)

                AB, AL, _ = audio.shape
                TB, TL, _ = text.shape
                concat = torch.cat((audio, text), dim=1)
                concat = self.audio_text_self_attention[i](concat)
                _audio = concat[:, :AL, :]
                _text = concat[:, AL:, :]

                audio = self.after_audio_upproject[i](_audio)
                text = self.after_text_upproject[i](_text)

                audio = F.gelu(audio)
                text = F.gelu(text)

                audio = self.after_audio_downproject[i](audio) + _audio
                text = self.after_text_downproject[i](text) + _text
            # audio_prompt = audio[:, :self.prompt_length, :]
            # text_prompt = text[:, :self.prompt_length, :]

            # audio = audio[:, self.prompt_length:, :]
            # text = text[:, self.prompt_length:, :] 

        # audio = audio_prompt
        # text = text_prompt
        
        # modalities = torch.cat((audio, text), dim=1).flatten(0,1)
        # modalities = self.discriminator(modalities)
        # modalities = F.cross_entropy(modalities,
        #                              torch.cat((torch.zeros(AB, self.prompt_length, dtype=torch.long, device=modalities.device),
        #                                         torch.ones(TB, self.prompt_length, dtype=torch.long, device=modalities.device)),
        #                                         dim=1).flatten(0,1))
        # y["modalities"] = modalities

        audio = audio.mean(dim=1)
        text = text[:, 0, :]

        # audio = self.audio_model(audio)[:, 0, :]
        # text = self.language_model(text).last_hidden_state[:, 0, :]

        if self.mode == "audio_only" or self.mode == "text_only":
            concat = audio if self.mode == "audio_only" else text
        else :
            concat = torch.cat((audio, text), dim=1)
        y["logit"] = self.fc_layer_1(self.dropout(concat))
        y["logit"] = self.relu(y["logit"])
        y["logit"] = self.classifier(self.dropout(y["logit"]))

        # y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))
        
        return y
    
    class Ours(nn.Module):
        def __init__(self, language_model=None, audio_model=None, output_size=128, num_class=4, dropout=0.3, mode="cross_entropy") -> None:
            super().__init__()

            if self.mode != 'audio_only':
                self.register_module("language_model", language_model)
                # if bert and vocab are not provided, raise an error
                assert self.language_model is not None, "bert must be provided"
            if self.mode != 'text_only':
                self.register_module("audio_model", audio_model)
                # define the LSTM layer, 4 of layers
                self.audio_feature_size = audio_model.get_feature_size()
                assert self.audio_model is not None, "audio_model must be provided"
            

            pass

        def forward(self, x):
            pass