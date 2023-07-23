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

class BPM_MT(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(BPM_MT, self).__init__()

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

        # self.encoder = nn.Sequential(*[SelfAttentionLayer(768, 8, dropout=dropout) for _ in range(3)])

        self.downproject = nn.Linear(768, 192)

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

        self.num_prompt = 40
        self.prompt_length = 5

        self.audio_prompt_keys = nn.Parameter(torch.randn(12, self.num_prompt, 768))
        self.text_prompt_keys = nn.Parameter(torch.randn(12, self.num_prompt, 768))

        self.audio_prompt_values = nn.Parameter(torch.randn(12, self.num_prompt, self.prompt_length * 768))
        self.text_prompt_values = nn.Parameter(torch.randn(12, self.num_prompt, self.prompt_length * 768))

        self.audio_mask = nn.Parameter(torch.randn(1, 1, 768))
        self.text_mask = nn.Parameter(torch.randn(1, 1, 768))

        self.audio_downproject = nn.Linear(768, 192)
        self.text_downproject = nn.Linear(768, 192)

        self.audio_decorder_pos = nn.Parameter(torch.randn(1, 70, 192))
        self.text_decorder_pos = nn.Parameter(torch.randn(1, 10, 192))

        self.audio_decorder = nn.Sequential(*[SelfAttentionLayer(192, 6, dropout=dropout) for _ in range(3)])
        self.text_decorder = nn.Sequential(*[SelfAttentionLayer(192, 6, dropout=dropout) for _ in range(3)])

        self.audio_predictor = nn.Linear(192 * 70, 24000)
        self.text_predictor = nn.Linear(192, 8002)

        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if self.mode == "audio_only" or self.mode == "text_only":
            self.fc_layer_1 = nn.Linear(768, output_size)
        elif self.mode == "flatten":
            self.fc_layer_1 = nn.Linear(768 * 65, output_size)
        else:
            self.fc_layer_1 = nn.Linear(768, output_size)
        self.relu = nn.ReLU()
        if self.mode == "hierarchical":
            self.classifier = nn.Linear(output_size, num_class - 1)
            self.BC_classifier = nn.Linear(output_size, 2)
        else:
            self.classifier = nn.Linear(output_size, num_class)

        self.BC_classifier = nn.Linear(output_size, 2)

        self.sentiment_fc_layer_1 = nn.Linear(768, sentiment_output_size)
        self.sentiment_relu = nn.ReLU()
        self.sentiment_classifier = nn.Linear(sentiment_output_size, 5)

    def audio_forward(self, x):
        y = {}
        audio = x["audio"]
        audio = audio[:, 0, :]
        AB, AL = audio.shape
        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)
        for i, audio_layer in enumerate(self.audio_model.model.encoder.layers):
            audio_similarity = torch.cosine_similarity(audio.mean(dim=1).unsqueeze(1), self.audio_prompt_keys[i].unsqueeze(0), dim=-1) / (len(self.audio_prompt_keys[i]) ** (2))
            audio_similarity = torch.softmax(audio_similarity, dim=-1)
            audio_prompt = torch.matmul(audio_similarity, self.audio_prompt_values[i]).reshape(-1, self.prompt_length, 768)
            audio = torch.cat((audio_prompt, audio), dim=1)
            audio = audio_layer(audio)[0]
            audio = audio[:, self.prompt_length:, :]
            audio_prompt = audio[:, :self.prompt_length, :]
        audio = audio_prompt
        audio = audio.mean(dim=1)
        x = self.fc_layer_1(self.dropout(audio))
        x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))

        return y
    
    def text_forward(self, x):
        y = {}
        text = x["text"]
        TB, TL = text.shape
        text = self.language_model.embeddings(text)
        for i, text_layer in enumerate(self.language_model.encoder.layer):
            text_similarity = torch.cosine_similarity(text.mean(dim=1).unsqueeze(1), self.text_prompt_keys[i].unsqueeze(0), dim=-1) / (len(self.audio_prompt_keys[i]) ** (2))
            text_similarity = torch.softmax(text_similarity, dim=-1)
            text_prompt = torch.matmul(text_similarity, self.text_prompt_values[i]).reshape(-1, self.prompt_length, 768)
            text = torch.cat((text_prompt, text), dim=1)
            text = text_layer(text)[0]
            text = text[:, self.prompt_length:, :]
            text_prompt = text[:, :self.prompt_length, :]
        text = text_prompt
        text = text.mean(dim=1)
        x = self.fc_layer_1(self.dropout(text))
        x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))

        return y

    def pretext_forward(self, x):

        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape

        audio = audio.reshape(AB, 10, -1)
        text = text.reshape(TB, 10, -1)

        original_audio = audio.clone()
        original_text = text.clone()

        audio = audio.reshape(AB*10, -1)
        text = text.reshape(TB*10, -1)

        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)

        text = self.language_model.embeddings(text)

        audio_select = torch.rand(AB, 10).argsort(dim=-1)[:, :5]
        text_select = torch.rand(TB, 10).argsort(dim=-1)[:,:5]

        audio = audio.reshape(AB, 10, -1, 768)
        text = text.reshape(TB, 10, -1, 768)

        audio = audio[torch.arange(AB).unsqueeze(-1), audio_select]
        text = text[torch.arange(TB).unsqueeze(-1), text_select]

        audio = audio.reshape(AB, -1, 768)
        text = text.reshape(TB, -1, 768)

        for i, (audio_layer, text_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            
            audio = audio_layer(audio)[0]
            text = text_layer(text)[0]

            if i > 9:
                _audio = self.before_audio_upproject[i](audio)
                _text = self.before_text_upproject[i](text)

                _audio = F.gelu(_audio)
                _text = F.gelu(_text)

                audio = self.before_audio_downproject[i](_audio) + audio
                text = self.before_text_downproject[i](_text) + text

                _audio = self.audio_to_text_attention[i](audio, text)
                _text = self.text_to_audio_attention[i](text, audio)

                audio = self.after_audio_upproject[i](_audio)
                text = self.after_text_upproject[i](_text)

                audio = F.gelu(audio)
                text = F.gelu(text)

                audio = self.after_audio_downproject[i](audio) + _audio
                text = self.after_text_downproject[i](text) + _text
            
        audio = self.audio_mask.expand(AB, 70, -1).reshape(AB, 10, 7, -1).index_put((torch.arange(AB).unsqueeze(-1), audio_select), audio.reshape(AB, 5, 7, -1)).reshape(AB, 70, -1)
        text = self.text_mask.expand(TB, 10, -1).reshape(TB, 10, 1, -1).index_put((torch.arange(TB).unsqueeze(-1), text_select), text.reshape(TB, 5, 1, -1)).reshape(TB, 10, -1)

        audio = self.audio_downproject(audio)
        text = self.text_downproject(text)

        audio = audio + self.audio_decorder_pos
        text = text + self.text_decorder_pos

        audio = self.audio_decorder(audio)
        text = self.text_decorder(text)

        audio = audio.reshape(AB, 192*70)
        text = text.reshape(TB, 10, 192)

        audio = self.audio_predictor(audio)
        text = self.text_predictor(text)

        loss = F.mse_loss(audio, original_audio.flatten(1,2)) + F.cross_entropy(text.flatten(0,1), original_text.flatten())   

        return loss

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

        audio = audio.reshape(AB, 10, -1)
        text = text.reshape(TB, 10)

        audio = audio.reshape(AB*10, -1)
        text = text.reshape(TB*10, -1)
        
        audio = self.audio_model.model.feature_extractor(audio)
        audio = self.audio_model.model.feature_projection(audio.transpose(1, 2))
        audio = self.audio_model.model.encoder.pos_conv_embed(audio)
        audio = self.audio_model.model.encoder.layer_norm(audio)
        audio = self.audio_model.model.encoder.dropout(audio)

        text = self.language_model.embeddings(text)

        audio = audio.reshape(AB, -1, 768)
        text = text.reshape(TB, -1, 768)

        for i, (audio_layer, text_layer) in enumerate(zip(self.audio_model.model.encoder.layers, self.language_model.encoder.layer)):
            
            # _audio = audio_layer(audio)[0]
            # _text = text_layer(text)[0]

            # audio_mean = _audio.mean(dim=1)
            # text_mean = _text.mean(dim=1)

            # audio_similarity = torch.cosine_similarity(audio_mean.unsqueeze(1), self.audio_prompt_keys[i].unsqueeze(0), dim=-1) / (len(self.audio_prompt_keys[i]) ** (2))
            # text_similarity = torch.cosine_similarity(text_mean.unsqueeze(1), self.text_prompt_keys[i].unsqueeze(0), dim=-1) / (len(self.audio_prompt_keys[i]) ** (2))
            # audio_similarity = torch.softmax(audio_similarity, dim=-1)
            # text_similarity = torch.softmax(text_similarity, dim=-1)
            # audio_prompt = torch.matmul(audio_similarity, self.audio_prompt_values[i]).reshape(-1, self.prompt_length, 768)
            # text_prompt = torch.matmul(text_similarity, self.text_prompt_values[i]).reshape(-1, self.prompt_length, 768)

            # audio = torch.cat((audio_prompt, audio), dim=1)
            # text = torch.cat((text_prompt, text), dim=1)

            audio = audio_layer(audio)[0]
            text = text_layer(text)[0]

            if i > 9:
                _audio = self.before_audio_upproject[i](audio)
                _text = self.before_text_upproject[i](text)

                _audio = F.gelu(_audio)
                _text = F.gelu(_text)

                audio = self.before_audio_downproject[i](_audio) + audio
                text = self.before_text_downproject[i](_text) + text

                _audio = self.audio_to_text_attention[i](audio, text)
                _text = self.text_to_audio_attention[i](text, audio)

                audio = self.after_audio_upproject[i](_audio)
                text = self.after_text_upproject[i](_text)

                audio = F.gelu(audio)
                text = F.gelu(text)

                audio = self.after_audio_downproject[i](audio) + _audio
                text = self.after_text_downproject[i](text) + _text
            
        #     audio_prompt = audio[:, :self.prompt_length, :]
        #     text_prompt = text[:, :self.prompt_length, :]

        #     audio = audio[:, self.prompt_length:, :]
        #     text = text[:, self.prompt_length:, :]

        # audio = audio_prompt
        # text = text_prompt
        
        # modalities = torch.cat((audio, text), dim=1).flatten(0,1)
        # modalities = self.discriminator(modalities)
        # modalities = F.cross_entropy(modalities,
        #                              torch.cat((torch.zeros(AB, self.prompt_length, dtype=torch.long, device=modalities.device),
        #                                         torch.ones(TB, self.prompt_length, dtype=torch.long, device=modalities.device)),
        #                                         dim=1).flatten(0,1))
        # y["modalities"] = modalities

        concat = torch.cat((audio, text), dim=1)
        concat = concat.mean(dim=1)
        x = self.fc_layer_1(self.dropout(concat))
        x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))
        
        return y