import torch
import torch.nn as nn

# BPM_ST is the model for the single task learning
class BPM_ST(nn.Module):
    def __init__(self, language_model=None, audio_model=None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(BPM_ST, self).__init__()

        self.mode = mode
        if self.mode != "audio_only":
            self.register_module("language_model", language_model)
            # if bert and vocab are not provided, raise an error
            assert self.language_model is not None, "bert must be provided"
        if self.mode != "text_only":
            self.register_module("audio_model", audio_model)
            # define the LSTM layer, 4 of layers
            self.audio_feature_size = audio_model.get_feature_size()
            assert self.audio_model is not None, "audio_model must be provided"

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if self.mode == "audio_only" or self.mode == "text_only":
            self.fc_layer_1 = nn.Linear(768, output_size)
        elif self.mode == "flatten":
            self.fc_layer_1 = nn.Linear(sum(self._get_feature_size()), output_size)
        else:
            self.fc_layer_1 = nn.Linear(768 + self.audio_feature_size, output_size)
        if self.mode == "hierarchical":
            self.classifier = nn.Linear(output_size, num_class - 1)
            self.BC_classifier = nn.Linear(output_size, 2)
        else:
            self.classifier = nn.Linear(output_size, num_class)

    def forward(self, x):
        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape

        if self.mode != "text_only":
            audio = self.audio_model(audio)
        if self.mode != "audio_only":
            text, _ = self.language_model(text)
        
        if self.mode == "flatten":
            audio = audio.reshape(AB, -1)
            concat = torch.cat((audio, text.reshape(TB, -1)), dim=1)
            text = text[:,0,:]
        elif self.mode == "audio_only":
            concat = audio.mean(dim=1)
        elif self.mode == "text_only":
            concat = text[:, 0, :]
        else:
            concat = torch.cat((audio, text), dim=1)

        x = self.fc_layer_1(self.dropout(concat))
        x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))
        
        return y
    
    def _get_feature_size(self):
        with torch.no_grad():
            dummy_audio = torch.randn(2, 24000)
            dummy_text = torch.ones(2, 20).long()
            dummy_audio = self.audio_model(dummy_audio)
            dummy_text, _ = self.language_model(dummy_text)
            dummy_audio = dummy_audio.reshape(2, -1)
            dummy_text = dummy_text.reshape(2, -1)
        return dummy_audio.shape[-1], dummy_text.shape[-1]

# BPM_MT is the model for the multi task learning
class BPM_MT(nn.Module):
    def __init__(self, language_model=None, audio_model=None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3, mode="cross_entropy"):
        super(BPM_MT, self).__init__()

        self.mode = mode
        if self.mode != "audio_only":
            self.register_module("language_model", language_model)
            # if bert and vocab are not provided, raise an error
            assert self.language_model is not None, "bert must be provided"
        if self.mode != "text_only":
            self.register_module("audio_model", audio_model)
            # define the LSTM layer, 4 of layers
            self.audio_feature_size = audio_model.get_feature_size()
            assert self.audio_model is not None, "audio_model must be provided"

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if self.mode == "audio_only" or self.mode == "text_only":
            self.fc_layer_1 = nn.Linear(768, output_size)
        elif self.mode == "flatten":
            self.fc_layer_1 = nn.Linear(sum(self._get_feature_size()), output_size)
        else:
            self.fc_layer_1 = nn.Linear(768 + self.audio_feature_size, output_size)
        if self.mode == "hierarchical":
            self.classifier = nn.Linear(output_size, num_class - 1)
            self.BC_classifier = nn.Linear(output_size, 2)
        else:
            self.classifier = nn.Linear(output_size, num_class)

        self.sentiment_fc_layer_1 = nn.Linear(768, sentiment_output_size)
        self.sentiment_relu = nn.ReLU()
        self.sentiment_classifier = nn.Linear(sentiment_output_size, 5)

    def forward(self, x):
        y = {}

        audio = x["audio"]
        text  = x["text"]

        # get audio only one channel
        audio = audio[:, 0, :]

        AB, AL = audio.shape
        TB, TL = text.shape

        if self.mode != "text_only":
            audio = self.audio_model(audio)
        if self.mode != "audio_only":
            text, _ = self.language_model(text)
        
        if self.mode == "flatten":
            audio = audio.reshape(AB, -1)
            concat = torch.cat((audio, text.reshape(TB, -1)), dim=1)
            text = text[:,0,:]
        elif self.mode == "audio_only":
            concat = audio.mean(dim=1)
        elif self.mode == "text_only":
            concat = text[:, 0, :]
        else:
            concat = torch.cat((audio, text), dim=1)

        x = self.fc_layer_1(self.dropout(concat))
        x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))

        y["sentiment"] = self.sentiment_classifier(self.dropout(self.sentiment_relu(self.sentiment_fc_layer_1(self.dropout(text)))))
        
        return y
    
    def _get_feature_size(self):
        with torch.no_grad():
            dummy_audio = torch.randn(2, 24000)
            dummy_text = torch.ones(2, 20).long()
            dummy_audio = self.audio_model(dummy_audio)
            dummy_text, _ = self.language_model(dummy_text)
            dummy_audio = dummy_audio.reshape(2, -1)
            dummy_text = dummy_text.reshape(2, -1)
        return dummy_audio.shape[-1], dummy_text.shape[-1]