import torch
import torch.nn as nn

class BPM_ST(nn.Module):
    def __init__(self, tokenizer=None, bert=None, vocab=None, mfcc_extractor=None, output_size=128, dropout=0.3):
        super(BPM_ST, self).__init__()

        # get the bert model and tokenizer from arguments        
        # tokenizer = SentencepieceTokenizer(get_tokenizer())
        # bert, vocab = get_pytorch_kobert_model()
        self.bert = bert
        self.vocab = vocab
        self.tokenizer = tokenizer
        # if bert and vocab are not provided, raise an error
        assert self.bert is not None and self.vocab is not None, "bert and vocab must be provided"
        
        # define the MFCC extractor
        # self.mfcc_extractor = MFCC(sample_rate=sample_rate,n_mfcc=13)
        self.mfcc_extractor = mfcc_extractor
        self.audio_feature_size = mfcc_extractor.n_mfcc

        self.dropout = nn.Dropout(dropout)

        # define the LSTM layer, 4 of layers
        self.LSTM = nn.LSTM(input_size=self.audio_feature_size, hidden_size=self.audio_feature_size, num_layers=4, batch_first=True, bidirectional=True)

        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        self.fc_layer = nn.Linear(768 + self.audio_feature_size, output_size)

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]

        # tokenize the text if tokenizer is provided
        if self.tokenizer is not None:
            text = self.tokenizer(text)
        # convert the text to index
        text = self.vocab(text)
        # extract the text feature from bert model
        _, text = self.bert(text)
        y = {}
        
        # extract the MFCC feature from audio
        audio = self.mfcc_extractor(audio)
        # reshape the MFCC feature to (batch_size, length, 13)
        audio = audio.permute(0, 2, 1)
        # pass the MFCC feature to LSTM layer
        audio, _ = self.LSTM(self.dropout(audio))

        # concatenate the audio and text feature
        x = torch.cat((audio, text), dim=1)
        # pass the concatenated feature to FC layer
        y["logit"] = self.fc_layer(self.dropout(x))
        
        return y

class BPM_MT(nn.Module):
    def __init__(self, tokenizer=None, bert=None, vocab=None, mfcc_extractor=None, output_size=128, sentiment_output_size=64, dropout=0.3):
        super(BPM_ST, self).__init__()

        # get the bert model and tokenizer from arguments        
        # tokenizer = SentencepieceTokenizer(get_tokenizer())
        # bert, vocab = get_pytorch_kobert_model()
        self.bert = bert
        self.vocab = vocab
        self.tokenizer = tokenizer
        # if bert and vocab are not provided, raise an error
        assert self.bert is not None and self.vocab is not None, "bert and vocab must be provided"
        
        # define the MFCC extractor
        # self.mfcc_extractor = MFCC(sample_rate=sample_rate,n_mfcc=13)
        self.mfcc_extractor = mfcc_extractor
        self.audio_feature_size = mfcc_extractor.n_mfcc

        # define the LSTM layer, 4 of layers
        self.LSTM = nn.LSTM(input_size=self.audio_feature_size, hidden_size=self.audio_feature_size, num_layers=4, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        self.fc_layer = nn.Linear(768 + self.audio_feature_size, output_size)
        self.sentiment_fc_layer = nn.Linear(768, sentiment_output_size)

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        
        # tokenize the text if tokenizer is provided
        if self.tokenizer is not None:
            text = self.tokenizer(text)
        # convert the text to index
        text = self.vocab(text)
        # extract the text feature from bert model
        _, text = self.bert(text)
        y = {}
        
        # extract the MFCC feature from audio
        audio = self.mfcc_extractor(audio)
        # reshape the MFCC feature to (batch_size, length, 13)
        audio = audio.permute(0, 2, 1)
        # pass the MFCC feature to LSTM layer
        audio, _ = self.LSTM(audio)

        # concatenate the audio and text feature
        x = torch.cat((audio, text), dim=1)
        # pass the concatenated feature to FC layer
        y["logit"] = self.fc_layer(self.dropout(x))

        # pass the concatenated feature to sentiment FC layer
        y["sentiment"] = self.sentiment_fc_layer(self.dropout(text))
        
        return y