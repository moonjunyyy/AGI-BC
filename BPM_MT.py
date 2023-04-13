import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from selfattention import SelfAttention

class BPM_ST(nn.Module):
    def __init__(self, tokenizer=None, bert=None, vocab=None, mfcc_extractor=None ,trans = None, hubert = None ,output_size=128, num_class=4, dropout=0.3):
        super(BPM_ST, self).__init__()

        # get the bert model and tokenizer from arguments        
        # tokenizer = SentencepieceTokenizer(get_tokenizer())
        # bert, vocab = get_pytorch_kobert_model()
        self.bert = bert
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.trans = None # trans
        self.hubert = None # hubert
        
        # if bert and vocab are not provided, raise an error
        assert self.bert is not None, "bert and vocab must be provided"
        
        # define the MFCC extractor
        # self.mfcc_extractor = MFCC(sample_rate=sample_rate,n_mfcc=13)
        self.mfcc_extractor = mfcc_extractor
        self.audio_feature_size = mfcc_extractor.n_mfcc

        self.dropout = nn.Dropout(dropout)

        # define the LSTM layer, 4 of layers
        self.LSTM = nn.LSTM(input_size=self.audio_feature_size, hidden_size=self.audio_feature_size, num_layers=4, batch_first=True, bidirectional=True)

        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        self.fc_layer = nn.Linear(768 + 2*self.audio_feature_size, output_size)
        # FC layer that has 4 of nodes which fed the text feature
        self.classifier = nn.Linear(output_size, num_class)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        audio = x["audio"].squeeze(1) #64,1,12000 -> 64,12000
        text  = x["text"]
        batch_size = audio.size(0)
        text = self.bert(text).last_hidden_state
        
        # extract the text feature from bert model
        if self.trans is None:
            text = text[:, 0, :].squeeze(1)
        else:
            text = text[:, :, :]
            
        if self.hubert is not None:
            haudio = self.Hubert(audio).last_hidden_state
        

        y = {}
        
        # base code
        # extract the MFCC feature from audio
        audio = self.mfcc_extractor(audio) # 64,13,61
        # reshape the MFCC feature to (batch_size, length, 13)
        audio = audio.permute(0, 2, 1)
        # pass the MFCC feature to LSTM layer
        audio, _ = self.LSTM(audio) # 64,61,26
        audio = audio[:,-1,:].squeeze(1)
        
        x = torch.cat((audio, text), dim=1)
        x = self.fc_layer(self.dropout(x))
        y["logit"] = self.classifier(self.dropout(x))
        
        return y

class BPM_MT(nn.Module):
    def __init__(self, tokenizer=None, bert=None, vocab=None, mfcc_extractor=None, sentiment_dict = None, trans = None, hubert = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.3):
        super(BPM_MT, self).__init__()

        # get the bert model and tokenizer from arguments        
        # tokenizer = SentencepieceTokenizer(get_tokenizer())
        # bert, vocab = get_pytorch_kobert_model()
        self.bert = bert
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.trans = None #trans
        self.hubert = None #hubert
        # if bert and vocab are not provided, raise an error
        assert self.bert is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None
        
        # define the MFCC extractor
        # self.mfcc_extractor = MFCC(sample_rate=sample_rate,n_mfcc=13)
        self.mfcc_extractor = mfcc_extractor
        self.audio_feature_size = mfcc_extractor.n_mfcc

        # define the LSTM layer, 4 of layers
        self.LSTM = nn.LSTM(input_size=self.audio_feature_size, hidden_size=self.audio_feature_size, num_layers=6, batch_first=True, bidirectional=True)

        # self.accustic_fc_layer = nn.Linear(self.audio_feature_size * 2, int(output_size/2))
        # self.bert_fc_layer = nn.Linear(768, output_size-int(output_size/2))
        
        self.dropout = nn.Dropout(dropout)
        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        self.fc_layer = nn.Linear(768 + self.audio_feature_size * 2, output_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class)

        # FC layer that has 64 of nodes which fed the text feature
        # FC layer that has 5 of nodes which fed the sentiment feature
        if self.is_MT:
            self.sentiment_fc_layer = nn.Linear(768, sentiment_output_size)
            self.sentiment_relu = nn.ReLU()
            self.sentiment_classifier = nn.Linear(sentiment_output_size, 5)

    def forward(self, x):
        audio = x["audio"].squeeze(1) #64,1,12000 -> 64,12000
        text  = x["text"]
        batch_size = audio.size(0)
        text = self.bert(text).last_hidden_state
        
        # extract the text feature from bert model
        if self.trans is None:
            text = text[:, 0, :].squeeze(1)
        else:
            text = text[:, :, :]
            
        #if self.hubert is not None:
            
        
        
        y = {}
        
        # extract the MFCC feature from audio
        audio = self.mfcc_extractor(audio) # 64,13,61
        # reshape the MFCC feature to (batch_size, length, 13)
        audio = audio.permute(0, 2, 1)
        # pass the MFCC feature to LSTM layer
        audio, _ = self.LSTM(audio) # 64,61,26
        audio = audio[:,-1,:].squeeze(1)

        
        
        x = torch.cat((audio, text), dim=1)
        x = self.fc_layer(self.dropout(x))
        y["logit"] = self.classifier(self.dropout(x))

        if self.is_MT:
            # pass the text feature to sentiment FC layer
            sentiment = self.sentiment_fc_layer(self.dropout(text))
            sentiment = self.sentiment_relu(sentiment)
            y["sentiment"] = self.sentiment_classifier(sentiment)
        
        return y