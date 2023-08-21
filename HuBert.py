import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from transformers import Wav2Vec2FeatureExtractor

class HuBert(nn.Module):
    def __init__(self, sample_rate) -> None:
        super().__init__()
        # self.processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960") 
        self.model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        self.sample_rate = sample_rate

    def forward(self, x):
        input_values = self.processor(x.squeeze(1), return_tensors="pt", sampling_rate=self.sample_rate, padding=True).input_values.squeeze()
        device = self.model.parameters().__next__().device
        outputs = self.model(input_values.to(device))
        return outputs.last_hidden_state
    
    def get_feature_size(self):
        return 768