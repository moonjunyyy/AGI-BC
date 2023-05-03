import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

class HuBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
        self.model = AutoModel.from_pretrained("facebook/hubert-base-ls960")

    def forward(self, x):
        input_values = self.processor(x, return_tensors="pt", sample_rate=16000, padding=True).input_values
        outputs = self.model(input_values.squeeze(1))
        print(outputs.output_hidden_states.shape)
        return outputs.output_hidden_states
    
    def get_feature_size(self):
        return 768