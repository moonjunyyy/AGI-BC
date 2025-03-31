import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel
# MCG-NJU/videomae-base
# MCG-NJU/videomae-small-finetuned-kinetics

class VideoMAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")

    def forward(self, x):
    #    x = self.image_processor(list(x), do_resize=False, do_center_crop=False, data_format=channels_last, return_tensors="pt")
        device = self.model.parameters().__next__().device
        outputs = self.model(x.float().to(device))
        
        return outputs.last_hidden_state
    
    def get_feature_size(self):
        return 768