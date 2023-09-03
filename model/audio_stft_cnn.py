import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.utils import save_image

class AudioStftCnn(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(*[
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 768, kernel_size=(3, 3), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(1),
        ])

    def forward(self, x):
        B, L = x.shape
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=800, hop_length=200, win_length=800, power=2.0).to(x.device)
        x = spectrogram(x)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(B, 1, -1)
        return x
    
    def get_feature_size(self):
        return 768
    
if __name__ == '__main__':
    model = AudioStftCnn()
    x = torch.randn(1, 16000)
    print(x.shape, '=>', end=' ')
    y = model(x)
    print(y.shape)