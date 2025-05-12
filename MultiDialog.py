import torch
from datasets import load_dataset

MultiD = load_dataset("IVLLab/MultiDialog", "train", use_auth_token=True)
# MultiD = load_dataset("IVLLab/MultiDialog", "valid_freq", use_auth_token=True)
# MultiD = load_dataset("IVLLab/MultiDialog", "valid_freq", use_auth_token=True)
# MultiD = load_dataset("IVLLab/MultiDialog", "valid_freq", use_auth_token=True)
# see structure
print(MultiD)

# load audio sample on the fly
audio_input = MultiD["valid_freq"][0]["audio"]  # first decoded audio sample
transcription = MultiD["valid_freq"][0]["value"]  # first transcription

