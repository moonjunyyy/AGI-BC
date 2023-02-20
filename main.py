from BPM_MT import BPM_MT, BPM_ST

import torch
import torch.nn.functional as F

from torchaudio.transforms import MFCC
from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer

is_MT = True    # True for MT, False for ST
use_CUDA = True # True for GPU, False for CPU
batch_size = 64 # Batch size
epochs = 60     # Number of epochs

def main():
    
    mfcc_extractor = MFCC(sample_rate=16000, n_mfcc=13)
    tokenizer = SentencepieceTokenizer(get_tokenizer())
    bert, vocab = get_pytorch_kobert_model()

    train_dataset = None # Dataset not implemented yet
    val_dataset = None   # Dataset not implemented yet

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if is_MT:
        model = BPM_MT(tokenizer=tokenizer, bert=bert, vocab=vocab, mfcc_extractor=mfcc_extractor)
    else:
        model = BPM_ST(tokenizer=tokenizer, bert=bert, vocab=vocab, mfcc_extractor=mfcc_extractor)
    
    # Get the model parameters divided into two groups : bert and others
    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
            

    adam_optimizer = torch.optim.Adam(bert_params, lr=0.0001)
    sgd_optimizer = torch.optim.SGD(other_params, lr=0.0001)

    # Training loop
    for epoch in range(epochs):
        for batch in train_dataloader:
            # Move the batch to GPU if CUDA is available
            if use_CUDA:
                for key in batch:
                    batch[key] = batch[key].cuda()

            y = model(batch)

            # Get the logit from the model
            logit     = y["logit"]
            sentiment = y["sentiment"]

            # Calculate the loss
            loss_BC = F.cross_entropy(logit, batch["label"])
            if is_MT:
                loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), batch["sentiment"])
                loss = 0.9 * loss_BC +  0.1 * loss_SP
            else:
                loss = loss_BC

            # Backpropagation
            loss.backward()

            # Update the model parameters
            adam_optimizer.step()
            sgd_optimizer.step()

            # Zero the gradients
            adam_optimizer.zero_grad()
            sgd_optimizer.zero_grad()

            print("Epoch : {}, Loss : {}".format(epoch, loss.item()))

        # Validation loop
        accuracy = 0
        loss     = 0
        for batch in val_dataloader:
            # Move the batch to GPU if CUDA is available
            if use_CUDA:
                for key in batch:
                    batch[key] = batch[key].cuda()

            y = model(batch)

            # Get the logit from the model
            logit     = y["logit"]
            sentiment = y["sentiment"]

            # Calculate the loss
            loss_BC = F.cross_entropy(logit, batch["label"])
            if is_MT:
                loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), batch["sentiment"])
                loss = 0.9 * loss_BC +  0.1 * loss_SP
            else:
                loss = loss_BC

            # Calculate the accuracy
            accuracy += (torch.argmax(logit, dim=1) == batch["label"]).sum().item()
            loss    += loss.item()

        accuracy /= len(val_dataset)
        loss     /= len(val_dataloader)
        print("Epoch : {}, Accuracy : {}, Loss : {}".format(epoch, accuracy, loss))

if __name__ == "__main__":
    main()