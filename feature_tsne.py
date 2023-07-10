import os
import gc
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import numpy as np
from BPM_MT import BPM_MT
import json
import torch
import torch.nn.functional as F
import argparse
import torch

from transformers import BertModel, AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Subset
from ETRI_Dataset import ETRI_Corpus_Dataset
from SWBD_Dataset import SWBD_Dataset
from HuBert import HuBert
from Audio_LSTM import Audio_LSTM

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForPreTraining

class Trainer:
    def __init__(self, args) -> None:
        print(args)

        self.model = args.model
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.world_size = args.world_size
        self.is_MT = args.is_MT
        self.language = args.language
        self.audio = args.audio

        self.ngpus_per_nodes = torch.cuda.device_count()
        self.node_rank = args.rank
        self.dist_backend = args.dist_backend

        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = os.environ.get("MASTER_PORT", "8888")
        self.dist_url = f"{args.dist_url}{self.master_addr}:{self.master_port}"

        self.seed = args.seed
        self.distributed = False
        self.rank = args.rank
        self.ngpus_per_node = torch.cuda.device_count()
        self.world_size = args.world_size * self.ngpus_per_node
        self.distributed = self.world_size > 1

        self.batch_size = int(self.batch_size / self.world_size)

        print("is_MT: ", self.is_MT)

        if os.environ.get("MASTER_ADDR") is None:
            os.environ["MASTER_ADDR"] = "localhost"
        if os.environ.get("MASTER_PORT") is None:
            os.environ["MASTER_PORT"] = "8888"

    def run(self):
        if self.distributed:
            mp.spawn(self._run, nprocs=self.world_size, args=(self.world_size,))
        else:
            self._run(0, 1)

    def _run(self, rank, world_size):

        self.local_rank = rank
        self.rank = self.rank * self.ngpus_per_node + rank
        self.world_size = world_size

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed) # if use multi-GPU
            cudnn.deterministic = True
            cudnn.benchmark = False
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
                
        
        if self.language == 'koBert':
            tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
            bert = BertModel.from_pretrained("skt/kobert-base-v1", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
            sentiment_dict = json.load(open('data/SentiWord_info.json', encoding='utf-8-sig', mode='r'))
            self.num_class = 4
        elif self.language == 'Bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
            sentiment_dict = {}
            self.num_class = 2
        elif self.language == 'ELECTRA':
            tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
            bert = AutoModelForPreTraining.from_pretrained("google/electra-base-discriminator")
            sentiment_dict = {}
            self.num_class = 2
        else:
            raise NotImplementedError
        
        if self.audio == 'LSTM':
            audio_model = Audio_LSTM()
        elif self.audio == 'HuBert':
            audio_model = HuBert(sample_rate=16000)

        tf = transforms.ToTensor()
        audio_model = audio_model.to(self.local_rank)
        bert = bert.to(self.local_rank)

        if self.language == 'koBert':
            dataset = ETRI_Corpus_Dataset(path = '/local_datasets', tokenizer=tokenizer, transform=tf, length=1.5)
        else :
            dataset = SWBD_Dataset(path = '/local_datasets', tokenizer=tokenizer, length=1.5)

        # for name, module in audio_model.named_modules():
        #     print(name)

        # for name, module in bert.named_modules():
        #     print(name)
        self.num_class = 2

        self.train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))
        self.val_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)


        # if self.is_MT:
        #     self.model = BPM_MT(language_model=bert, audio_model=audio_model, sentiment_dict=sentiment_dict, num_class=self.num_class, mode=self.mode)
        # else:
        #     self.model = BPM_MT(language_model=bert, audio_model=audio_model, sentiment_dict=None, num_class=self.num_class, mode=self.mode)
        
        # self.model = self.model.to(self.local_rank)
        # self.model_without_ddp = self.model
        # if self.distributed:
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # Get the model parameters divided into two groups : bert and others
        # bert_params = []
        # other_params = []
        # fc_params = []

        # for name, param in self.model.named_parameters():
        #     if 'language_model' in name or 'audio_model' in name:
        #         bert_params.append(param)
        #     elif 'fc_layer' in name or 'prompt' in name or 'classifier' in name:
        #         fc_params.append(param)
        #     else:
        #         other_params.append(param)

        # adam_optimizer = torch.optim.Adam(other_params, lr=0.0005, weight_decay=0.01)
        # fc_optimizer = torch.optim.Adam(fc_params, lr=0.0005, weight_decay=0.01)
        # sgd_optimizer = torch.optim.SGD(bert_params, lr=0.0005)

        # for epoch in range(self.epochs//2):
        #     for b, batch in enumerate(self.train_dataloader):
        #          # Move the batch to GPU if CUDA is available
        #         for key in batch:
        #             batch[key] = batch[key].to(self.local_rank)

        #         loss = self.model.pretext_forward(batch)

        #         # Zero the gradients
        #         adam_optimizer.zero_grad()
        #         sgd_optimizer.zero_grad()
        #         fc_optimizer.zero_grad()

        #         # Backpropagation
        #         loss.backward()

        #         # Update the model parameters
        #         adam_optimizer.step()
        #         sgd_optimizer.step()
        #         fc_optimizer.step()

        #         print("Epoch : {}, {}/{},  Loss : {:.6f},".format(epoch, b+1, len(self.train_dataloader), loss.item()), end=' ')
        #         gc.collect()

        audio_feature = []
        text_feature = []
        labels = []
        for b, batch in enumerate(self.train_dataloader):
            # Move the batch to GPU if CUDA is available
            for key in batch:
                batch[key] = batch[key].to(self.local_rank)
            audio = batch['audio']
            text = batch['text']
            label = batch['label']

            
            audio = audio_model(audio)
            text = bert(text).last_hidden_state

            audio = audio.mean(1)
            text = text.mean(1)

            audio_feature.append(audio.detach().cpu().numpy())
            text_feature.append(text.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            print(b, end=' ')
            if b == 5000:
                break
        print()
        audio_feature = np.concatenate(audio_feature, axis=0)
        text_feature = np.concatenate(text_feature, axis=0)
        labels =np.concatenate(labels, axis=0)
        print(audio_feature.shape)
        print(text_feature.shape)
        print(labels.shape)

        tsne = TSNE(n_components=2, random_state=0)
        audio_tsne = tsne.fit_transform(audio_feature)
        for l in range(4):
            plt.scatter(audio_tsne[labels==l, 0], audio_tsne[labels==l, 1], s=1)
        plt.savefig('audio_tsne.png')
        plt.clf()
        
        tsne = TSNE(n_components=2, random_state=0)
        text_tsne = tsne.fit_transform(text_feature)
        for l in range(4):
            plt.scatter(text_tsne[labels==l, 0], text_tsne[labels==l, 1], s=1)
        plt.savefig('text_tsne.png')
        plt.clf()

        tsne = TSNE(n_components=2, random_state=0)
        combined_tsne = tsne.fit_transform(np.concatenate([audio_feature, text_feature], axis=0))
        for l in range(4):
            plt.scatter(combined_tsne[:audio_feature.shape[0]][labels==l, 0], combined_tsne[:audio_feature.shape[0]][labels==l, 1], s=1)
            plt.scatter(combined_tsne[audio_feature.shape[0]:][labels==l, 0], combined_tsne[audio_feature.shape[0]:][labels==l, 1], s=1)
        plt.savefig('combined_tsne.png')
        plt.clf()

        tsne = TSNE(n_components=2, random_state=0)
        concated_tsne = tsne.fit_transform(np.concatenate([audio_feature, text_feature], axis=1))
        for l in range(4):
            plt.scatter(concated_tsne[labels==l, 0], concated_tsne[labels==l, 1], s=1)
        plt.savefig('concated_tsne.png')
        plt.clf()

is_MT = True    # True for MT, False for ST
use_CUDA = True # True for GPU, False for CPU
batch_size = 64 # Batch size
epochs = 60     # Number of epochs
torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BPM_MT' if is_MT else 'BPM_ST')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--is_MT', action='store_true', default=False)
    parser.add_argument('--language', type=str, default='koBert')
    parser.add_argument('--audio', type=str, default='LSTM')
    parser.add_argument('--use_CUDA', type=bool, default=use_CUDA)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='tcp://')
    parser.add_argument('--mode', type=str, default='cross_entropy')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()
    pass

if __name__ == "__main__":
    main()