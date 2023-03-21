import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import numpy as np
from BPM_MT import BPM_MT, BPM_ST
import json
import torch
import torch.nn.functional as F

from transformers import BertModel
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchaudio.transforms import MFCC
from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
from ETRI_Dataset import ETRI_Corpus_Dataset

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


        self.seed = args.seed
        self.distributed = False
        self.rank = args.rank
        self.ngpus_per_node = torch.cuda.device_count()
        self.world_size = args.world_size * self.ngpus_per_node
        self.distributed = self.world_size > 1

        self.is_MT = args.is_MT

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

        self.init_distributed()
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
                
        
        mfcc_extractor = MFCC(sample_rate=16000, n_mfcc=13)
        if self.language == 'ko':
            tokenizer = SentencepieceTokenizer(get_tokenizer())
            bert, vocab = get_pytorch_kobert_model()
            sentiment_dict = json.load(open('data/SentiWord_info.json', encoding='utf-8-sig', mode='r'))
        elif self.language == 'en':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, output_hidden_states=True, output_attentions=True)
            vocab = None
            sentiment_dict = {}
            with open('data/subjclueslen1-HLTEMNLP05.tff', 'r', encoding='utf-8') as f:
                # sentiment_dict = { re.split(" =\n", line) for line in f.readlines()}
                for line in f.readlines():
                    line = re.split("=| |\n", line)
                    if line[11] == 'neutral':
                        sentiment_dict[line[5]] = 0
                    elif line[11] == 'positive':
                        if line[0] == 'strongsubj':
                            sentiment_dict[line[5]] = 2
                        else:
                            sentiment_dict[line[5]] = 1
                    elif line[11] == 'negative':
                        if line[0] == 'strongsubj':
                            sentiment_dict[line[5]] = -2
                        else:
                            sentiment_dict[line[5]] = -1
        else:
            raise NotImplementedError

        tf = transforms.ToTensor()
        dataset = ETRI_Corpus_Dataset(path = '/local_datasets', tokenizer=tokenizer, vocab=vocab, transform=tf, length=1.5)
        self.train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))
        self.val_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)

        if self.is_MT:
            self.model = BPM_MT(tokenizer=tokenizer, bert=bert, vocab=vocab, sentiment_dict=sentiment_dict, mfcc_extractor=mfcc_extractor)
        else:
            self.model = BPM_ST(tokenizer=tokenizer, bert=bert, vocab=vocab, mfcc_extractor=mfcc_extractor)
        
        self.model = self.model.to(self.local_rank)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # Get the model parameters divided into two groups : bert and others
        bert_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        adam_optimizer = torch.optim.Adam(bert_params, lr=0.0001)
        sgd_optimizer = torch.optim.SGD(other_params, lr=0.0001)

        # Training loop
        for epoch in range(self.epochs):
            for batch in self.train_dataloader:
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].cuda()
                y = self.model(batch)

                # Get the logit from the model
                logit     = y["logit"]
                sentiment = y["sentiment"]

                # Calculate the loss
                loss_BC = F.cross_entropy(logit, batch["label"])
                if self.is_MT:
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
            for batch in self.val_dataloader:
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].cuda()
                y = self.model(batch)

                # Get the logit from the model
                logit     = y["logit"]
                sentiment = y["sentiment"]

                # Calculate the loss
                loss_BC = F.cross_entropy(logit, batch["label"])
                if self.is_MT:
                    loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), batch["sentiment"])
                    loss = 0.9 * loss_BC +  0.1 * loss_SP
                else:
                    loss = loss_BC

                # Calculate the accuracy
                accuracy += (torch.argmax(logit, dim=1) == batch["label"]).sum().item()
                loss    += loss.item() * len(batch["label"])
                if self.distributed:
                    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            accuracy /= len(self.val_dataset)
            loss     /= len(self.val_dataset)
            print("Epoch : {}, Accuracy : {}, Loss : {}".format(epoch, accuracy, loss))
        

    def init_distributed(self):
        if self.distributed:
            if torch.cuda.is_available():
                self.gpu    = self.local_rank % self.ngpus_per_nodes
                self.device = torch.device(self.gpu)
                if self.distributed:
                    self.local_rank = self.gpu
                    self.rank = self.node_rank * self.ngpus_per_nodes + self.gpu
                    time.sleep(self.rank * 0.1) # prevent port collision
                    print(f'rank {self.rank} is running...')
                    dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                            world_size=self.world_size, rank=self.rank)
                    dist.barrier()
                    self.setup_for_distributed(self.is_main_process())
        else:
            self.device = torch.device('cpu')


    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def get_rank(self):
        if self.distributed:
            return dist.get_rank()
        return 0
    
    def get_world_size(self):
        if self.distributed:
            return dist.get_world_size()
        return 1
    