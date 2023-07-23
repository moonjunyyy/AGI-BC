import os
import gc
import time
import torch
import torch.nn as nn
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

from transformers import BertModel, AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Subset
from ETRI_Dataset import ETRI_Corpus_Dataset
from SWBD_Dataset import SWBD_Dataset
from HuBert import HuBert
from Audio_LSTM import Audio_LSTM

from transformers import AutoTokenizer, AutoModelForPreTraining
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Trainer:
    def __init__(self, args) -> None:
        print(args)

        self.mode = args.mode
        self.loss_functions = {
            "focal": self._focal_loss,
            "counting": self._counting_loss,
            "hierarchical": self._hierarchical_loss,
            "no_one_left_behind": self._no_one_left_behind,
            "mean_pooling": self._cross_entropy_loss,
            "audio_only" : self._cross_entropy_loss,
            "text_only" : self._cross_entropy_loss,
            "flatten" : self._cross_entropy_loss,
        }
        self.criteria = self.loss_functions[self.mode]

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

        # select from dataset label 2 and 3
        # NoBC = []
        # Continuer = []
        # Understanding = []
        # Empathic = []
        # for i, b in enumerate(dataset):
        #     if b['label'] == 0:
        #         NoBC.append(i)
        #     elif b['label'] == 1:
        #         Continuer.append(i)
        #     elif b['label'] == 2:
        #         Understanding.append(i)
        #     elif b['label'] == 3:
        #         Empathic.append(i)

        # Understanding = random.sample(Understanding, len(Empathic))
        # subset = Understanding + Empathic
        # random.shuffle(subset)
        # dataset = Subset(dataset, subset)

        # self.num_class = 2

        self.train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))
        self.val_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)


        if self.is_MT:
            self.model = BPM_MT(language_model=bert, audio_model=audio_model, sentiment_dict=sentiment_dict, num_class=self.num_class, mode=self.mode)
        else:
            self.model = BPM_MT(language_model=bert, audio_model=audio_model, sentiment_dict=None, num_class=self.num_class, mode=self.mode)
        
        self.model = self.model.to(self.local_rank)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # Get the model parameters divided into two groups : bert and others
        bert_params = []
        other_params = []
        fc_params = []
        discriminator_params = []

        for name, param in self.model.named_parameters():
            if 'language_model' in name or 'audio_model' in name:
                bert_params.append(param)
            elif 'fc_layer' in name or 'prompt' in name or 'classifier' in name:
                fc_params.append(param)
            elif 'discriminator' in name:
                discriminator_params.append(param)
            else:
                other_params.append(param)

        adam_optimizer = torch.optim.Adam(other_params, lr=0.0005, weight_decay=5e-4)
        fc_optimizer = torch.optim.Adam(fc_params, lr=0.0005, weight_decay=5e-4)
        sgd_optimizer = torch.optim.SGD(bert_params, lr=0.0005)

        # for epoch in range(self.epochs):
        #     for b, batch in enumerate(self.train_dataloader):
        #         # Move the batch to GPU if CUDA is available
        #         for key in batch:
        #             batch[key] = batch[key].to(self.local_rank)

        #         y = self.model.text_forward(batch)
                
        #         loss, logit = self.criteria(batch, y)
        #         loss = loss.mean()
            
        #         accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()

        #         # Zero the gradients
        #         adam_optimizer.zero_grad()
        #         sgd_optimizer.zero_grad()
        #         fc_optimizer.zero_grad()
        #         discriminator_optimizer.zero_grad()

        #         # Backpropagation
        #         loss = loss# + 0.5 * y['modalities']
        #         loss.backward()

        #         adam_optimizer.step()
        #         # sgd_optimizer.step()
        #         fc_optimizer.step()

        #         print("Epoch : {}, {}/{},  Loss : {:.6f}, Acc : {:.3f},".format(epoch, b+1, len(self.train_dataloader), loss.item(), accuracy.item()*100), end='\r')
        #         # print("Epoch : {}, {}/{},  Loss : {:.6f}, Acc : {:.3f},".format(epoch, b+1, len(self.train_dataloader), loss.item(), accuracy.item()*100), end=' ')
        #         l, c = logit.argmax(dim=-1).unique(return_counts=True)
        #         # for i in range(len(l)):
        #         #     print(l[i].item(), ':', c[i].item(), end=' ')
        #         # print()
        #         gc.collect()
        #     print()
        # self.model.text_prompt_keys.requires_grad_(False)
        # self.model.text_prompt_values.requires_grad_(False)

        pretext_loss = 0
        pretext_count = 0

        # for epoch in range(self.epochs):
        #     for b, batch in enumerate(self.train_dataloader):
        #         for key in batch:
        #             batch[key] = batch[key].to(self.local_rank)
                
        #         y = self.model.pretext_forward(batch)
                
        #         adam_optimizer.zero_grad()
        #         sgd_optimizer.zero_grad()
        #         fc_optimizer.zero_grad()

        #         y.backward()

        #         adam_optimizer.step()
        #         # sgd_optimizer.step()
        #         fc_optimizer.step()

        #         pretext_loss += y.item() * len(batch['label'])
        #         pretext_count += len(batch['label'])

        #         print("Epoch : {}, {}/{},  Loss : {:.6f},".format(epoch, b+1, len(self.train_dataloader), y.item()), end='\r')
        #         gc.collect()
        #     print(f"Epoch : {epoch}, Pretext loss : {pretext_loss / pretext_count}")

        for epoch in range(self.epochs):
            for b, batch in enumerate(self.train_dataloader):
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].to(self.local_rank)
                # batch['label'] = batch['label'] - 2

                y = self.model.forward(batch)
                loss, logit = self.criteria(batch, y)
                loss = loss.mean()
            
                accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()

                # Zero the gradients
                adam_optimizer.zero_grad()
                sgd_optimizer.zero_grad()
                fc_optimizer.zero_grad()

                # Backpropagation
                loss = loss #+ 0.5 * y['modalities']
                loss.backward()

                # Update the model parameters
                adam_optimizer.step()
                sgd_optimizer.step()
                fc_optimizer.step()

                print("Epoch : {}, {}/{},  Loss : {:.6f}, Acc : {:.3f},".format(epoch, b+1, len(self.train_dataloader), loss.item(), accuracy.item()*100), end='\r')
                # print("Epoch : {}, {}/{},  Loss : {:.6f}, Acc : {:.3f},".format(epoch, b+1, len(self.train_dataloader), loss.item(), accuracy.item()*100), end=' ')
                l, c = logit.argmax(dim=-1).unique(return_counts=True)
                # for i in range(len(l)):
                #     print(l[i].item(), ':', c[i].item(), end=' ')
                # print()
                gc.collect()

            with torch.no_grad():
                # Validation loop
                accuracy = 0
                loss     = 0
                
                tp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                tn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)

                for batch in self.val_dataloader:
                    # Move the batch to GPU if CUDA is available
                    for key in batch:
                        batch[key] = batch[key].to(self.local_rank)
                    # batch['label'] = batch['label'] - 2
                    
                    y = self.model(batch)

                    loss_t, logit = self.criteria(batch, y)
                    loss_t = loss_t.mean()

                    # Calculate the accuracy
                    accuracy += (torch.argmax(logit, dim=1) == batch["label"]).float().sum()
                    loss     += loss_t * len(batch["label"])

                    # Calculate the confusion matrix
                    for i in range(len(batch["label"])):
                        for l in range(self.num_class):
                            if batch["label"][i] == l:
                                if logit.argmax(dim=-1)[i] == l:
                                    tp[l] += 1
                                else:
                                    fn[l] += 1
                            else:
                                if logit.argmax(dim=-1)[i] == l:
                                    fp[l] += 1
                                else:
                                    tn[l] += 1

                if self.distributed:
                    accuracy = accuracy.to(self.local_rank)
                    loss     = loss.to(self.local_rank)
                    tp = tp.to(self.local_rank)
                    fp = fp.to(self.local_rank)
                    fn = fn.to(self.local_rank)
                    tn = tn.to(self.local_rank)

                    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
                    dist.all_reduce(fp, op=dist.ReduceOp.SUM)
                    dist.all_reduce(fn, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tn, op=dist.ReduceOp.SUM)

                    accuracy = accuracy.cpu()
                    loss     = loss.cpu()
                    tp = tp.cpu()
                    fp = fp.cpu()
                    fn = fn.cpu()
                    tn = tn.cpu()

                accuracy /= len(self.val_dataset)
                loss     /= len(self.val_dataset)
                precision = tp / (tp + fp)
                recall    = tp / (tp + fn)
                f1_score  = 2 * precision * recall / (precision + recall)
                print("Epoch : {}, Accuracy : {}, Loss : {}, F1 score : {}".format(epoch, accuracy.item(), loss.item(), f1_score.cpu().tolist()))
            gc.collect()
        
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
    
    def _focal_loss(self, batch, output):
        gamma = 2
        softmax = F.softmax(output['logit'], dim=-1)
        loss = softmax[torch.arange(len(batch['label'])), batch['label']]
        loss = - (1 - loss) ** gamma * (loss + 1e-6).log()
        return loss, output['logit']

    def _counting_loss(self, batch, output):
        unique, count = batch['label'].unique(return_counts=True)
        count = count[(unique == batch['label'].unsqueeze(1)).nonzero()[:,1]]
        loss = F.cross_entropy(output['logit'], batch["label"], reduction='none')        
        loss = loss / count * len(batch['label'])
        return loss.mean(), output['logit']

    def _hierarchical_loss(self, batch, output):
        logit = output['logit']
        BC_logit = output['logit_BC']
        
        bc_loss = F.cross_entropy(BC_logit, (batch['label'] > 0).long(), reduction='none')
        logit = logit[batch['label'] > 0]
        loss = F.cross_entropy(logit, batch["label"][batch['label'] > 0] - 1, reduction='none')
        loss = loss.mean() + bc_loss.mean()

        softmax = F.softmax(output["logit"], dim=-1)
        BC_softmax = F.softmax(output["logit_BC"], dim=-1)
        logit = torch.cat((
            BC_softmax[:,:1], BC_softmax[:,1:] * softmax
        ),dim=1)
        return loss, logit
    
    def _no_one_left_behind(self, batch, output):
        unique, count = batch['label'].unique(return_counts=True)
        logit = output['logit']
        logit = logit - logit.max(dim=-1, keepdim=True)[0]
        logit = logit.exp()
        
        cnt = torch.zeros(logit.shape[0], device=self.local_rank)
        for c in range(self.num_class):
            if c in unique:
                cnt[batch['label'] == c] = count[unique == c].item()
            else:
                cnt[batch['label'] == c] = 0
        logit = logit * cnt.unsqueeze(-1)
        p = logit[torch.arange(len(batch['label'])), batch['label']]
        logit = p / logit.sum(dim=-1)
        logit = logit.squeeze()

        loss = 0
        for c, u in enumerate(unique):
            loss = loss - torch.log(logit[batch["label"] == u] + 1e-6).sum() / count[c]
        loss = loss / u
        return loss, output['logit']
    
    def _cross_entropy_loss(self, batch, output):
        loss_BC = F.cross_entropy(output['logit'], batch["label"], reduction='none')
        if self.is_MT:
            loss_SP = F.binary_cross_entropy(torch.sigmoid(output['sentiment']), batch["sentiment"])
            loss = 0.9 * loss_BC + 0.1 * loss_SP
        else:
            loss = loss_BC
        return loss.mean(), output['logit']