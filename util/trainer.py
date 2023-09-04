import os
import gc
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn.functional as F
from util.utils import get_dataset, get_audio_model,\
     get_language_model, get_backchannel_prediction_model
from util.criterions import get_criterion
from util.koalpaca import KoAlpaca

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Trainer:
    def __init__(self, args) -> None:
        print(args)

        self.path = args.path
        self.mode = args.mode
        self.criteria = get_criterion(self.mode)

        self.verbose = args.verbose
        self.model = args.model
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.world_size = args.world_size
        self.is_MT = args.is_MT
        self.language = args.language
        self.audio = args.audio
        self.weight_decay = args.weight_decay

        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = os.environ.get("MASTER_PORT", "8888")
        self.dist_url = f"{args.dist_url}{self.master_addr}:{self.master_port}"

        self.seed = args.seed
        self.distributed = False
        self.rank = args.rank
        self.ngpus_per_node = torch.cuda.device_count()
        self.dist_backend = args.dist_backend
        self.world_size = args.world_size * self.ngpus_per_node
        self.distributed = self.world_size > 1

        if os.path.exists(self.path) is False:
            os.makedirs(self.path)
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
        
        tokenizer, language_model = get_language_model(self.language)
        audio_model = get_audio_model(self.audio)

        self.train_dataset, self.val_dataset, self.num_class = get_dataset(self.dataset, tokenizer)    
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)
        
        self.model = get_backchannel_prediction_model(self.model)(
            language_model=language_model,
            audio_model=audio_model,
            output_size=128,
            num_class=self.num_class,
            sentiment_output_size=64,
            dropout=0.3,
            mode=self.mode,
            tokenizer=tokenizer,
            path=self.path)
        self.model = self.model.to(self.local_rank)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # Get the model parameters divided into two groups : bert and others
        bert_params = []
        other_params = []
        discriminator_params = []
        prompt_params = []

        # for name, param in self.model.named_parameters():
        #     if 'language_model' in name or 'audio_model' in name:
        #         bert_params.append(param)
        #     elif 'prompt' in name:
        #         prompt_params.append(param)
        #     else:
        #         other_params.append(param)

        # adam_optimizer = torch.optim.Adam(other_params, lr=1e-4, weight_decay=self.weight_decay)
        # sgd_optimizer = torch.optim.SGD(bert_params, lr=1e-4)
        # if prompt_params != []:
        #     adam_optimizer.add_param_group({'params': prompt_params, 'lr': 0.0001, 'weight_decay': self.weight_decay})
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5, weight_decay=self.weight_decay)
        self.pretext_epoch = 100
        self.model.train()
        try:
            state_dict = torch.load(f'{self.path}/pretrained.pt')
            self.model_without_ddp.load_state_dict(state_dict)
            print('Load pretrained model')
        except Exception as e:
            tiktok = False
            print(e)
            print('No pretrained model')
            if hasattr(self.model_without_ddp, 'pretext_forward'):
                # koalpaca = KoAlpaca()
                for epoch in range(self.pretext_epoch):
                    self.train_sampler.set_epoch(epoch)
                    pre_loss = 0
                    pre_count = 0
                    for b, batch in enumerate(self.train_dataloader):
                        pre_count += 1
                        tiktok = not tiktok
                        if tiktok:
                            optimizer.param_groups[0]['lr'] = 1e-4
                            optimizer.param_groups[0]['weight_decay'] = 0.001
                        else:
                            optimizer.param_groups[0]['lr'] = 1e-4
                            optimizer.param_groups[0]['weight_decay'] = self.weight_decay
                        for key in batch:
                            batch[key] = batch[key].to(self.local_rank)
                        # print(f"Epoch : {epoch}, {b+1}/{len(self.train_dataloader)}", end=' ')
                        loss = self.model_without_ddp.pretext_forward(batch)
                        loss = loss.mean()
                        pre_loss += loss.item()
                        # Zero the gradients
                        optimizer.zero_grad()
                        # Backpropagation
                        loss.backward()
                        # Update the model parameters
                        optimizer.step()
                        # if self.verbose:
                            # print("Epoch : {}, {}/{},  Loss : {:.6f}".format(epoch, b+1, len(self.train_dataloader), loss.item()), end='\r')
                            # ta, fa, tt, ft = self.model_without_ddp.get_generation_result(batch, tokenizer)
                    print(f'Epoch : {epoch}, Loss : {pre_loss/pre_count:.6f}', " "*20)
                    torch.save(self.model_without_ddp.state_dict(), f'{self.path}/pretrained.pt')
                    sys.stdout.flush()
            else:
                print('No pretext training')

        for epoch in range(self.epochs):
            self.train_sampler.set_epoch(epoch if not hasattr(self.model_without_ddp, 'pretext_forward') else epoch + self.pretext_epoch)
            self.model.train()
            for b, batch in enumerate(self.train_dataloader):
                
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].to(self.local_rank)

                y = self.model.forward(batch)
                loss, logit = self.criteria(batch, y)
                loss = loss.mean()

                if self.model == 'BPM_MT':
                    loss += F.cross_entropy(y['sentiment'], batch['sentiment'], reduction='mean')

                accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()

                # Zero the gradients
                optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update the model parameters
                optimizer.step()

                if self.verbose:
                    print("Epoch : {}, {}/{},  Loss : {:.6f}, Acc : {:.3f},".format(epoch, b+1, len(self.train_dataloader), loss.item(), accuracy.item()*100), end='\r')
                l, c = logit.argmax(dim=-1).unique(return_counts=True)
                gc.collect()

            self.model.eval()
            with torch.no_grad():

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
                    loss = loss.to(self.local_rank)
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
                precision = tp / (tp + fp + 1e-6)
                recall    = tp / (tp + fn + 1e-6)
                f1_score  = 2 * precision * recall / (precision + recall + 1e-6)
                f1_score = f1_score.nan_to_num(0).detach().cpu()
                number_of_classes = self.val_dataset.get_sample_in_class()
                weighted_f1_score = (f1_score * number_of_classes).sum() / number_of_classes.sum()
                print(f"Epoch : {epoch}, Loss : {loss.item():.6f}, Acc : {accuracy.item()*100:.3f}, F1 : {weighted_f1_score.item()*100:.3f}, {f1_score.tolist()}")
                sys.stdout.flush()
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