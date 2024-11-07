import os
import gc
import sys
import time
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
import torch.backends.cudnn as cudnn
import numpy as np
from util.utils import get_dataset, get_audio_model, get_language_model, get_backchannel_prediction_model, get_video_model
from util.criterions import get_criterion
from util.koalpaca import KoAlpaca
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from peft import LoraConfig, get_peft_model
import re
import pandas as pd


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
        self.is_MT = ("MT" in self.model)
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
        video_model = get_video_model("videomae")

        if self.dataset == "ETRI_2S" or self.dataset == "ETRI_2S_random" or self.dataset == "ETRI_TT_random" or self.dataset == "ETRI_ortega_random" or self.dataset == "ETRI_video" or self.dataset == "ETRI_TT_video":
            self.train_dataset, self.val_dataset, self.num_class, number_of_classes  = get_dataset(self.dataset, tokenizer)    
        else:     
            self.train_dataset, self.val_dataset, self.num_class = get_dataset(self.dataset, tokenizer)    
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)
        
        self.warmup2022_dataset = get_dataset("warmup2022", tokenizer)  
        self.warmup2023_dataset = get_dataset("warmup2023", tokenizer) 
        self.warmup_dataset = torch.utils.data.ConcatDataset([self.warmup2022_dataset, self.warmup2023_dataset])
        # self.warmup_dataset = get_dataset("youtube_warmup", tokenizer)  
          
        self.warmup_sampler = torch.utils.data.distributed.DistributedSampler(self.warmup_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.warmup_dataloader = torch.utils.data.DataLoader(self.warmup_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.warmup_sampler, num_workers=self.num_workers)
        
        if self.model == "BPM_MT" or self.model == "BPM_ST":
            self.model = get_backchannel_prediction_model(self.model)(
                language_model=language_model,
                audio_model=audio_model,
                output_size=128,
                num_class=self.num_class,
                sentiment_output_size=64,
                dropout=0.3,
                mode=self.mode)
        else:
            self.model = get_backchannel_prediction_model(self.model)(
                language_model=language_model,
                audio_model=audio_model,
                video_model=video_model,
                output_size=128,
                num_class=self.num_class,
                sentiment_output_size=64,
                dropout=0.3,
                mode=self.mode)
        self.model = self.model.to(self.local_rank)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # Get the model parameters divided into two groups : bert and others
        bert_params = []
        other_params = []
        fc_params = []

        self.pretext_epoch = 100
        self.model.train()
        
        for name, param in self.model.named_parameters():
            if 'language' in name or 'audio' in name or 'video' in name or 'marlin' in name:
                bert_params.append(param)
            elif "fc_layer" in name or "classifier" in name:
                fc_params.append(param)
            else:
                other_params.append(param)
                
        b_lr = 5e-6
        o_lr = 5e-4
        f_lr = 5e-5
        
        b_optimizer = torch.optim.Adam(bert_params, lr=b_lr, weight_decay=self.weight_decay)
        o_optimizer = torch.optim.Adam(other_params, lr=o_lr, weight_decay=self.weight_decay)
        f_optimizer = torch.optim.Adam(fc_params, lr=f_lr, weight_decay=self.weight_decay)
        
        print(f"b_lr : {b_lr}, o_lr : {o_lr}, f_lr : {f_lr} ")
        b_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=b_optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
        o_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=o_optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

        for epoch in range(0):
            self.model.train()
            start=time.time()
            
            train_acc = 0
            train_loss = 0
            ce_loss = 0
            cosine_loss = 0
            train_loss3 = 0
            count = 0

            for b, batch in enumerate(self.warmup_dataloader):
                for key in batch:
                    if key == "video_":
                        continue
                    batch[key] = batch[key].to(self.local_rank)

                y = self.model.forward(batch,warmup=True)
                
                contrastive_loss = y["InfoNCE"]
                cosine_loss += contrastive_loss.item() * len(batch["audio"])
                loss = contrastive_loss
                 
                train_loss += loss.item() * len(batch["audio"])
                count += len(batch["audio"])

                # Zero the gradients
                b_optimizer.zero_grad()
                o_optimizer.zero_grad()
                f_optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update the model parameters
                b_optimizer.step()
                o_optimizer.step()
                f_optimizer.step()
                
            loss     /= len(self.warmup_dataset)
            sec = time.time() - start
            times = str(datetime.timedelta(seconds=sec))
            short = times.split(".")[0]
                
            print(f"Epoch : {epoch}, Train Loss : {train_loss:.6f}, Loss : {loss.item():.6f}, Time taken : {short}")                
            gc.collect()
            
            b_scheduler.step() 
            o_scheduler.step()
            
            if self.rank == 0:
                model_save_path = f"/data/minjae/BC/Final/MAE/MOON/AGI-BC/save path/{epoch+1}_step_warmup_3_2.pt"
                torch.save(self.model.state_dict(), model_save_path)

        pt = '10step_warmup.pt'
        print(f"Pretrained Version : {pt}")
        model = torch.load(f'/data/minjae/BC/Final/MAE/MOON/AGI-BC/save path/{pt}', map_location=f'cuda:{self.local_rank}')

        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        model.to(f'cuda:{self.local_rank}')
        self.model = model 

        bert_params = []
        other_params = []
        fc_params = []

        self.pretext_epoch = 100
        self.model.train()
        
        for name, param in self.model.named_parameters():
            if 'language' in name or 'audio' in name or 'video' in name or 'marlin' in name:
                bert_params.append(param)
            elif "fc_layer" in name or "classifier" in name:
                fc_params.append(param)
            else:
                other_params.append(param)
                
        b_lr = 5e-6
        o_lr = 5e-4
        f_lr = 5e-5
        
        b_optimizer = torch.optim.Adam(bert_params, lr=b_lr, weight_decay=self.weight_decay)
        o_optimizer = torch.optim.Adam(other_params, lr=o_lr, weight_decay=self.weight_decay)
        f_optimizer = torch.optim.Adam(fc_params, lr=f_lr, weight_decay=self.weight_decay)
        
        print(f"b_lr : {b_lr}, o_lr : {o_lr}, f_lr : {f_lr} ")
        b_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=b_optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
        o_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=o_optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

        for epoch in range(self.epochs):
            self.train_sampler.set_epoch(epoch if not hasattr(self.model_without_ddp, 'pretext_forward') else epoch + self.pretext_epoch)
            self.model.train()
            start=time.time()
            
            train_acc = 0
            train_loss = 0
            ce_loss = 0
            cosine_loss = 0
            train_loss3 = 0
            count = 0

            for b, batch in enumerate(self.train_dataloader):
                for key in batch:
                    if key == "video_":
                        continue
                    batch[key] = batch[key].to(self.local_rank)

                y = self.model.forward(batch)
                bianry_label = (batch["label"] != 0).long()
                
                loss, logit = self.criteria(batch, y)
                loss = loss.mean()
                
                if self.is_MT:
                    loss = loss*0.9 + 0.1* F.cross_entropy(y['sentiment'], batch['sentiment'], reduction='mean')
                    
                contrastive_loss = y["InfoNCE"]
                
                accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()

                train_acc += accuracy.item() * len(batch["label"])
                ce_loss += loss.item() * len(batch["label"])
                cosine_loss += contrastive_loss.item() * len(batch["label"])
                
                loss = loss + contrastive_loss
                 
                train_loss += loss.item() * len(batch["label"])
                count += len(batch["label"])

                # Zero the gradients
                b_optimizer.zero_grad()
                o_optimizer.zero_grad()
                f_optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update the model parameters
                b_optimizer.step()
                o_optimizer.step()
                f_optimizer.step()           
                
                if self.verbose:    
                    print("Epoch : {}, {}/{},  Loss : {:.6f}, {:.6f}, {:.6f}, Acc : {:.3f}".format(epoch, b+1, len(self.train_dataloader), loss.item(), ce_loss, cosine_loss, accuracy.item()*100))#, end='\r')

                l, c = logit.argmax(dim=-1).unique(return_counts=True)
                gc.collect()
    
            b_scheduler.step() 
            o_scheduler.step()

            if self.rank == 0:
                model_save_path = f"/data/minjae/BC/Final/MAE/MOON/AGI-BC/save path/{epoch+1}_step.pt"
                torch.save(self.model.state_dict(), model_save_path)
            
            train_acc /= count
            train_loss /= count
            self.model.eval()
            with torch.no_grad():

                accuracy = 0
                loss     = 0
                tp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                tn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)

                for b, batch in enumerate(self.val_dataloader):
                    for key in batch:
                        if key == "video_":
                            continue
                        batch[key] = batch[key].to(self.local_rank)

                    y = self.model(batch)
                    binary_label = (batch["label"] != 0).long()
                    
                    loss_t, logit = self.criteria(batch, y)
                    loss_t = loss_t.mean()

                    # Calculate the accuracy
                    accuracy += (torch.argmax(logit, dim=1) == batch["label"]).float().sum()
                    loss     += loss_t * len(batch["label"])
                    
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
                if self.dataset == "ETRI_2S" or self.dataset == "ETRI_2S_random" or self.dataset == "ETRI_TT_random" or self.dataset == "ETRI_ortega_random" or self.dataset == "ETRI_video" or self.dataset == "ETRI_TT_video":
                    number_of_classes = number_of_classes
                else:    
                    number_of_classes = self.val_dataset.get_sample_in_class()
                weighted_f1_score = (f1_score * number_of_classes).sum() / number_of_classes.sum()
                
                sec = time.time() - start
                times = str(datetime.timedelta(seconds=sec))
                short = times.split(".")[0]
                
                print(f"Epoch : {epoch}, Train Loss : {train_loss:.6f}, Train Acc : {train_acc*100:.3f}, Time taken : {short}")
                print(f"Loss : {loss.item():.6f}, Acc : {accuracy.item()*100:.3f}, F1 : {weighted_f1_score.item()*100:.2f}, {f1_score.tolist()}")
                print(f"Train Loss : {train_loss:.6f}, CE Loss : {ce_loss:.6f}, Cosine Distance : {cosine_loss:.6f}")          
                
                gc.collect()
        
    def init_distributed(self):
        if self.distributed:
            if torch.cuda.is_available():
                self.gpu    = self.local_rank % self.ngpus_per_node
                self.device = torch.device(self.gpu)
                if self.distributed:
                    self.local_rank = self.gpu
                #    self.rank = self.node_rank * self.ngpus_per_nodes + self.gpu
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
    
    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=item.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=item.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        all_qs = torch.cat(all_qs, dim=0)
        return all_qs
    
    def get_target_modules(self, model):
        pattern = r'\((\w+)\): Linear'
        linear_layers = re.findall(pattern, str(model.modules))
        target_modules = list(set(linear_layers))        
        return target_modules
