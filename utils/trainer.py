import os
import gc
import sys
import time
import random
import torch
import torch.amp
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import get_dataset, get_audio_model,\
     get_language_model, get_backchannel_prediction_model, get_video_model
from utils.criterions import get_criterion
from m00nny_utils.warmup_cosine_anneling import WarmUpCosineAnnelingScheduler
from m00nny_utils.parameter_hook import ParameterHook
from m00nny_utils.sharded_modules import all_gather, all_reduce
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Trainer:
    def __init__(self, args) -> None:
        print(args)

        self.path = args.path
        self.mode = args.mode
        self.criteria = get_criterion(self.mode)
        self.data_path = args.data_path

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
        self.video = args.video
        self.weight_decay = args.weight_decay

        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = os.environ.get("MASTER_PORT", "8888")
        self.dist_url = f"{args.dist_url}{self.master_addr}:{self.master_port}"

        self.seed = args.seed
        self.distributed = False
        self.rank = args.rank
        self.ngpus_per_node = torch.cuda.device_count()
        self.dist_backend = args.dist_backend
        self.node_rank = args.rank
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
        print(f"world_size: {self.world_size}, rank: {self.rank}, ngpus_per_node: {self.ngpus_per_node}, distributed: {self.distributed}")

    def run(self):
        if self.distributed:
            mp.spawn(self._run, nprocs=self.world_size, args=(self.world_size,))
        else:
            self._run(0, 1)

    def _run(self, rank, world_size):
        self.local_rank = rank
        self.rank = self.rank * self.ngpus_per_node + rank
        self.world_size = world_size
        torch.cuda.set_device(self.local_rank)

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

        if self.language is not None:
            tokenizer, language_model = get_language_model(self.language)
        if self.audio is not None:
            audio_model = get_audio_model(self.audio)
        if self.video is not None:
            video_model = get_video_model(self.video)

        self.train_dataset, self.val_dataset, self.num_class = get_dataset(self.dataset, self.data_path, tokenizer)
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True,  num_replicas=self.world_size, rank=self.rank, drop_last=True)
        self.val_sampler =   DistributedSampler(self.val_dataset,   shuffle=False, num_replicas=self.world_size, rank=self.rank, drop_last=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader =   DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler,   num_workers=self.num_workers)

        if self.model == 'BPM_MT':
            self.is_MT = True
        else:
            self.is_MT = False

        self.model = get_backchannel_prediction_model(self.model)(
            language_model=language_model,
            audio_model=audio_model,
            video_model=video_model,
            output_size=128,
            num_class=self.num_class,
            sentiment_output_size=64,
            dropout=0.3,
            mode=self.mode)

        bert_params = []
        other_params = []
        prompt_params = []

        self.model = self.model.cuda()
        self.model_without_ddp = self.model
        if self.distributed:
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
            self.hooker = ParameterHook(self.model)

        self.pretext_epoch = 10
        self.model.train()
        try:
            state_dict = torch.load(f'{self.path}/pretrained.pt', map_location=f'cpu', weights_only=True)
            self.model_without_ddp.load_state_dict(state_dict, strict=False)
            self.model_without_ddp.to('cuda')
            print('Load pretrained model')
        except Exception as e:
            print(e)
            print('No pretrained model')
            if hasattr(self.model_without_ddp, 'pretext_task'):
                self.model_without_ddp.pretext_task(self.train_dataloader)
                self.train_sampler.epoch += 1
                self.train_sampler.set_epoch(self.train_sampler.epoch)
                if dist.get_rank() == 0:
                    torch.save(self.model_without_ddp.state_dict(), f'{self.path}/pretrained.pt')
                sys.stdout.flush()
            else:
                print('No pretext training')
        # import matplotlib.pyplot as plt
        bert_params = []
        other_params = []
        prompt_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'language_model' in name or 'audio_model' in name or 'video_model' in name:
                bert_params.append(param)
            elif 'prompt' in name:
                prompt_params.append(param)
            elif 'fc_layer' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                other_params.append(param)
        whole_params = bert_params + other_params + prompt_params + classifier_params        
        
        # optimizer = torch.optim.AdamW(bert_params, lr=5e-6, weight_decay=self.weight_decay)
        optimizer = torch.optim.AdamW(bert_params, lr=5e-6)
        optimizer.add_param_group({'params': other_params, 'lr': 5e-5})
        optimizer.add_param_group({'params': classifier_params, 'lr': 5e-4})
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        scheduler = WarmUpCosineAnnelingScheduler(optimizer, warmup_steps=5, t_total=20)
        scheduler.step() # the first step is zero
        optimizer.param_groups[2]['lr'] = 5e-4
        scaler = torch.amp.GradScaler('cuda')
        
        for epoch in range(self.epochs):
            self.model.train()
            # self.train_sampler.set_epoch(epoch)
            if hasattr(self.model_without_ddp, 'pre_epoch'):
                self.model_without_ddp.pre_epoch(self.train_dataloader)
                self.train_sampler.epoch += 1
                self.train_sampler.set_epoch(self.train_sampler.epoch)

            self.model.train()
            print(f"Epoch : {epoch}, Learning Rate : {[param_group['lr'] for param_group in optimizer.param_groups]}")

            train_acc = 0
            train_loss = 0
            count = 0

            for b, batch in enumerate(self.train_dataloader):
                # print(batch['identity'].max())
                # print(batch['identity'].min())
                # continue
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].cuda()
                # print(f"load data {self.rank} : {time.time() - start}"); start = time.time()

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    y = self.model.forward(batch)

                    loss = 0
                    if 'logit' in y.keys():
                        loss, logit = self.criteria(batch, y)
                        loss = loss.mean()

                        accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()
                        
                        train_acc += accuracy.item() * len(batch["label"])
                        train_loss += loss.item() * len(batch["label"])
                        count += len(batch["label"])
                        # if self.verbose:
                        #     print("Epoch : {}, {}/{},  Loss : {:.6f}, Acc : {:.3f},".format(epoch, b+1, len(self.train_dataloader), loss.item(), accuracy.item()*100), end='\r')
                        l, c = logit.argmax(dim=-1).unique(return_counts=True)
                    if 'consistency_loss' in y.keys():
                        loss = loss + 0.1 * y['consistency_loss'].mean()
                    if 'identity' in y.keys():
                        loss = loss + 0.5 * y['identity'].mean()
                    if 'contrastive_loss' in y.keys():
                        loss = loss + 0.5 * y['contrastive_loss'].mean()
                    if "audio_key_loss" in y.keys():
                        loss = loss + 0.1 * y["audio_key_loss"]
                    if "text_key_loss" in y.keys():
                        loss = loss + 0.1 * y["text_key_loss"]
                    if "InfoNCE" in y.keys():
                        loss = loss + 0.1 * y["InfoNCE"]
                    if self.is_MT:
                        loss = 0.9 * loss + 0.1 * F.cross_entropy(y['sentiment'], batch['sentiment'], reduction='mean')
                    if 'audio' in y.keys():
                        loss = loss + y['audio'] * 0.5
                    if 'text' in y.keys():
                        loss = loss + y['text'] * 0.5
                    if 'video' in y.keys():
                        loss = loss + y['video'] * 0.5
                    if 'similarity' in y.keys():
                        loss = loss + y['similarity']

                # Zero the gradients
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                gc.collect()
                print(f"Epoch : {epoch}, {b}/{len(self.train_dataloader)}, Loss : {loss.item():.6f}, Acc : {accuracy.item()*100:.3f}", end='\r')
            print(flush=True)
            gc.collect()
            self.train_sampler.epoch += 1
            self.train_sampler.set_epoch(self.train_sampler.epoch)
            train_acc /= count
            train_loss /= count
            scheduler.step()
            optimizer.param_groups[2]['lr'] = 5e-4

            self.model.eval()
            with torch.no_grad():
                accuracy = 0
                loss     = 0
                tp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                tn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)

                label = []
                pred = []

                for i, batch in enumerate(self.val_dataloader):
                    print(f"Validation {i}/{len(self.val_dataloader)}", end='\r')
                    # Move the batch to GPU if CUDA is available
                    for key in batch:
                        batch[key] = batch[key].cuda()

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

                    label.append(batch["label"])
                    pred.append(logit.argmax(dim=-1))

                label = torch.cat(label, dim=0)
                pred = torch.cat(pred, dim=0)

                if self.distributed:
                    label = label.cuda()
                    pred = pred.cuda()

                    accuracy = accuracy.cuda()
                    loss = loss.cuda()
                    tp = tp.cuda()
                    fp = fp.cuda()
                    fn = fn.cuda()
                    tn = tn.cuda()

                    label = all_gather(label)
                    pred = all_gather(pred)
                    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
                    dist.all_reduce(fp, op=dist.ReduceOp.SUM)
                    dist.all_reduce(fn, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tn, op=dist.ReduceOp.SUM)

                    label = torch.cat(label, dim=0)
                    pred = torch.cat(pred, dim=0)

                    label = label.cpu()
                    pred = pred.cpu()
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
                print(f"Epoch : {epoch}, Loss : {loss.item():.6f}, Train Loss : {train_loss:.6f}, Train Acc : {train_acc*100:.3f},\nAcc : {accuracy.item()*100:.3f}, Loss : {loss.item():.6f}, Weighted F1 : {weighted_f1_score.item()*100:.3f}, F1 : ", *(f1_score*100).cpu().tolist()) 

                # Print a confusion matrix
                for a in range(self.num_class):
                    for p in range(self.num_class):
                        a = label == a
                        p = pred == p
                        print(f"{(a & p).sum().item():5d}", end=' ')
                    print()
            
            if hasattr(self.model_without_ddp, 'post_epoch'):
                self.model_without_ddp.post_epoch(self.val_dataloader)
            sys.stdout.flush()
            gc.collect()
        
    def init_distributed(self):
        if self.distributed:
            if torch.cuda.is_available():
                self.gpu    = self.local_rank % self.ngpus_per_node
                self.device = torch.device(self.gpu)
                if self.distributed:
                    self.local_rank = self.gpu
                    self.rank = self.node_rank * self.ngpus_per_node + self.gpu
                    time.sleep(self.rank * 0.1) # prevent port collision
                    print(f"rank {self.rank}/{self.world_size} is running on GPU {self.device}")
                    torch.cuda.set_device(self.device)
                    dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                            world_size=self.world_size, rank=self.rank)
                    self.setup_for_distributed(self.is_main_process())
        else: self.device = torch.device('cpu')

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
        if self.distributed: return dist.get_rank()
        return 0