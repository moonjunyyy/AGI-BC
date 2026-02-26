import gc
import sys
import time
import json
import torch
import requests
import argparse
import torch.distributed as dist
from utils.utils import get_dataset
from subprocess import Popen, STDOUT, DEVNULL


class LLama_Inference:
    def __init__(self, args) -> None:
        print(args)

        self.seed = args.seed
        self.llama_path = args.llama_path
        self.data_path = args.data_path
        self.n_shot = args.n_shot
        self.use_CUDA = args.use_CUDA
        self.num_workers = args.num_workers
        self.n_shot = args.n_shot
        self.dataset = args.dataset
        self.batch_size = 32

        self.daemon = Popen([f'{self.llama_path}/llama-server',
                             '-m',
                             f'{self.llama_path}/models/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-F16.gguf',
                             '-ngl',
                             f'{33 if self.use_CUDA else 0}',
                             '--port',
                             '24763',
                             '-np',
                             f'{self.batch_size}',
                             '-ns',
                             f'{self.batch_size}',],
                            stdout=DEVNULL, stdin=DEVNULL, stderr=STDOUT)

    def readln_process(self, process, _timeout=0.1):
        accumulation = []
        while True:
            ret = None
            begin = time.time()
            while time.time() - begin < _timeout:
                try:
                    if process.stdout.peek(1) is None: 
                        raise Exception
                    ret = process.stdout.read(1);
                    break
                except Exception as e: 
                    continue
            accumulation.append(ret)
            if ret is None or ret == b'\n': break
        try:
            accumulation = b''.join(accumulation).decode()
        except Exception as e: 
            accumulation = ''
        return accumulation
    
    def write_process(self, process, input):
        process.stdin.write(input)
        process.stdin.flush()

    def close_process(self, process):
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        process.wait()

    def run(self):
        self.train_dataset, self.val_dataset, self.num_class = get_dataset(self.dataset, self.data_path, None)
        self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        self.val_sampler = torch.utils.data.RandomSampler(self.val_dataset)
        self.example_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers) 
        self.example_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)

        classes = ['NoBC', 'Continuer', 'Understanding', 'Empathic Response']

        train_acc = 0
        count = 0
        labels = []
        print()
        data = {
                "prompt" : [],
                "n_predict": 2048
            }
        tp = torch.tensor([0 for _ in range(self.num_class)])
        fp = torch.tensor([0 for _ in range(self.num_class)])
        fn = torch.tensor([0 for _ in range(self.num_class)])
        tn = torch.tensor([0 for _ in range(self.num_class)])
        for b, batch in enumerate(self.train_dataloader):
            example = [[] for _ in range(self.num_class)]
            for i, batch in enumerate(self.example_dataloader):
                label = batch["label"].item()
                if len(example[label]) < self.n_shot:
                    example[label].append(batch)
                if all([len(e) == self.n_shot for e in example]):
                    break
            prompt = \
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\\n"\
            # "You are a helpful AI assistant.\\\n"\
            "You are a helpful AI assistant.<|eot_id|>\\\n\\\n"\
            "<|start_header_id|>user<|end_header_id|>\\\n"\
            "I am a Korean language learner.<|eot_id|>\\\n\\\n"\
            "<|start_header_id|>assistant<|end_header_id|>\\\n"\
            "Great! I can help you with that.<|eot_id|>\\\n\\\n"\
            "<|start_header_id|>system<|end_header_id|>\\\n"\
            "A backchannel is a brief response made during a conversation that indicates engagement of the listener without adding direct content to the discussion.\\\n"\
            "There are four types of backchannels: NoBC, Continuer, Understanding, and Empathic Response.\\\n"\
            "NoBC: Indicates that no backchanneling occurs; the conversation continues by the speaker or is initiated by the listener.\\\n"\
            "Continuer: A type of backchannel that encourages the speaker to keep talking naturally.\\\n"\
            "Understanding: A backchannel that shows the listener comprehends the conversation well.\\\n"\
            "Empathic Response: A backchannel that conveys emotional empathy or a reaction to what the speaker has said.\\\n"\
            # "You must then decide on one of four responses: NoBC, Continuer, Understanding, or Empathic Response.\\\n"\
            # "You look at the context of the dialogue and choose the most appropriate response out of the four.\\\n"\
            # "You generate a five-word context for each of the four responses.\\\n"\
            "Examples of the Korean five words before the backchannel are shown next.\\\n\\\n"
            # prompt += f"Examples: \\\n\\\n"
            for i, e in enumerate(example):
                for j, batch in enumerate(e):
                    prompt += f"\tThe five words before the {classes[batch['label'][0]]} :\\\n"
                    prompt += f"\t{batch['text'][0]}\\\n"
                    # prompt += f"\t<|start_header_id|>user<|end_header_id|>\\\n"
                    # prompt += f"\t{batch['text'][0]}<|eot_id|>\\\n"
                    # prompt += f"\t<|start_header_id|>assistant<|end_header_id|>\\\n"
                    # prompt += f"\t{classes[batch['label'][0]]}<|eot_id|>\\\n\\\n"

            # prompt += f"<|start_header_id|>system<|end_header_id|>\\\n"
            prompt += "Now, generate a Korean five words before the backchannel for the each category.<|eot_id|>\\\n\\\n"
            # prompt += "<|start_header_id|>user<|end_header_id|>\\\n"

            # input_text = prompt + batch["text"][0] + "<|eot_id|>\\\n"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\\\n"
            data["prompt"].append(prompt)
            labels.append(batch["label"][0])
            # if len(data["prompt"]) == self.batch_size or b == len(self.train_dataloader) - 1:
            if len(data["prompt"]) == self.batch_size or b == 2047:
                print(f"Training {b}/{len(self.train_dataloader)}" + (f" Train Acc : {train_acc / count*100:.3f}" if count != 0 else ""), end="\r", flush=True)
                res = requests.post('http://localhost:24763/completion', data=json.dumps(data))
                res = res.json()['results']
                print(res)
                for i, r in enumerate(res):
                    r = r['content']
                    if classes[labels[i]].lower() in r.lower():
                        train_acc += 1
                    for j in range(self.num_class):
                        if classes[j].lower() == r.lower():
                            if j == labels[i]:
                                tp[j] += 1
                            else:
                                fp[j] += 1
                        else:
                            if j == labels[i]:
                                fn[j] += 1
                            else:
                                tn[j] += 1
                    count += 1
                data = {
                    "prompt" : [],
                    "n_predict": 2048
                }
            if b == 2047: break
        print(f"Train Acc : {train_acc / count * 100:.3f}")
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1_score  = 2 * precision * recall / (precision + recall + 1e-6)
        f1_score = f1_score.nan_to_num(0).detach().cpu()
        number_of_classes = self.train_dataset.get_sample_in_class()
        weighted_f1_score = (f1_score * number_of_classes).sum() / number_of_classes.sum()
        print(f"Weighted F1 : {weighted_f1_score.item()*100:.3f}, F1 : ", *(f1_score*100).cpu().tolist())
        print(flush=True)

        with torch.no_grad():
            accuracy = 0
            loss     = 0
            count    = 0
            tp = torch.tensor([0 for _ in range(self.num_class)])
            fp = torch.tensor([0 for _ in range(self.num_class)])
            fn = torch.tensor([0 for _ in range(self.num_class)])
            tn = torch.tensor([0 for _ in range(self.num_class)])

            label = []
            pred = []

            for b, batch in enumerate(self.val_dataloader):
                input_text = prompt + batch["text"][0] + "<|eot_id|>\\\n"
                input_text += "<|start_header_id|>assistant<|end_header_id|>\\\n"
                data["prompt"].append(input_text)
                labels.append(batch["label"][0])
                if len(data["prompt"]) == self.batch_size or b == 2047:
                # if len(data["prompt"]) == self.batch_size or b == len(self.train_dataloader) - 1:
                    print(f"Training {b}/{len(self.train_dataloader)}" + (f" Test Acc : {accuracy / count*100:.3f}" if count != 0 else ""), end="\r", flush=True)
                    res = requests.post('http://localhost:24763/completion', data=json.dumps(data))
                    res = res.json()['results']
                    for i, r in enumerate(res):
                        r = r['content']
                        if classes[labels[i]].lower() in r.lower():
                            accuracy += 1
                        for j in range(self.num_class):
                            if classes[j].lower() == r.lower():
                                if j == labels[i]:
                                    tp[j] += 1
                                else:
                                    fp[j] += 1
                            else:
                                if j == labels[i]:
                                    fn[j] += 1
                                else:
                                    tn[j] += 1
                        count += 1
                    data = {
                        "prompt" : [],
                        "n_predict": 2048
                    }
                if b == 2047: break
            print(f"Test Acc : {accuracy / count * 100:.3f}")
            precision = tp / (tp + fp + 1e-6)
            recall    = tp / (tp + fn + 1e-6)
            f1_score  = 2 * precision * recall / (precision + recall + 1e-6)
            f1_score = f1_score.nan_to_num(0).detach().cpu()
            number_of_classes = self.train_dataset.get_sample_in_class()
            weighted_f1_score = (f1_score * number_of_classes).sum() / number_of_classes.sum()
            print(f"Weighted F1 : {weighted_f1_score.item()*100:.3f}, F1 : ", *(f1_score*100).cpu().tolist())
            print(flush=True)
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

    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        # dist.all_gather(all_sizes, local_size, async_op=False)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=self.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        # dist.all_gather(all_qs_padded, item)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs
    
    def __del__(self):
        self.daemon.terminate()
        self.daemon.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--llama_path', type=str, default='/home/llama')
    parser.add_argument('--data_path', type=str, default='/local_datasets')
    parser.add_argument('--dataset', type=str, default='ETRI')
    parser.add_argument('--n-shot', type=int, default=1)
    parser.add_argument('--use_CUDA', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    trainer = LLama_Inference(args)
    trainer.run()
    pass

if __name__ == "__main__":
    main()
