import argparse
import torch
from trainer import Trainer
from rl_trainer import RLTrainer

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

    if args.model == 'BPM_RL':
        trainer = RLTrainer(args)
    else:   
        trainer = Trainer(args)
    trainer.run()
    pass

if __name__ == "__main__":
    main()