import argparse
from trainer import Trainer

is_MT = True    # True for MT, False for ST
use_CUDA = True # True for GPU, False for CPU
batch_size = 64 # Batch size
epochs = 60     # Number of epochs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BPM_MT' if is_MT else 'BPM_ST')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--is_MT', type=bool, default=True)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--use_CUDA', type=bool, default=use_CUDA)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()
    pass

if __name__ == "__main__":
    main()