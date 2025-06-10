import argparse
import torch
from make_dataset import load_dataset, MatrixDataset
import time
from models.ann_conv import MFCC_CNN
from torch.utils.data import DataLoader

#device = 'cuda' if torch.cuda_is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-d', '--dataset', required=True)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-r', '--learn_rate', type=float, default=0.005)
parser.add_argument('-b', '--batch_size', type=int, default=128)

# extract args
args = parser.parse_args()
OUT_PATH = args.output if args.output.endswith('/') else args.output+'/'
EPOCHS, LR, BATCH_SIZE = args.epochs, args.learn_rate, args.batch_size

# load dataset
metadata, train_set, train_labels, train_origins, validation_set, validation_labels, validation_origins, _, _, _ = load_dataset(args.dataset)

ds = MatrixDataset(train_set, train_labels)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# train logic
model = MFCC_CNN()

for epoch in range(EPOCHS):
    print(f'Running epoch {epoch}...')
    t0 = time.time()

    model.train()
    for data, labels in train_loader:
        print(data)
        print(labels)
        exit()


    print(f'Took {(t0-time.time()):.2f} seconds')