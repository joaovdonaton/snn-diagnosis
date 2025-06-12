import argparse
import torch
from make_dataset import load_dataset, MatrixDataset
import time
from models.rn18 import model_18
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW

#device = 'cuda' if torch.cuda_is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-d', '--dataset', required=True)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-r', '--learn_rate', type=float, default=0.005)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-w', '--weight_decay', type=float, default=0.01)

# extract args
args = parser.parse_args()
OUT_PATH = args.output if args.output.endswith('/') else args.output+'/'
EPOCHS, LR, BATCH_SIZE, WEIGHT_DECAY = args.epochs, args.learn_rate, args.batch_size, args.weight_decay

# some stuff we need
def precision(tp, fp): 
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1_score(prec, rec):
    return 2 * ((prec*rec)/(prec+rec))

# load dataset
metadata, train_set, train_labels, train_origins, validation_set, validation_labels, validation_origins, _, _, _ = load_dataset(args.dataset)

ds = MatrixDataset(torch.from_numpy(train_set), torch.from_numpy(train_labels))
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

dsv = MatrixDataset(torch.from_numpy(validation_set), torch.from_numpy(validation_labels))
val_loader = DataLoader(dsv, batch_size=BATCH_SIZE, shuffle=True)

# train logic
model = model_18
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0)

for epoch in range(EPOCHS):
    print(f'Running epoch {epoch}...')
    t0 = time.time()

    # validation set 
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for data, labels in val_loader:
            out = model(data)
            
            v, ind = torch.max(out, 1)

            loss_sum += loss_fn(out, labels).item()
            
            correct += torch.sum(ind == labels).item()

            total += data.shape[0]

            fp += ((labels != ind) & (labels == 0)).sum()
            fn += ((labels != ind) & (labels == 1)).sum()
            tp += ((labels == ind) & (labels == 1)).sum()
            tn += ((labels == ind) & (labels == 0)).sum()
    
    print(f'Val set run, loss={(loss_sum/len(val_loader))}. {(100*(correct/total)):.2f}% accuracy ({correct}/{total})')
    print(f'\t- f1 score: {f1_score(precision(tp, fp), recall(tp, fn)):2f} (precision: {precision(tp, fp)}, recall {recall(tp, fn)})')

    # update weights
    model.train()
    loss_sum, correct, total = 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for data, labels in train_loader:
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    
    print(f'Train loss: {(loss_sum/len(train_loader))}')

    print(f'Took {(time.time()-t0):.2f} seconds -----------------------------')