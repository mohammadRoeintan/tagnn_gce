# 04_train_eval.py
import torch, pickle, argparse, numpy as np
from torch.utils.data import Dataset, DataLoader
from 03_model import TAGNN_Base, TAGNN_GCE_PLUS

class SeqDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path) as f:
            self.seqs = [list(map(int, line.strip().split(','))) for line in f]
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[-1])

def collate(batch):
    seqs, labels = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    max_len = lens.max()
    padded = [torch.cat([s, torch.zeros(max_len-len(s), dtype=torch.long)]) for s in seqs]
    return torch.stack(padded), lens, torch.stack(labels)

def train_epoch(model, loader, optimizer, adj, device):
    model.train()
    total_loss = 0
    for seq, lens, label in loader:
        seq, lens, label = seq.to(device), lens.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(seq, lens, adj, seq.float().mean(1))
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, adj, device, k=20):
    model.eval()
    hits, mrrs = [], []
    with torch.no_grad():
        for seq, lens, label in loader:
            seq, lens, label = seq.to(device), lens.to(device), label.to(device)
            logits = model(seq, lens, adj, seq.float().mean(1))
            _, pred = logits.topk(k, dim=-1)
            for l, p in zip(label, pred):
                hits.append(int(l in p))
                mrrs.append(1 / (p.tolist().index(l) + 1) if l in p else 0)
    return np.mean(hits)*100, np.mean(mrrs)*100

def run(dataset='yoochoose1_64', epochs=20, emb_dim=100, batch_size=100, lr=1e-3):
    root = f'datasets/{dataset}'
    # load global graph
    with open(f'{root}/global_graph.pkl','rb') as f: adj = pickle.load(f)
    n_items = max([max([it for seq in SeqDataset(f'{root}/train.txt').seqs for it in seq]),
                   max([it for seq in SeqDataset(f'{root}/test.txt').seqs  for it in seq])]) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = SeqDataset(f'{root}/train.txt')
    test_ds  = SeqDataset(f'{root}/test.txt')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate)

    for name, model in [('TAGNN', TAGNN_Base(n_items, emb_dim)),
                        ('TAGNN-GCE++', TAGNN_GCE_PLUS(n_items, emb_dim))]:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(1, epochs+1):
            train_loss = train_epoch(model, train_loader, optimizer, adj, device)
            p20, m20 = evaluate(model, test_loader, adj, device, k=20)
            print(f'{name}  epoch{ep:02d}  loss={train_loss:.3f}  P@20={p20:.2f}  MRR@20={m20:.2f}')
        print('-'*50)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['yoochoose1_64','diginetica'], default='yoochoose1_64')
    parser.add_argument('--epochs',  type=int, default=20)
    args = parser.parse_args()
    run(dataset=args.dataset, epochs=args.epochs)