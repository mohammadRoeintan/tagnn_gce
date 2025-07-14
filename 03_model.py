# 03_model.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class GGNN(nn.Module):
    def __init__(self, n_items, emb_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.gru = nn.GRUCell(2*emb_dim, emb_dim)
        self.W_out = nn.Linear(emb_dim, emb_dim)
    def forward(self, adj, items):
        x = self.embedding(items)
        batch, n, d = x.size()
        adj_out, adj_in = adj[:,0], adj[:,1]
        a_out = torch.bmm(adj_out, x)
        a_in  = torch.bmm(adj_in,  x)
        a = torch.cat([a_out, a_in], -1)
        h = self.gru(a.view(-1,2*d), x.view(-1,d))
        return self.W_out(h).view(batch,n,d)

class TAGNN_Base(nn.Module):
    def __init__(self, n_items, emb_dim=100):
        super().__init__()
        self.n_items = n_items
        self.embedding = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.ggnn = GGNN(n_items, emb_dim)
        self.W = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W3 = nn.Linear(3*emb_dim, emb_dim)
    def forward(self, seq, lens):
        batch, max_len = seq.size()
        device = seq.device
        adj_out = torch.zeros(batch,max_len,max_len).to(device)
        adj_in  = torch.zeros(batch,max_len,max_len).to(device)
        for b in range(batch):
            for i in range(lens[b]-1):
                adj_out[b,i,i+1] = 1
                adj_in[b,i+1,i]  = 1
            adj_out[b,range(max_len),range(max_len)] = 1
            adj_in[b,range(max_len),range(max_len)]  = 1
        adj = torch.stack([adj_out, adj_in], 1)
        h = self.ggnn(adj, seq)
        mask = (seq>0).float()
        h_last = h[torch.arange(batch), lens-1]
        # target attention (dummy target = last item)
        target = seq[torch.arange(batch), lens-1]
        target_emb = self.embedding(target)
        att = torch.softmax(torch.sum(h * self.W(target_emb).unsqueeze(1), -1), -1)
        h_global = (h * att.unsqueeze(-1) * mask.unsqueeze(-1)).sum(1)
        sess = torch.cat([h_global, h_last, target_emb], -1)
        sess = torch.tanh(self.W3(sess))
        return torch.matmul(sess, self.embedding.weight.t())

class GlobalEncoder(nn.Module):
    def __init__(self, n_items, emb_dim=100, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.W1 = nn.Linear(emb_dim+1, emb_dim)
        self.q1 = nn.Linear(emb_dim, 1, bias=False)
        self.W2 = nn.Linear(2*emb_dim, emb_dim)
        self.layers = layers
    def forward(self, nodes, adj, s_vec):
        x = self.emb(nodes)
        for _ in range(self.layers):
            nei, wei = [], []
            for n in nodes.cpu().tolist():
                if n in adj: nei.append([v for v,_ in adj[n]]), wei.append([w for _,w in adj[n]])
                else: nei.append([n]), wei.append([1.0])
            max_len = max([len(n) for n in nei])
            nei_t = [F.pad(torch.tensor(n),(0,max_len-len(n))) for n in nei]
            wei_t = [F.pad(torch.tensor(w,dtype=torch.float),(0,max_len-len(w))) for w in wei]
            nei_t, wei_t = torch.stack(nei_t).to(x.device), torch.stack(wei_t).to(x.device)
            h_nei = self.emb(nei_t)
            score = self.q1(torch.tanh(self.W1(torch.cat([h_nei * s_vec.unsqueeze(1), wei_t.unsqueeze(-1)],-1)))).squeeze(-1)
            att = F.softmax(score,-1)
            msg = (att.unsqueeze(-1)*h_nei).sum(1)
            x = F.relu(self.W2(torch.cat([x,msg],-1)))
        return x

class TAGNN_GCE_PLUS(nn.Module):
    def __init__(self, n_items, emb_dim=100):
        super().__init__()
        self.tagnn = TAGNN_Base(n_items, emb_dim)
        self.global_enc = GlobalEncoder(n_items, emb_dim)
        self.gate = nn.Sequential(nn.Linear(2*emb_dim, emb_dim), nn.Tanh(), nn.Linear(emb_dim,1))
        # adjacency will be loaded later
    def forward(self, seq, lens, adj, s_vec):
        logits_base = self.tagnn(seq, lens)
        last = seq[torch.arange(seq.size(0)), lens-1]
        h_global = self.global_enc(last, adj, s_vec)
        # reversed position weight (simple)
        pos = torch.arange(seq.size(1), device=seq.device).flip(0).float()/seq.size(1)
        pos_emb = pos.unsqueeze(0).unsqueeze(-1)
        h_global = h_global * (1-pos_emb[:,0,:])
        gate = torch.sigmoid(self.gate(torch.cat([logits_base.mean(1,keepdim=True), h_global],-1)))
        sess = gate * h_global + (1-gate) * logits_base.mean(1,keepdim=True)
        return torch.matmul(sess, self.tagnn.embedding.weight.t())