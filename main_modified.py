# -*- coding: utf-8 -*-
# main_final_with_resume.py

# =====================================================================================
# بخش ۱: وارد کردن کتابخانه‌ها و تنظیمات
# =====================================================================================
import argparse
import time
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm

parser = argparse.ArgumentParser(description="GCE-TAGNN v2 End-to-End Training with Resume Capability")
parser.add_argument('--dataset', default='diginetica', help='نام پوشه دیتاست: diginetica یا yoochoose1_64.')
parser.add_argument('--batch_size', type=int, default=100, help='اندازه بچ.')
parser.add_argument('--hidden_size', type=int, default=100, help='اندازه لایه‌های پنهان.')
parser.add_argument('--epoch', type=int, default=30, help='تعداد کل اپک‌های آموزش.')
parser.add_argument('--lr', type=float, default=0.001, help='نرخ یادگیری اولیه.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='ضریب کاهش نرخ یادگیری.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='گام کاهش نرخ یادگیری (بر حسب اپک).')
parser.add_argument('--l2', type=float, default=1e-5, help='جریمه L2 (Weight Decay).')
parser.add_argument('--gnn_step', type=int, default=1, help='تعداد گام‌های انتشار در GNN.')
parser.add_argument('--n_heads', type=int, default=4, help='تعداد سرها در Multi-Head Attention.')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='نرخ دراپ‌اوت.')
parser.add_argument('--patience', type=int, default=10, help='تعداد اپک برای توقف زودهنگام.')
parser.add_argument('--resume_path', type=str, default=None, help='مسیر فایل checkpoint برای ادامه آموزش.')
opt = parser.parse_args()

print(f"OPTIONS: {opt}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =====================================================================================
# بخش ۲: توابع کمکی برای ساخت گراف و دیتاست
# =====================================================================================
def load_and_preprocess_data(dataset_name):
    print(f"شروع بارگذاری و پیش‌پردازش دیتاست: {dataset_name}...")
    processed_dir = 'processed_data'; os.makedirs(processed_dir, exist_ok=True)
    paths = {'train': os.path.join(processed_dir, f'train_{dataset_name}.pkl'), 'test': os.path.join(processed_dir, f'test_{dataset_name}.pkl'), 'all_train_seq': os.path.join(processed_dir, f'all_train_seq_{dataset_name}.pkl'), 'item_map': os.path.join(processed_dir, f'item_map_{dataset_name}.pkl')}
    if all([os.path.exists(p) for p in paths.values()]):
        print("فایل‌های پردازش‌شده یافت شد. در حال بارگذاری از کش..."); train_data = pickle.load(open(paths['train'], 'rb')); test_data = pickle.load(open(paths['test'], 'rb')); all_train_seq = pickle.load(open(paths['all_train_seq'], 'rb')); item_map = pickle.load(open(paths['item_map'], 'rb')); num_items = len(item_map) + 1
        return train_data, test_data, all_train_seq, num_items
    print("فایل‌های پردازش‌شده یافت نشد. شروع پیش‌پردازش از ابتدا...")
    if dataset_name == 'diginetica':
        raw_path = os.path.join(dataset_name, 'train-item-views.csv'); raw_data = pd.read_csv(raw_path, sep=','); raw_data.columns = ['session_id', 'user_id', 'item_id', 'timeframe', 'eventdate']; raw_data['eventdate'] = pd.to_datetime(raw_data['eventdate'], unit='s'); raw_data = raw_data.sort_values(['session_id', 'eventdate']); split_date = raw_data['eventdate'].max() - pd.to_timedelta(7, 'd'); train_session_ids = raw_data[raw_data['eventdate'] < split_date]['session_id'].unique(); test_session_ids = raw_data[raw_data['eventdate'] >= split_date]['session_id'].unique()
    elif dataset_name == 'yoochoose1_64':
        raw_path = os.path.join(dataset_name, 'yoochoose-clicks.dat'); raw_data = pd.read_csv(raw_path, sep=',', header=None, usecols=[0, 1, 2], dtype={0: np.int32, 1: str, 2: np.int64}); raw_data.columns = ['session_id', 'timestamp', 'item_id']; raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp']); raw_data = raw_data.sort_values(['session_id', 'timestamp']); split_date = raw_data['timestamp'].max() - pd.to_timedelta(1, 'd'); train_session_ids = raw_data[raw_data['timestamp'] < split_date]['session_id'].unique(); test_session_ids = raw_data[raw_data['timestamp'] >= split_date]['session_id'].unique()
    else: raise ValueError("دیتاست نامعتبر است.")
    session_lengths = raw_data.groupby('session_id').size(); raw_data = raw_data[np.in1d(raw_data.session_id, session_lengths[session_lengths > 1].index)]; item_counts = raw_data['item_id'].value_counts(); raw_data = raw_data[np.in1d(raw_data.item_id, item_counts[item_counts >= 5].index)]; session_lengths = raw_data.groupby('session_id').size(); raw_data = raw_data[np.in1d(raw_data.session_id, session_lengths[session_lengths > 1].index)]; item_ids = raw_data['item_id'].unique(); item_map = {item: i + 1 for i, item in enumerate(item_ids)}; raw_data['item_id'] = raw_data['item_id'].map(item_map); num_items = len(item_map) + 1; sessions = raw_data.groupby('session_id')['item_id'].apply(list); train_sessions = sessions[sessions.index.isin(train_session_ids)]; test_sessions = sessions[sessions.index.isin(test_session_ids)]
    if dataset_name == 'yoochoose1_64':
        print("کاهش مجموعه آموزش به جدیدترین 1/64 جلسات برای Yoochoose..."); train_session_times = raw_data[raw_data['session_id'].isin(train_sessions.index)].groupby('session_id')['timestamp'].max(); latest_session_ids = train_session_times.sort_values(ascending=False).index[:len(train_session_times) // 64]; train_sessions = train_sessions[train_sessions.index.isin(latest_session_ids)]
    def process_sessions_to_sequences(session_data):
        out_seqs, labs = [], [];
        for session in tqdm(session_data, desc="Processing sessions into sequences"):
            if len(session) > 1:
                for i in range(1, len(session)): out_seqs.append(session[:i]); labs.append(session[i])
        return out_seqs, labs
    train_seq, train_lab = process_sessions_to_sequences(train_sessions); test_seq, test_lab = process_sessions_to_sequences(test_sessions); train_data = (train_seq, train_lab); test_data = (test_seq, test_lab); all_train_seq = train_sessions.tolist()
    print("پیش‌پردازش تمام شد. در حال ذخیره فایل‌ها در کش...")
    pickle.dump(train_data, open(paths['train'], 'wb')); pickle.dump(test_data, open(paths['test'], 'wb')); pickle.dump(all_train_seq, open(paths['all_train_seq'], 'wb')); pickle.dump(item_map, open(paths['item_map'], 'wb'))
    return train_data, test_data, all_train_seq, num_items

def build_global_graph(all_train_seq, num_items):
    print("شروع ساخت گراف سراسری...")
    rows, cols = [], [];
    for session in tqdm(all_train_seq, desc="Building global graph edges"):
        for i in range(len(session) - 1): rows.append(session[i]); cols.append(session[i+1]); rows.append(session[i+1]); cols.append(session[i])
    data = np.ones(len(rows)); adj = csr_matrix((data, (rows, cols)), shape=(num_items, num_items)); rowsum = np.array(adj.sum(axis=1)).flatten(); d_inv_sqrt = np.power(rowsum, -0.5); d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = csr_matrix(np.diag(d_inv_sqrt)); adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(); indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)); values = torch.from_numpy(adj_norm.data.astype(np.float32)); shape = torch.Size(adj_norm.shape)
    print("ساخت گراف سراسری تمام شد.")
    return torch.sparse_coo_tensor(indices, values, shape, device=device)

class SessionDataset(Dataset):
    def __init__(self, data): self.sequences, self.labels = data
    def __len__(self): return len(self.sequences)
    def __getitem__(self, index): return self.sequences[index], self.labels[index]

def collate_fn(batch):
    seqs, labs = zip(*batch); lengths = [len(s) for s in seqs]; max_len = max(lengths)
    padded_seqs = torch.zeros(len(seqs), max_len).long()
    for i, seq in enumerate(seqs): padded_seqs[i, :lengths[i]] = torch.LongTensor(seq)
    adj_matrix = torch.zeros(len(seqs), max_len, max_len)
    for i, seq in enumerate(seqs):
        for j in range(len(seq) - 1): adj_matrix[i, j, j + 1] = 1; adj_matrix[i, j + 1, j] = 1
    return padded_seqs, torch.LongTensor(labs), torch.LongTensor(lengths), adj_matrix


# =====================================================================================
# بخش ۳: تعریف کامل مدل
# =====================================================================================
class SessionGNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(SessionGNN, self).__init__(); self.step = step; self.hidden_size = hidden_size; self.input_size = hidden_size * 2; self.gate_size = hidden_size * 3
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size)); self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size)); self.b_ih = nn.Parameter(torch.Tensor(self.gate_size)); self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True); self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A, self.linear_edge_in(hidden)); input_out = torch.matmul(A, self.linear_edge_out(hidden)); inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih); gh = F.linear(hidden, self.w_hh, self.b_hh); i_r, i_i, i_n = gi.chunk(3, 2); h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r); inputgate = torch.sigmoid(i_i + h_i); newgate = torch.tanh(i_n + resetgate * h_n); hy = newgate + inputgate * (hidden - newgate)
        return hy
    def forward(self, A, hidden):
        for _ in range(self.step): hidden = self.GNNCell(A, hidden)
        return hidden

class GlobalGNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GlobalGNN, self).__init__(); self.step = step; self.hidden_size = hidden_size; self.linear_transform = nn.Linear(hidden_size, hidden_size, bias=True)
    def forward(self, A_global, hidden_global):
        for _ in range(self.step): hidden_global = torch.sparse.mm(A_global, hidden_global); hidden_global = self.linear_transform(hidden_global); hidden_global = F.relu(hidden_global)
        return hidden_global

class GCE_TAGNN_v2(nn.Module):
    def __init__(self, num_items, hidden_size, gnn_step, n_heads, dropout_rate):
        super(GCE_TAGNN_v2, self).__init__(); self.num_items = num_items; self.hidden_size = hidden_size; self.n_heads = n_heads
        self.embedding = nn.Embedding(self.num_items, self.hidden_size, padding_idx=0); self.global_gnn = GlobalGNN(self.hidden_size, step=gnn_step); self.session_gnn = SessionGNN(self.hidden_size, step=gnn_step)
        self.position_embedding = nn.Embedding(200, self.hidden_size); self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.n_heads, dropout=dropout_rate, batch_first=True)
        self.w_target = nn.Linear(self.hidden_size, self.hidden_size, bias=False); self.w_3 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False); self.dropout_global = nn.Dropout(p=dropout_rate)
        self.reset_parameters()
    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1: nn.init.xavier_uniform_(weight)
    def forward(self, session_items, session_len, session_adj, global_adj):
        all_item_embeddings = self.embedding.weight; global_item_embeds = self.global_gnn(global_adj, all_item_embeddings); session_global_embeds = global_item_embeds[session_items]
        session_local_embeds_initial = self.embedding(session_items); session_local_embeds = self.session_gnn(session_adj, session_local_embeds_initial); rich_session_embeds = self.dropout_global(session_global_embeds) + session_local_embeds
        positions = torch.arange(session_items.shape[1], device=session_items.device).unsqueeze(0); reversed_positions = session_len.unsqueeze(1) - 1 - positions; reversed_positions[session_items == 0] = 0; pos_embeds = self.position_embedding(reversed_positions)
        final_session_embeds = rich_session_embeds + pos_embeds
        last_item_idx = session_len - 1; last_item_embed = final_session_embeds[torch.arange(final_session_embeds.shape[0]).long(), last_item_idx]; s_local = last_item_embed
        key_padding_mask = (session_items == 0)
        attn_output, _ = self.multihead_attn(last_item_embed.unsqueeze(1), final_session_embeds, final_session_embeds, key_padding_mask=key_padding_mask); s_global = attn_output.squeeze(1)
        candidate_embeds = self.embedding.weight[1:]; trans_candidates = self.w_target(candidate_embeds).transpose(0, 1)
        target_attention_scores = torch.matmul(final_session_embeds, trans_candidates); mask = (session_items == 0).unsqueeze(-1).expand_as(target_attention_scores); target_attention_scores = target_attention_scores.masked_fill(mask, -torch.inf); target_attention_scores = F.softmax(target_attention_scores, dim=1)
        s_target = torch.matmul(target_attention_scores.transpose(1, 2), final_session_embeds)
        s_local_re = s_local.unsqueeze(1).expand_as(s_target); s_global_re = s_global.unsqueeze(1).expand_as(s_target); combined_session_vec = torch.cat([s_target, s_local_re, s_global_re], dim=2); final_session_vec = self.w_3(combined_session_vec)
        candidate_embeds_expanded = candidate_embeds.unsqueeze(0).expand(final_session_vec.shape[0], -1, -1)
        scores = torch.sum(final_session_vec * candidate_embeds_expanded, dim=2)
        return scores


# =====================================================================================
# بخش ۴: توابع آموزش و ارزیابی
# =====================================================================================
def train_epoch(model, train_loader, optimizer, global_adj):
    model.train(); total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training Epoch")
    for seq, target, lengths, adj_matrix in pbar:
        seq, target, lengths, adj_matrix = seq.to(device), target.to(device), lengths.to(device), adj_matrix.to(device)
        optimizer.zero_grad()
        scores = model(seq, lengths, adj_matrix, global_adj)
        loss = F.cross_entropy(scores, target - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))
    return total_loss / len(train_loader)

def evaluate(model, test_loader, global_adj):
    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for seq, target, lengths, adj_matrix in tqdm(test_loader, desc="Evaluating"):
            seq, lengths, adj_matrix = seq.to(device), lengths.to(device), adj_matrix.to(device)
            scores = model(seq, lengths, adj_matrix, global_adj)
            _, top_preds = torch.topk(scores, k=20, dim=1)
            all_preds.append(top_preds.cpu()); all_targets.append(target.cpu())
    all_preds = torch.cat(all_preds); all_targets = torch.cat(all_targets).unsqueeze(1); correct_preds = (all_preds == (all_targets - 1))
    hr20 = correct_preds.any(dim=1).float().mean().item()
    ranks = (correct_preds.to(torch.float).argmax(dim=1) + 1); ranks[correct_preds.sum(dim=1) == 0] = 0
    mrr20 = (1.0 / ranks).sum().item() / len(all_targets)
    return hr20, mrr20


# =====================================================================================
# بخش ۵: اجرای اصلی برنامه
# =====================================================================================
def main():
    train_data, test_data, all_train_seq, num_items = load_and_preprocess_data(opt.dataset)
    global_adj = build_global_graph(all_train_seq, num_items)
    train_dataset = SessionDataset(train_data); test_dataset = SessionDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    model = GCE_TAGNN_v2(num_items=num_items, hidden_size=opt.hidden_size, gnn_step=opt.gnn_step, n_heads=opt.n_heads, dropout_rate=opt.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    
    start_epoch = 0
    best_mrr = 0
    
    # *** تغییر جدید: بارگذاری checkpoint در صورت وجود ***
    if opt.resume_path:
        if os.path.exists(opt.resume_path):
            print(f"در حال بارگذاری checkpoint از مسیر: {opt.resume_path}")
            checkpoint = torch.load(opt.resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_mrr = checkpoint.get('best_mrr', 0) # get best_mrr if it exists
            print(f"مدل با موفقیت بارگذاری شد. آموزش از اپک {start_epoch} ادامه می‌یابد.")
        else:
            print(f"هشدار: مسیر checkpoint مشخص شده یافت نشد: {opt.resume_path}. آموزش از ابتدا شروع می‌شود.")

    print("شروع آموزش مدل...")
    patience_counter = 0
    for epoch in range(start_epoch, opt.epoch):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, global_adj)
        hr20, mrr20 = evaluate(model, test_loader, global_adj)
        
        print(f"\n--- Epoch {epoch+1}/{opt.epoch} Summary ---")
        print(f"Time: {time.time() - start_time:.2f}s | Train Loss: {train_loss:.4f} | HR@20: {hr20*100:.2f}% | MRR@20: {mrr20*100:.2f}%")
        print("-" * (30 + len(str(epoch+1)) + len(str(opt.epoch))))
        
        if mrr20 > best_mrr:
            best_mrr = mrr20
            patience_counter = 0
            checkpoint_path = f'checkpoint_{opt.dataset}.pt'
            print(f"مدل بهتر با MRR@20: {best_mrr*100:.2f}% یافت شد! در حال ذخیره checkpoint در '{checkpoint_path}'...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mrr': best_mrr,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= opt.patience:
                print(f"عملکرد برای {opt.patience} اپک بهبود نیافت. توقف زودهنگام...")
                break
        
        # scheduler.step() را بعد از اپتیمازر و در انتهای هر اپک فراخوانی کنید
        scheduler.step()

    print("آموزش تمام شد.")

if __name__ == '__main__':
    main()