# -*- coding: utf-8 -*-
# main_optimized_final.py

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
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm

# --- پارسرهای ورودی با تمام قابلیت‌ها ---
parser = argparse.ArgumentParser(description="GCE-TAGNN v2 - Final Optimized Version")
parser.add_argument('--dataset', default='diginetica', help='نام پوشه دیتاست: diginetica یا yoochoose.')
parser.add_argument('--batch_size', type=int, default=100, help='اندازه بچ.')
parser.add_argument('--hidden_size', type=int, default=100, help='اندازه لایه‌های پنهان.')
parser.add_argument('--epoch', type=int, default=30, help='تعداد کل اپک‌های آموزش.')
parser.add_argument('--lr', type=float, default=0.001, help='نرخ یادگیری اولیه.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='گام کاهش نرخ یادگیری (بر حسب اپک).')
parser.add_argument('--l2', type=float, default=1e-5, help='جریمه L2 (Weight Decay).')
parser.add_argument('--gnn_step', type=int, default=1, help='تعداد گام‌های انتشار در GNN.')
parser.add_argument('--n_heads', type=int, default=4, help='تعداد سرها در Multi-Head Attention.')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='نرخ دراپ‌اوت.')
parser.add_argument('--patience', type=int, default=10, help='تعداد اپک برای توقف زودهنگام.')
parser.add_argument('--n_neg_samples', type=int, default=100, help='تعداد نمونه‌های منفی برای آموزش.')
parser.add_argument('--resume_path', type=str, default=None, help='مسیر فایل checkpoint برای ادامه آموزش.')
opt = parser.parse_args()

print(f"OPTIONS: {opt}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =====================================================================================
# بخش ۲: توابع کمکی برای ساخت گراف و دیتاست
# =====================================================================================
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
# بخش ۳: تعریف کامل مدل بهینه شده
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

class GCE_TAGNN_v2_Optimized(nn.Module):
    def __init__(self, num_items, hidden_size, gnn_step, n_heads, dropout_rate):
        super(GCE_TAGNN_v2_Optimized, self).__init__(); self.num_items = num_items; self.hidden_size = hidden_size; self.n_heads = n_heads
        self.embedding = nn.Embedding(self.num_items, self.hidden_size, padding_idx=0)
        self.session_gnn = SessionGNN(self.hidden_size, step=gnn_step)
        self.position_embedding = nn.Embedding(200, self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.n_heads, dropout=dropout_rate, batch_first=True)
        self.w_target = nn.Linear(self.hidden_size, self.hidden_size, bias=False); self.w_3 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False); self.dropout_global = nn.Dropout(p=dropout_rate)
        self.reset_parameters()
    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1: nn.init.xavier_uniform_(weight)

    def forward(self, session_items, session_len, session_adj, global_embeds, targets=None):
        session_global_embeds = global_embeds[session_items]
        session_local_embeds_initial = self.embedding(session_items); session_local_embeds = self.session_gnn(session_adj, session_local_embeds_initial); rich_session_embeds = self.dropout_global(session_global_embeds) + session_local_embeds
        positions = torch.arange(session_items.shape[1], device=session_items.device).unsqueeze(0); reversed_positions = session_len.unsqueeze(1) - 1 - positions; reversed_positions[session_items == 0] = 0; pos_embeds = self.position_embedding(reversed_positions)
        final_session_embeds = rich_session_embeds + pos_embeds
        last_item_idx = session_len - 1; last_item_embed = final_session_embeds[torch.arange(final_session_embeds.shape[0]).long(), last_item_idx]; s_local = last_item_embed
        key_padding_mask = (session_items == 0)
        attn_output, _ = self.multihead_attn(last_item_embed.unsqueeze(1), final_session_embeds, final_session_embeds, key_padding_mask=key_padding_mask); s_global = attn_output.squeeze(1)
        
        if targets is not None:
            candidate_embeds = self.embedding(targets)
            trans_candidates = self.w_target(candidate_embeds).transpose(1, 2)
            target_attention_scores = torch.matmul(final_session_embeds, trans_candidates)
        else:
            candidate_embeds = self.embedding.weight[1:]
            trans_candidates = self.w_target(candidate_embeds).transpose(0, 1)
            target_attention_scores = torch.matmul(final_session_embeds, trans_candidates)
        
        mask = (session_items == 0).unsqueeze(-1).expand_as(target_attention_scores); target_attention_scores = target_attention_scores.masked_fill(mask, -torch.inf); target_attention_scores = F.softmax(target_attention_scores, dim=1)
        s_target = torch.matmul(target_attention_scores.transpose(1, 2), final_session_embeds)
        s_local_re = s_local.unsqueeze(1).expand_as(s_target); s_global_re = s_global.unsqueeze(1).expand_as(s_target); combined_session_vec = torch.cat([s_target, s_local_re, s_global_re], dim=2); final_session_vec = self.w_3(combined_session_vec)
        
        if targets is not None:
             scores = torch.sum(final_session_vec * candidate_embeds, dim=2)
        else:
             candidate_embeds_expanded = candidate_embeds.unsqueeze(0).expand(final_session_vec.shape[0], -1, -1)
             scores = torch.sum(final_session_vec * candidate_embeds_expanded, dim=2)
        return scores


# =====================================================================================
# بخش ۴: توابع آموزش و ارزیابی
# =====================================================================================
def train_epoch(model, global_gnn, train_loader, optimizer, num_items, global_adj):
    model.train(); global_gnn.train(); total_loss = 0.0
    with torch.no_grad():
        all_item_embeddings = model.embedding.weight
        global_embeds = global_gnn(global_adj, all_item_embeddings)
    pbar = tqdm(train_loader, desc=f"Training Epoch", dynamic_ncols=True)
    for seq, pos_targets, lengths, adj_matrix in pbar:
        seq, pos_targets, lengths, adj_matrix = seq.to(device), pos_targets.to(device), lengths.to(device), adj_matrix.to(device)
        neg_targets = torch.randint(1, num_items, (pos_targets.shape[0], opt.n_neg_samples), device=device)
        targets = torch.cat([pos_targets.unsqueeze(1), neg_targets], dim=1)
        labels = torch.zeros_like(targets, dtype=torch.float32); labels[:, 0] = 1.0
        optimizer.zero_grad()
        scores = model(seq, lengths, adj_matrix, global_embeds, targets)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))
    return total_loss / len(train_loader)

def evaluate(model, global_gnn, test_loader, global_adj):
    model.eval(); global_gnn.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        all_item_embeddings = model.embedding.weight
        global_embeds = global_gnn(global_adj, all_item_embeddings)
        for seq, target, lengths, adj_matrix in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
            seq, lengths, adj_matrix = seq.to(device), lengths.to(device), adj_matrix.to(device)
            scores = model(seq, lengths, adj_matrix, global_embeds)
            _, top_preds = torch.topk(scores, k=20, dim=1); all_preds.append(top_preds.cpu()); all_targets.append(target.cpu())
    all_preds = torch.cat(all_preds); all_targets = torch.cat(all_targets).unsqueeze(1); correct_preds = (all_preds == (all_targets - 1))
    hr20 = correct_preds.any(dim=1).float().mean().item()
    ranks = (correct_preds.to(torch.float).argmax(dim=1) + 1).to(torch.float); reciprocal_ranks = 1.0 / ranks; miss_mask = (correct_preds.sum(dim=1) == 0); reciprocal_ranks[miss_mask] = 0
    mrr20 = reciprocal_ranks.sum().item() / len(all_targets)
    return hr20, mrr20


# =====================================================================================
# بخش ۵: اجرای اصلی برنامه
# =====================================================================================
def main():
    try:
        print("مرحله ۱: در حال بارگذاری فایل‌های pickle..."); dataset_name = opt.dataset
        data_folder = 'yoochoose1_64' if dataset_name == 'yoochoose' else dataset_name
        train_path = os.path.join(data_folder, 'train.txt'); test_path = os.path.join(data_folder, 'test.txt'); all_seq_path = os.path.join(data_folder, 'all_train_seq.txt')
        if not all([os.path.exists(p) for p in [train_path, test_path, all_seq_path]]):
            print(f"خطا: فایل‌های پیش‌پردازش شده یافت نشد. لطفاً preprocess.py را اجرا کنید."); return
        train_data = pickle.load(open(train_path, 'rb')); test_data = pickle.load(open(test_path, 'rb')); all_train_seq = pickle.load(open(all_seq_path, 'rb'))
        print("فایل‌ها با موفقیت بارگذاری شدند.")

        print("مرحله ۲: در حال محاسبه تعداد آیتم‌ها..."); num_items = 0
        for seq in all_train_seq:
            if len(seq) > 0 and max(seq) > num_items: num_items = max(seq)
        for seq in test_data[0]:
            if len(seq) > 0 and max(seq) > num_items: num_items = max(seq)
        num_items += 1; print(f"تعداد کل آیتم‌های منحصر به فرد: {num_items}")

        global_adj = build_global_graph(all_train_seq, num_items)
        train_dataset = SessionDataset(train_data); test_dataset = SessionDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
        print("مرحله ۳: ساخت گراف و لودرها با موفقیت انجام شد.")

        global_gnn = GlobalGNN(opt.hidden_size, step=opt.gnn_step).to(device)
        model = GCE_TAGNN_v2_Optimized(num_items=num_items, hidden_size=opt.hidden_size, gnn_step=opt.gnn_step, n_heads=opt.n_heads, dropout_rate=opt.dropout_rate).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(global_gnn.parameters()), lr=opt.lr, weight_decay=opt.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=0.1)
        print("مرحله ۴: مدل و بهینه‌ساز با موفقیت ساخته شدند.")
        
        start_epoch = 0; best_mrr = 0
        if opt.resume_path:
            if os.path.exists(opt.resume_path):
                print(f"در حال بارگذاری checkpoint از مسیر: {opt.resume_path}")
                checkpoint = torch.load(opt.resume_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict']); global_gnn.load_state_dict(checkpoint['global_gnn_state_dict']); optimizer.load_state_dict(checkpoint['optimizer_state_dict']); scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1; best_mrr = checkpoint.get('best_mrr', 0)
                print(f"مدل با موفقیت بارگذاری شد. آموزش از اپک {start_epoch+1} ادامه می‌یابد.")
            else: print(f"هشدار: مسیر checkpoint یافت نشد. آموزش از ابتدا.")

        print("مرحله ۵: شروع چرخه آموزش...")
        patience_counter = 0
        for epoch in range(start_epoch, opt.epoch):
            train_loss = train_epoch(model, global_gnn, train_loader, optimizer, num_items, global_adj)
            hr20, mrr20 = evaluate(model, global_gnn, test_loader, global_adj)
            print(f"\n--- Epoch {epoch+1}/{opt.epoch} Summary ---")
            print(f"Train Loss: {train_loss:.4f} | HR@20: {hr20*100:.2f}% | MRR@20: {mrr20*100:.2f}%")
            print("-" * 60)
            if mrr20 > best_mrr:
                best_mrr = mrr20; patience_counter = 0; checkpoint_path = f'checkpoint_{opt.dataset}.pt'
                print(f"مدل بهتر یافت شد! در حال ذخیره checkpoint در '{checkpoint_path}'...")
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'global_gnn_state_dict': global_gnn.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_mrr': best_mrr}, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= opt.patience: print(f"توقف زودهنگام..."); break
            scheduler.step()
        print("آموزش تمام شد.")

    except Exception as e:
        print("\n !!! یک خطا در حین اجرای برنامه رخ داد !!!"); import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()