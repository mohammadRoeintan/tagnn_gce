# 02_global_graph.py
import pickle, argparse, os
from collections import defaultdict

def build_global_graph(dataset='yoochoose1_64', topN=12, eps=3):
    root = f'datasets/{dataset}'
    with open('/kaggle/input/yoochoose1-64/yoochoose1_64/train.txt') as f:
        train_seqs = [list(map(int, line.strip().split(','))) for line in f]
    freq = defaultdict(int)
    for seq in train_seqs:
        for i in range(len(seq)):
            for j in range(max(0,i-eps), min(len(seq),i+eps+1)):
                if i!=j: freq[(seq[i],seq[j])] += 1
    adj = defaultdict(list)
    for (u,v),w in freq.items(): adj[u].append((v,w))
    for u in adj: adj[u] = sorted(adj[u], key=lambda x:-x[1])[:topN]
    with open(f'{root}/global_graph.pkl','wb') as f: pickle.dump(dict(adj),f)
    print(f'global_graph.pkl saved for {dataset}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['yoochoose1_64','diginetica'], default='yoochoose1_64')
    args = parser.parse_args()
    build_global_graph(args.dataset)