# 01_preprocess.py
import pandas as pd, os, argparse
from pathlib import Path

def preprocess(dataset='yoochoose1_64', min_item=5, min_len=2):
    root = Path('datasets') / dataset
    root.mkdir(parents=True, exist_ok=True)
    file = root / 'yoochoose-clicks.dat' if 'yoochoose' in dataset else root / 'train-item-views.csv'

    if not file.exists():
        raise FileNotFoundError(f'{file} not found. please download first.')

    if 'yoochoose' in dataset:
        df = pd.read_csv(file, header=None, usecols=[0,1,2],
                         names=['session','ts','item'], parse_dates=['ts'])
    else:  # Diginetica
        df = pd.read_csv(file, usecols=[0,1,3], names=['session','user','item'])
    df = df.sort_values(['session', 'ts' if 'yoochoose' in dataset else 'session'])
    # حذف آیتم‌ها و سشن‌های کوتاه
    item_counts = df['item'].value_counts()
    df = df[df['item'].isin(item_counts[item_counts>=min_item].index)]
    seqs = df.groupby('session')['item'].apply(list).reset_index()
    seqs = seqs[seqs['item'].map(len)>=min_len]

    # اسپلیت روز آخر برای تست
    if 'yoochoose' in dataset:
        split_day = df['ts'].max().normalize() - pd.Timedelta(days=1)
        train = seqs[seqs.session.isin(df[df['ts']<split_day]['session'].unique())]
        test  = seqs[seqs.session.isin(df[df['ts']>=split_day]['session'].unique())]
    else:  # Diginetica: هفته آخر
        split_day = df['ts'].max().normalize() - pd.Timedelta(weeks=1)
        train = seqs[seqs.session.isin(df[df['ts']<split_day]['session'].unique())]
        test  = seqs[seqs.session.isin(df[df['ts']>=split_day]['session'].unique())]

    # ذخیره
    with open(root/'train.txt','w') as f:
        for seq in train['item']: f.write(','.join(map(str,seq))+'\n')
    with open(root/'test.txt','w') as f:
        for seq in test['item']:  f.write(','.join(map(str,seq))+'\n')
    print(f'{dataset}: train={len(train)}  test={len(test)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['yoochoose1_64','yoochoose1_4','diginetica'], default='yoochoose1_64')
    args = parser.parse_args()
    preprocess(args.dataset)