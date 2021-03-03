#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import itertools

# データの読み込み
datapath = "../data/processed_data/"
df_master = pd.read_csv(os.path.join(datapath, "master_data.csv"))
lis = df_master.columns[:-2].to_list()  # カラム名の抽出

# 組み合わせを取得
result = []
for n in range(2, len(lis)+1):
    for conb in itertools.combinations(lis, n):
        result.append(list(conb))  # タプルをリスト型に変換
print("Number of Input Combination: {}".format(len(result)))

# 空のDataFrameを作成
df = pd.DataFrame(np.zeros([len(result), len(lis)]), columns=lis)
for i, conb in enumerate(result):
    if i % 1000 == 0 and i != 0:
        print("{} Data Finished!".format(i))
    if i == len(result)-1:
        print("Processing Completed!")
    for item in conb:
        if item in lis:
            df[item][i] = 1.
df.to_csv(datapath + "input_combination.csv", encoding="utf-8")
