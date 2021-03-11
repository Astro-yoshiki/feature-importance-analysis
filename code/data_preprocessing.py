#!/usr/bin/env python
# coding: utf-8

import itertools
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocess:
    def __init__(self, data_path=None, save_path=None):
        self.data_path = data_path
        if save_path is None:
            save_path = "../Data/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path

        df = pd.read_csv(self.data_path)
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, [-1]]
        self.idx = self.x.columns.to_list()

        self.x_std = None
        self.y_std = None

    def scaling(self):
        x_sc = StandardScaler()
        y_sc = StandardScaler()
        self.x_std = x_sc.fit_transform(self.x)
        self.y_std = y_sc.fit_transform(self.y)
        return self.x_std, self.y_std

    def input_combination(self):
        # パラメータの組み合わせを取得
        result = []
        for n in range(2, len(self.idx) + 1):
            for comb in itertools.combinations(self.idx, n):
                result.append(list(comb))  # タプルをリスト型に変換
        print("Number of Input Combination: {}".format(len(result)))

        # 予め空のDataFrameを作成
        df = pd.DataFrame(np.zeros([len(result), len(self.idx)]), columns=self.idx)
        for i, comb in enumerate(result):
            if i % 1000 == 0 and i != 0:
                print("{} Data Finished!".format(i))
            if i == len(result) - 1:
                print("Processing Completed!")
            for item in comb:
                if item in self.idx:
                    df[item][i] = 1.
        df["RMSE"] = 0  # RMSEの行を追加
        df.to_csv(self.save_path + "input_combination.csv", index=False, encoding="utf-8")
