#!/usr/bin/env python
# coding: utf-8
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.3


class Plot:
    def __init__(self, data_path=None, cond_path=None, save_path=None, best_model_path=None):
        self.data_path = data_path
        if cond_path is None:
            cond_path = "../Data/input_combination.csv"
        if save_path is None:
            save_path = "../Result/"
        if best_model_path is None:
            best_model_path = "../Result/best_model.pickle"

        self.cond_path = cond_path
        self.save_path = save_path
        self.best_model_path = best_model_path
        self.importance = None
        self.index = None
        self.y_train_pred = None
        self.y_test_pred = None

        df = pd.read_csv(self.data_path)
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, [-1]].values
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=0.1, random_state=0)
        self.x_sc = StandardScaler()
        self.y_sc = StandardScaler()
        self.x_train_std = self.x_sc.fit_transform(self.x_train)
        self.x_test_std = self.x_sc.transform(self.x_test)
        self.y_train_std = self.y_sc.fit_transform(self.y_train)
        self.y_test_std = self.y_sc.transform(self.y_test)

    def rank_table(self, selection_ratio=0.001):
        df = pd.read_csv(self.cond_path)
        # RMSEを小さい順に並び替え, 上位1%を抽出
        df_ranked = df.sort_values("RMSE", ascending=True)
        slice_idx = int(np.ceil(len(df) * selection_ratio))
        df_ranked = df_ranked[:slice_idx]
        df_variable = df_ranked.iloc[:, :-1]
        self.importance = np.sum(df_variable, axis=0)
        self.index = df_variable.columns

    def predict_by_best_model(self):
        model = pickle.load(open(self.best_model_path, "rb"))
        y_train_pred_std = model.predict(self.x_train_std)
        self.y_train_pred = self.y_sc.inverse_transform(y_train_pred_std)
        y_test_pred_std = model.predict(self.x_test_std)
        self.y_test_pred = self.y_sc.inverse_transform(y_test_pred_std)

    def feature_importance(self, xticks_rotation: bool = True, title: str = None):
        # yを昇順ソート後、逆順にindexを取得
        sorted_index = np.argsort(self.importance)[::-1]
        # 棒グラフの可視化
        plt.figure(figsize=(10, 5))
        plt.bar(
            # ラベルが数値だと自動ソートされるため、x軸は文字列型にしておく
            self.index[sorted_index].astype("str"),
            np.sort(self.importance)[::-1]
        )
        if title is not None:
            plt.title(title)
        if xticks_rotation:
            plt.xticks(rotation=90)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        plt.savefig(self.save_path + "feature_importance.png", dpi=100, bbox_inches="tight")
        plt.close()

    def parity_plot(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.y_train, self.y_train_pred, "bo", label="Train")
        ax.plot(self.y_test, self.y_test_pred, "ro", label="Test")
        ax.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)], color="black", linestyle="dashed")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Prediction")
        ax.legend(loc="best")

        # save plots
        fig.tight_layout()
        plt.savefig(self.save_path + "parity-plot.png", dpi=100)
        plt.close()
