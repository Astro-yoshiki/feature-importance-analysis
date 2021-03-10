#!/usr/bin/env python
# coding: utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel as Wh
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
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


class ModelSelection:
    def __init__(self, path_) -> None:
        """
        前処理を事前に行う. ここではStandardScalerを用いた標準化をしている
        @type path_: string
        """
        df = pd.read_csv(path_)
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, [-1]].values
        self.x_train, self.x_test, self.y_train, self.y_test =\
            train_test_split(self.x, self.y, test_size=0.1, random_state=0)
        self.x_sc = StandardScaler()
        self.y_sc = StandardScaler()
        self.x_train_std = self.x_sc.fit_transform(self.x_train)
        self.x_test_std = self.x_sc.transform(self.x_test)
        self.y_train_std = self.y_sc.fit_transform(self.y_train)
        self.y_test_std = self.y_sc.transform(self.y_test)

    @staticmethod
    def rmse_score(y_true: np.array, y_pred: np.array) -> float:
        """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def cross_validation(self, model_) -> float:
        """Cross Validationによるモデルの性能検証を行う関数"""
        kf = KFold(n_splits=6, shuffle=True, random_state=0)
        score_funcs = {
            "rmse": make_scorer(self.rmse_score, greater_is_better=False)
        }
        scores = cross_validate(model_, self.x_train_std, self.y_train_std, cv=kf, scoring=score_funcs)
        mean_rmse = scores["test_rmse"].mean()
        return mean_rmse * (-1)

    def gaussian_process(self, kernel_=None) -> object:
        """ガウス過程回帰を行う関数 """
        gp_ = GaussianProcessRegressor(kernel=kernel_, n_restarts_optimizer=30)
        result_ = self.cross_validation(model_=gp_)
        gp_.fit(self.x_train_std, self.y_train_std)
        return result_, gp_

    def gbdt(self) -> object:
        """勾配ブースティングによる回帰を行う関数"""
        gbdt_ = xgb.XGBRegressor()
        result_ = self.cross_validation(model_=gbdt_)
        gbdt_.fit(self.x_train_std, self.y_train_std)
        return result_, gbdt_

    def ga2m(self) -> object:
        """一般化加法モデルによる回帰を行う関数"""
        ga2m_ = ExplainableBoostingRegressor()
        result_ = self.cross_validation(model_=ga2m_)
        ga2m_.fit(self.x_train_std, self.y_train_std)
        return result_, ga2m_

    def visualization(self, model_, method=None) -> None:
        """予測結果の可視化を行う関数"""
        if method == "Ensemble":
            y_train_pred = np.zeros((len(model_), len(self.y_train)))
            y_test_pred = np.zeros((len(model_), len(self.y_test)))
            for i in range(len(model_)):
                # 訓練データ
                y_train_pred_std = model_[i].predict(self.x_train_std).reshape(-1)
                y_train_pred[i] = self.y_sc.inverse_transform(y_train_pred_std)
                # テストデータ
                y_test_pred_std = model_[i].predict(self.x_test_std).reshape(-1)
                y_test_pred[i] = self.y_sc.inverse_transform(y_test_pred_std)
            y_train_pred_mean = np.mean(y_train_pred, axis=0)
            y_test_pred_mean = np.mean(y_test_pred, axis=0)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(self.y_train, y_train_pred_mean, "bo", label="Train")
            ax.plot(self.y_test, y_test_pred_mean, "ro", label="Test")
            ax.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)], color="black", linestyle="dashed")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Prediction")
            ax.set_title(method)
            ax.legend(loc="best")
        else:
            y_train_pred_std = model_.predict(self.x_train_std)
            y_train_pred = self.y_sc.inverse_transform(y_train_pred_std)
            y_test_pred_std = model_.predict(self.x_test_std)
            y_test_pred = self.y_sc.inverse_transform(y_test_pred_std)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(self.y_train, y_train_pred, "bo", label="Train")
            ax.plot(self.y_test, y_test_pred, "ro", label="Test")
            ax.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)], color="black", linestyle="dashed")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Prediction")
            ax.set_title(method)
            ax.legend(loc="best")

        # save plots
        fig.tight_layout()
        plt.savefig("../figure/prediction_{}.png".format(str(method)), dpi=100)
        plt.close()


if __name__ == "__main__":
    label = "length"
    path = "../data/processed_data/model_selection_{}.csv".format(label)

    # ガウス過程回帰
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + Wh(0.01, (1e-2, 1e2))
    ms = ModelSelection(path)
    result, gp = ms.gaussian_process(kernel_=kernel)
    print("RMSE of GPR: ", result)
    ms.visualization(model_=gp, method="GPR")

    # 勾配ブースティング
    ms = ModelSelection(path)
    result, gbdt = ms.gbdt()
    print("RMSE of GBDT: ", result)
    ms.visualization(model_=gbdt, method="GBDT")

    # 一般化加法モデル
    ms = ModelSelection(path)
    result, ga2m = ms.ga2m()
    print("RMSE of GA2M: ", result)
    ms.visualization(model_=ga2m, method="GA2M")

    # アンサンブル
    ms = ModelSelection(path)
    model = [gp, gbdt, ga2m]
    ms.visualization(model_=model, method="Ensemble")
