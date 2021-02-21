#!/usr/bin/env python
# coding: utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
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
    def __init__(self, path_, kernel_=None) -> None:
        """
        前処理を事前に行う. ここではStandardScalerを用いた標準化をしている
        @type path_: string
        @type kernel_: object
        """
        df = pd.read_csv(path_)
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, [-1]].values
        self.x_sc = StandardScaler()
        self.y_sc = StandardScaler()
        self.x_std = self.x_sc.fit_transform(self.x)
        self.y_std = self.y_sc.fit_transform(self.y)

        self.gp = GaussianProcessRegressor(kernel=kernel_, n_restarts_optimizer=30)
        # FIXME: ここでインスタンス変数にすると, 「"XGBRegressor" object is not callable :152」と表示される...
        # self.gbdt = xgb.XGBRegressor()
        # self.ga2m = ExplainableBoostingRegressor()

    @staticmethod
    def rmse_score(y_true: np.array, y_pred: np.array) -> float:
        """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def cross_validation(self, model_) -> float:
        """Cross Validationによるモデルの性能検証を行う関数"""
        kf = KFold(n_splits=7, shuffle=True, random_state=0)
        score_funcs = {
            "rmse": make_scorer(self.rmse_score)
        }

        scores = cross_validate(model_, self.x_std, self.y_std, cv=kf, scoring=score_funcs)
        mean_rmse = scores["test_rmse"].mean()
        return mean_rmse

    def gaussian_process(self) -> object:
        """ガウス過程回帰を行う関数 """
        result_ = self.cross_validation(model_=self.gp)
        return result_, self.gp

    def gbdt(self) -> object:
        """勾配ブースティングによる回帰を行う関数"""
        gbdt_ = xgb.XGBRegressor()
        result_ = self.cross_validation(model_=gbdt_)
        return result_, self.gbdt

    def ga2m(self) -> object:
        """一般化加法モデルによる回帰を行う関数"""
        ga2m_ = ExplainableBoostingRegressor()
        result_ = self.cross_validation(model_=ga2m_)
        return result_, self.ga2m

    def visualization(self, model_, method=None) -> None:
        """予測結果の可視化を行う関数"""
        # ガウス過程回帰の場合は分散も可視化するため, 処理を分けている
        # TODO: 横軸をDayにする必要はない. parity-plotに変更すること(GPRの分岐は要るか...)
        # TODO: それぞれの手法の予測結果をsubplotでまとめて表示したい
        # TODO: アンサンブル予測結果を示したい

        if method == "GPR":
            y_pred_std, y_var_std = model_.predict(self.x_std, return_std=True)
            y_pred = self.y_sc.inverse_transform(y_pred_std)
            y_var = y_var_std * self.y_sc.scale_
            y_std = y_var ** 0.5
            day_total = np.linspace(1, len(y_pred), len(y_pred))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(day_total, self.y, c="b", label="raw data")
            ax.plot(day_total, y_pred, "r", label="prediction")
            ax.fill_between(day_total, y_pred.reshape(-1) - y_std, y_pred.reshape(-1) + y_std,
                            alpha=0.3, color="steelblue", label="1σ")
            ax.set_xlabel("Day")
            ax.set_ylabel("Length")
            ax.set_title(method)
            ax.legend(loc="best")
        else:
            y_pred_std = model_.predict(self.x_std)
            y_pred = self.y_sc.inverse_transform(y_pred_std)
            day_total = np.linspace(1, len(y_pred), len(y_pred))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(day_total, self.y, c="b", label="raw data")
            ax.plot(day_total, y_pred, "r", label="prediction")
            ax.set_xlabel("Day")
            ax.set_ylabel("Length")
            ax.set_title(method)
            ax.legend(loc="best")

        # show plots
        fig.tight_layout()
        # plt.savefig("result/" + str(method) + ".png", dpi=100)
        plt.show()


if __name__ == "__main__":
    path = "../../../data/processed_data/master_data_length.csv"

    # ガウス過程回帰
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + Wh(0.01, (1e-2, 1e2))
    ms = ModelSelection(path, kernel_=kernel)
    result, gp = ms.gaussian_process()
    print("RMSE: ", result)
    # FIXME: 以下のコードで, 「process finished with exit code 139 (interrupted by signal 11: sigsegv)」というエラーが出る
    # ms.visualization(model_=gp, method="GPR")

    # 勾配ブースティング
    ms = ModelSelection(path, kernel_=None)
    result, gbdt = ms.gbdt()
    print("RMSE: ", result)
    # ms.visualization(model_=gbdt, method="GBDT")

    # 一般化加法モデル
    ms = ModelSelection(path, kernel_=None)
    result, ga2m = ms.ga2m()
    print("RMSE: ", result)
    # ms.visualization(model_=ga2m, method="GA2M")
