#!/usr/bin/env python
# coding: utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
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
    def __init__(self, path):
        self.path = path
    
    def standardization(self):
        """データを標準化(平均0, 分散1)を行う関数"""
        df = pd.read_csv(self.path)
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, [-1]].values
    
        self.x_stdsc = StandardScaler()
        self.y_stdsc = StandardScaler()
        self.x_std = self.x_stdsc.fit_transform(self.x)
        self.y_std = self.y_stdsc.fit_transform(self.y)
        
    def rmse_score(self, y_true, y_pred):
        """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def gaussian_regression(self, kernel):
        """
        ガウス過程回帰を行う関数
        
        Parameter
        ----------
        kernel : ガウス過程回帰におけるカーネルを指定
        
        """
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)
        self.model.fit(self.x_std, self.y_std)
        
    def gbdt(self):
        """勾配ブースティングによる回帰を行う関数"""
        model = xgb.XGBRegressor()

        # ハイパーパラメータ探索(グリッドサーチ)
        model_cv = GridSearchCV(model, {"max_depth": [2, 4, 6, 8, 10], "n_estimators": [30, 50, 100, 200]}, verbose=0)
        model_cv.fit(self.x_std, self.y_std)

        # 改めて最適パラメータで学習
        self.model = xgb.XGBRegressor(**model_cv.best_params_)
        self.model.fit(self.x_std, self.y_std)
        
    def ga2m(self):
        """一般化加法モデルによる回帰を行う関数"""
        self.model = ExplainableBoostingRegressor()
        self.model.fit(self.x_std, self.y_std)  
    
    def cross_validation(self):
        """Cross Validationによるモデルの性能検証を行う関数"""
        kf = KFold(n_splits=7, shuffle=True, random_state=0)
        score_funcs = {
            "rmse": make_scorer(self.rmse_score)
        }

        scores = cross_validate(self.model, self.x_std, self.y_std, cv=kf, scoring=score_funcs)
        mean_rmse = scores["test_rmse"].mean()
        return mean_rmse
    
    def visualization(self, method=None):
        """予測結果の可視化を行う関数"""
        # ガウス過程回帰の場合は分散も可視化するため, 処理を分けている
        # TODO: 横軸をDayにする必要はない. parity-plotに変更すること
        # TODO: それぞれの手法の予測結果をsubplotでまとめて表示したい
        if method == "GPR":
            y_pred_std, y_var_std = self.model.predict(self.x_std, return_std=True)
            y_pred = self.y_stdsc.inverse_transform(y_pred_std); y_var = y_var_std * self.y_stdsc.scale_
            y_std = y_var ** 0.5
            day_total = np.linspace(1, len(y_pred), len(y_pred))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(day_total, self.y, c="b", label="raw data")
            ax.plot(day_total, y_pred, "r", label="prediction")
            ax.fill_between(day_total, y_pred.reshape(-1)-y_std, y_pred.reshape(-1)+y_std, 
                            alpha=0.3, color="steelblue", label="1σ")
            ax.set_xlabel("Day")
            ax.set_ylabel("Width")
            ax.set_title(method)
            ax.legend(loc="best")
        else:
            y_pred_std = self.model.predict(self.x_std)
            y_pred = self.y_stdsc.inverse_transform(y_pred_std)
            day_total = np.linspace(1, len(y_pred), len(y_pred))
            
            fig, ax = plt.subplots(figsize=(10,5))
            ax.scatter(day_total, self.y, c="b", label="raw data")
            ax.plot(day_total, y_pred, "r", label="prediction")
            ax.set_xlabel("Day")
            ax.set_ylabel("Width")
            ax.set_title(method)
            ax.legend(loc="best")
    
        # show plots
        fig.tight_layout()
        plt.savefig("result/" + str(method) + ".png", dpi=100)
        plt.show()
