#!/usr/bin/env python
# coding: utf-8
import math
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler


class Modeling:
    def __init__(self, data_path=None, cond_path=None, model_save_path=None, result_path=None):
        self.data_path = data_path
        if cond_path is None:
            cond_path = "../Data/input_combination.csv"
        if model_save_path is None:
            model_save_path = "../Model/"
        if result_path is None:
            result_path = "../Result/"

        self.cond_path = cond_path
        self.model_save_path = model_save_path
        self.result_path = result_path
        self.idx_not_zero_list = None
        self.df_result = None

    # noinspection PyMethodMayBeStatic
    def rmse_score(self, y_true, y_pred):
        """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def choose_input(self, model_id):
        """機械学習に用いる入力を選択する関数"""
        df = pd.read_csv(self.cond_path)
        df_extracted = df.iloc[model_id, :]  # FIXME: 列名が合致しているか確認すること
        idx_not_zero = df_extracted[df_extracted != 0].index  # 0でない列名を取得
        idx_not_zero_list = idx_not_zero.to_list()
        return idx_not_zero_list

    def build_model(self, model_id, params_, result_, save_all_models=False, return_best=False):
        # データの読み込みと抽出
        df = pd.read_csv(self.data_path)
        x = pd.DataFrame(np.zeros((len(df), len(params_))), columns=params_)
        for param in params_:
            x[param] = df[param].values
        y = df.iloc[:, [-1]].values

        # 正規化
        x_sc = StandardScaler()
        y_sc = StandardScaler()
        x_std = x_sc.fit_transform(x)
        y_std = y_sc.fit_transform(y)

        # グリッドサーチの設定
        param_grid = {"max_depth": [4, 6], "learning_rate":  [0.01, 0.02, 0.05, 0.1]}
        score_funcs = {"rmse": make_scorer(self.rmse_score, greater_is_better=False)}
        kf = KFold(n_splits=7, shuffle=True, random_state=0)

        # グリッドサーチによるハイパーパラメータ探索
        model = xgb.XGBRegressor(n_estimators=1000)
        model_cv = GridSearchCV(model, param_grid, iid=True, cv=kf, refit=False,
                                scoring=score_funcs["rmse"], n_jobs=-1, verbose=0)
        model_cv.fit(x_std, y_std)

        # RMSEの抽出
        mean_rmse = model_cv.cv_results_["mean_test_score"]
        mean_rmse = np.mean(-mean_rmse)  # 正の値に変換
        # save RMSE
        result_.append(mean_rmse)

        # 改めて最適パラメータで学習
        model = xgb.XGBRegressor(**model_cv.best_params_)
        model.fit(x_std, y_std)
        # save model to file
        if save_all_models:
            pickle.dump(model, open("{0}xgb_model_{1}.pickle".format(self.model_save_path, str(model_id)), "wb"))
        if return_best:
            pickle.dump(model, open("{}best_model.pickle".format(self.result_path), "wb"))
        else:
            pass

    def solver(self):
        df_result = pd.read_csv(self.cond_path)
        result = []
        num = len(df_result)
        start_time = time.time()

        # モデル評価開始
        for i in range(num):
            params = self.choose_input(i)
            self.build_model(i, params_=params, result_=result, save_all_models=False, return_best=False)
            if i % 1000 == 0 and i != 0:
                print("{}th Modeling Finished!".format(i))
            if i == len(df_result) - 1:
                print("Modeling Completed!")
        self.df_result.iloc[:num, -1] = result
        self.df_result.to_csv(self.cond_path, encoding="utf-8")

        # 経過時間の表示
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time[s]: {:.3f}".format(elapsed_time))

    def save_best_model(self):
        best_model_idx = np.argmin(self.df_result["RMSE"])
        params = self.choose_input(best_model_idx)
        self.build_model(best_model_idx, params_=params, result_=[], save_all_models=False, return_best=True)
