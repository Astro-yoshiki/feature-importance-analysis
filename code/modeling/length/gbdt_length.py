#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
import pickle # モデルの保存用
import math

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

data_path = "../../../data/processed_data/master_data.csv"
cond_path = "../../../data/processed_data/input_combination.csv"
model_save_path = "result/"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)


def rmse_score(y_true, y_pred):
    """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return rmse


def choose_input(model_id, path=cond_path):  # kwargは後に置く
    """機械学習に用いる入力を選択する関数"""
    df = pd.read_csv(path)
    df_ex = df.iloc[model_id, 1:]
    ind_not_zero = df_ex[df_ex != 0].index  # 0でない列名を取得
    return ind_not_zero.to_list()


def main(conds, model_id, result, path1=data_path, path2=cond_path):
    # データの読み込み
    df = pd.read_csv(path1)
    x = pd.DataFrame(np.zeros((len(df), len(conds))), columns=conds)
    for cond in conds:
        x[cond] = df[cond].values
    y = df.iloc[:, [-2]].values
    # 正規化
    x_stdsc = StandardScaler()
    y_stdsc = StandardScaler()
    x_std = x_stdsc.fit_transform(x)
    y_std = y_stdsc.fit_transform(y)

    # 勾配ブースティングを使用
    model = xgb.XGBRegressor()

    # ハイパーパラメータ探索
    model_cv = GridSearchCV(model, {"max_depth": [2, 4, 6, 8], "n_estimators": [50, 100, 200]}, verbose=0)
    model_cv.fit(x_std, y_std)

    # 改めて最適パラメータで学習
    model = xgb.XGBRegressor(**model_cv.best_params_)
    model.fit(x_std, y_std)

    kf = KFold(n_splits=7, shuffle=True, random_state=0)

    score_funcs = {
        'rmse': make_scorer(rmse_score),
    }

    # cross validationによる評価
    scores = cross_validate(model, x_std, y_std, cv=kf, scoring=score_funcs)
    mean_rmse = scores["test_rmse"].mean()
    print("RMSE:", mean_rmse)

    # save RMSE
    result.append(mean_rmse)
    # save model to file
    pickle.dump(model, open(model_save_path + "xgb_model_length_" + str(model_id) + ".pickle", "wb"))

    """
    # load model from file
    loaded_model = pickle.load(open("xgb_model.pickle", "rb"))
    """


if __name__ == "__main__":
    df_result = pd.read_csv(cond_path)
    result = []
    num = 10
    start_time = time.time()

    # モデル評価開始
    for i in range(num):
        conds = choose_input(i, cond_path)
        main(conds, i, result=result, path1=data_path, path2=cond_path)
    df_result.iloc[:num, -1] = result
    df_result.to_csv("model_selection.csv", encoding="utf-8")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time[s]: {:.3f}".format(elapsed_time))