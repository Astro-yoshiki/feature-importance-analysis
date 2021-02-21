#!/usr/bin/env python
# coding: utf-8
import math
import os
import pickle  # モデルの保存用
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

data_path = "../data/processed_data/master_data.csv"
cond_path = "../data/processed_data/input_combination.csv"
model_save_path = "../model/length/"
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


def main(conds_, model_id, result_, path=data_path):
    # データの読み込み
    df = pd.read_csv(path)
    x = pd.DataFrame(np.zeros((len(df), len(conds_))), columns=conds_)
    for cond in conds_:
        x[cond] = df[cond].values
    y = df.iloc[:, [-2]].values
    # 正規化
    x_stdsc = StandardScaler()
    y_stdsc = StandardScaler()
    x_std = x_stdsc.fit_transform(x)
    y_std = y_stdsc.fit_transform(y)

    # グリッドサーチの設定
    param_grid = {"max_depth": [4, 6],
                  "learning_rate":  [0.01, 0.05, 0.1],
                  'min_samples_leaf': [3, 5, 9, 17]
                  }
    score_funcs = {
        "rmse": make_scorer(rmse_score, greater_is_better=False)
    }
    kf = KFold(n_splits=7, shuffle=True, random_state=0)

    # グリッドサーチによるハイパーパラメータ探索
    model = xgb.XGBRegressor(n_estimators=1000)
    model_cv = GridSearchCV(model, param_grid, iid=True, cv=kf, scoring=score_funcs["rmse"],
                            refit=False, n_jobs=-1, verbose=0)
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
    pickle.dump(model, open(model_save_path + "xgb_model_length_" + str(model_id) + ".pickle", "wb"))

    """
    # load model from file
    loaded_model = pickle.load(open("xgb_model.pickle", "rb"))
    """


if __name__ == "__main__":
    df_result = pd.read_csv(cond_path)
    result = []
    # TODO: len(df_result)に変更
    num = 10
    start_time = time.time()

    # モデル評価開始
    for i in range(num):
        conds = choose_input(i, cond_path)
        main(conds, i, result_=result, path=data_path)
        if i % 1000 == 0 and i != 0:
            print("{}th Modeling Finished!".format(i))
        if i == len(df_result) - 1:
            print("Modeling Completed!")
    df_result.iloc[:num, -1] = result
    df_result.to_csv("../data/processed_data/variable_selection_length.csv", encoding="utf-8")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time[s]: {:.3f}".format(elapsed_time))
