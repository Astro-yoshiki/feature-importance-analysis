#!/usr/bin/env python
# coding: utf-8
import os

import numpy as np
import pandas as pd

# データの読み込み
datapath = "../data/raw_data/"
savepath = "../data/processed_data/"
df_master = pd.read_csv(os.path.join(datapath, "Input_sum_logdata_master.csv"))
df_log = pd.read_csv(os.path.join(datapath, "HISTORY_201909-202007.csv"))
df_date = pd.read_csv(os.path.join(datapath, "dataset_for_aixtal.csv"))

# 出力の得られている日付の取得
ind_not_nan = []
for row in range(len(df_date)):
    if not np.isnan(df_date["B_length"][row]):  # dataframeは[列][行]で取り出す
        date_not_nan = str(df_date["date"][row]).replace("-", "")
        ind_not_nan.append(int(date_not_nan))

# ログデータの前処理
df_log["date"] = pd.to_datetime(df_log["date"])
df_log["Year"] = df_log["date"].apply(lambda x: x.year)
df_log["Month"] = df_log["date"].apply(lambda x: x.month)
df_log["Month"] = df_log["Month"].map("{:02}".format)
df_log["Day"] = df_log["date"].apply(lambda x: x.day)
df_log["Day"] = df_log["Day"].map("{:02}".format)
df_log["Date"] = None
for row in range(len(df_log)):
    df_log["Date"][row] = str(df_log["Year"][row]) + str(df_log["Month"][row]) + str(df_log["Day"][row])
    df_log["Date"][row] = int(df_log["Date"][row])


# 統計量を計算する関数を定義
def calculate_table(df, ind, colname=None, method=None):
    result = np.zeros((len(ind)-1, 1))
    for i in range(len(ind)-1):
        start_ind = ind[i]
        end_ind = ind[i+1]
        if i == 0:
            df_ex = df[(df["Date"] >= start_ind) & (df["Date"] <= end_ind)]
        else:
            df_ex = df[(df["Date"] > start_ind) & (df["Date"] <= end_ind)]
        if method == "sum":
            result[i] = np.sum(df_ex[colname], axis=0)
        elif method == "mean":
            result[i] = np.mean(df_ex[colname], axis=0)
        elif method == "std":
            result[i] = np.std(df_ex[colname], axis=0)
        elif method == "max":
            result[i] = np.amax(df_ex[colname], axis=0)
        elif method == "min":
            result[i] = np.amin(df_ex[colname], axis=0)
    return result


def calculate_each_temperature(df, ind, colname1=None, colname2=None, method=None):
    result_daytime = np.zeros((len(ind)-1, 1))
    result_night = np.zeros((len(ind)-1, 1))
    # 区間内のデータを抽出
    for i in range(len(ind)-1):
        start_ind = ind[i]
        end_ind = ind[i+1]
        if i == 0:
            df_ex = df[(df["Date"] >= start_ind) & (df["Date"] <= end_ind)]
            start_flag = df_ex.index[0]
            end_flag = df_ex.index[-1]
        else:
            df_ex = df[(df["Date"] > start_ind) & (df["Date"] <= end_ind)]
            start_flag = df_ex.index[0]
            end_flag = df_ex.index[-1]
        # 以降df_exを用いている点に注意
        if method == "sum":
            tmp_daytime = []
            tmp_night = []
            for row_ in range(start_flag, end_flag+1):
                if df_ex[colname2][row_] != 0:
                    tmp_daytime.append(df_ex[colname1][row_])
                else:
                    tmp_night.append(df_ex[colname1][row_])
            result_daytime[i] = np.sum(tmp_daytime)
            result_night[i] = np.sum(tmp_night)
        elif method == "mean":
            tmp_daytime = []
            tmp_night = []
            for row_ in range(start_flag, end_flag+1):
                if df_ex[colname2][row_] != 0:
                    tmp_daytime.append(df_ex[colname1][row_])
                else:
                    tmp_night.append(df_ex[colname1][row_])
            result_daytime[i] = np.mean(tmp_daytime)
            result_night[i] = np.mean(tmp_night)
        elif method == "std":
            tmp_daytime = []
            tmp_night = []
            for row_ in range(start_flag, end_flag+1):
                if df_ex[colname2][row_] != 0:
                    tmp_daytime.append(df_ex[colname1][row_])
                else:
                    tmp_night.append(df_ex[colname1][row_])
            result_daytime[i] = np.std(tmp_daytime)
            result_night[i] = np.std(tmp_night)
        elif method == "max":
            tmp_daytime = []
            tmp_night = []
            for row_ in range(start_flag, end_flag+1):
                if df_ex[colname2][row_] != 0:
                    tmp_daytime.append(df_ex[colname1][row_])
                else:
                    tmp_night.append(df_ex[colname1][row_])
            result_daytime[i] = np.amax(tmp_daytime)
            result_night[i] = np.amax(tmp_night)
        elif method == "min":
            tmp_daytime = []
            tmp_night = []
            for row_ in range(start_flag, end_flag+1):
                if df_ex[colname2][row_] != 0:
                    tmp_daytime.append(df_ex[colname1][row_])
                else:
                    tmp_night.append(df_ex[colname1][row_])
            result_daytime[i] = np.amin(tmp_daytime)
            result_night[i] = np.amin(tmp_night)
    return result_daytime, result_night


# 結果の出力
B_temperature_sum_daytime, B_temperature_sum_night = \
    calculate_each_temperature(df_log, ind_not_nan, colname1="ハウス内データ-温度1", colname2="ハウス内推定日射量", method="sum")
df_master["B_temperature_sum_daytime"] = B_temperature_sum_daytime
df_master["B_temperature_sum_night"] = B_temperature_sum_night
df_master.to_csv(savepath + "master_data.csv", encoding="utf-8", float_format="%.3f")
print("Preprocessing Finished!")
