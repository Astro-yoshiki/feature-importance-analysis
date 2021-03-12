#!/usr/bin/env python
# coding: utf-8
import math
import os
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
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
        if result_path is None:
            result_path = "../Result/"
            if not os.path.exists(result_path):
                os.mkdir(result_path)

        self.cond_path = cond_path
        self.model_save_path = model_save_path
        self.result_path = result_path
        self.idx_not_zero_list = None
        self.df_result = None

    # noinspection PyMethodMayBeStatic
    def rmse_score(self, y_true, y_pred):
        """Function to calculate RMSE

        Parameters
        ----------
        y_true: raw data
        y_pred: predicted value
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        return rmse

    def choose_input(self, model_id):
        """Function to select input for machine learning

        Parameters
        ----------
        model_id: int(Specify the line number.)
        """
        df = pd.read_csv(self.cond_path)
        df_extracted = df.iloc[model_id, :]
        idx_not_zero = df_extracted[df_extracted != 0].index
        idx_not_zero_list = idx_not_zero.to_list()
        return idx_not_zero_list

    def build_model(self, model_id, params_, result_, save_all_models=False, save_best_model=False, save_table=False):
        """Function to create a model using grid search

        Parameters
        ----------
        model_id: int(Specify the line number)
        params_: list(Combination of input)
        result_: list(Where to store the results)
        save_all_models: bool(if True, all models will be saved.)
        save_best_model: bool(if True, best model will be saved.)
        save_table: bool(if True, selected parameters will be saved.)
        """
        df = pd.read_csv(self.data_path)
        x = pd.DataFrame(np.zeros((len(df), len(params_))), columns=params_)
        for param in params_:
            x[param] = df[param].values
        y = df.iloc[:, [-1]].values
        y_idx = df.columns[-1]

        # Normalization using StandardScaler
        x_sc = StandardScaler()
        y_sc = StandardScaler()
        x_std = x_sc.fit_transform(x)
        y_std = y_sc.fit_transform(y)

        # setting of grid search
        param_grid = {"max_depth": [4, 6], "learning_rate":  [0.01, 0.02, 0.05, 0.1]}
        score_funcs = {"rmse": make_scorer(self.rmse_score, greater_is_better=False)}
        kf = KFold(n_splits=7, shuffle=True, random_state=0)

        # Hyperparameter search by grid search
        model = xgb.XGBRegressor(n_estimators=1000)
        model_cv = GridSearchCV(model, param_grid, iid=True, cv=kf, refit=False,
                                scoring=score_funcs["rmse"], n_jobs=-1, verbose=0)
        model_cv.fit(x_std, y_std)

        # Extract the value of RMSE
        mean_rmse = model_cv.cv_results_["mean_test_score"]
        mean_rmse = np.mean(-mean_rmse)  # Convert to positive value
        result_.append(mean_rmse)

        # Train again with optimal parameters
        model = xgb.XGBRegressor(**model_cv.best_params_)
        model.fit(x_std, y_std)

        # save model and table to file
        if save_all_models:
            pickle.dump(model, open("{0}xgb_model_{1}.pickle".format(self.model_save_path, str(model_id)), "wb"))
        if save_best_model:
            pickle.dump(model, open("{}best_model.pickle".format(self.result_path), "wb"))
        if save_table:
            stacked_data = np.hstack([x, y])
            columns = params_ + [y_idx]
            df = pd.DataFrame(stacked_data, columns=columns, index=None)
            df.to_csv("../Data/selected_data.csv", index=False, encoding="utf-8")

    def solver(self, verbose=1):
        """Main function

        Parameters
        ----------
        verbose: int(Select 1 for logging, 0 for not.)
        """
        self.df_result = pd.read_csv(self.cond_path)
        result = []
        num = len(self.df_result)
        start_time = time.time()

        # Start model evaluation
        for i in range(num):
            params = self.choose_input(i)
            self.build_model(i, params_=params, result_=result)
            if verbose:
                if i % 1000 == 0 and i != 0:
                    print("{}th Modeling Finished!".format(i))
                if i == len(self.df_result) - 1:
                    print("Modeling Completed!")
        self.df_result.iloc[:num, -1] = result
        self.df_result.to_csv(self.cond_path, index=False, encoding="utf-8")

        # Display of elapsed time
        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time[s]: {:.3f}".format(elapsed_time))

    def save_best_model(self):
        """Function to save the best model"""
        best_model_idx = np.argmin(self.df_result["RMSE"])
        params = self.choose_input(best_model_idx)
        self.build_model(best_model_idx, params_=params, result_=[],
                         save_best_model=True, save_table=True)
