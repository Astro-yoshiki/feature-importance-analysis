#!/usr/bin/env python
# coding: utf-8

import itertools
import os

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, data_path=None, save_path=None):
        self.data_path = data_path
        if save_path is None:
            save_path = "../Data/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path
        self.idx = None

    def read_data(self):
        """Function to get the index of the input"""
        df = pd.read_csv(self.data_path)
        x = df.iloc[:, :-1]
        self.idx = x.columns.to_list()

    def input_combination(self, verbose=1):
        """Function to get a combination of inputs

        Parameters
        ----------
        verbose : int(Select 1 for logging, 0 for not.)
        """
        result = []
        for n in range(2, len(self.idx) + 1):
            for comb in itertools.combinations(self.idx, n):
                result.append(list(comb))
        if verbose:
            print("Number of Input Combination: {}".format(len(result)))

        df = pd.DataFrame(np.zeros([len(result), len(self.idx)]), columns=self.idx)
        for i, comb in enumerate(result):
            for item in comb:
                if item in self.idx:
                    df[item][i] = 1.
            if verbose:
                if i % 1000 == 0 and i != 0:
                    print("{} Data Finished!".format(i))
                if i == len(result) - 1:
                    print("Processing Completed!")
        df["RMSE"] = 0.
        df.to_csv(self.save_path + "input_combination.csv", index=False, encoding="utf-8")
