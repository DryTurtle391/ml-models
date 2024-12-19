import numpy as np
import pandas as pd
import copy


class Linear_Regression_Multi:

    def __init__(self, alpha, num_iters):
        self.alpha = alpha
        self.num_iters = num_iters

    def fit(self, X, Y, w, b):

        self.m, self.n = X.shape

        self.X = X
        self.Y = Y
        self.w = w
        self.b = b

    def compute_cost(self, w, b):

        cost = 0

        for i in range(self.m):
            fwb_i = np.dot(w, self.X[i]) + b

            cost = cost + (fwb_i - self.Y[i]) ** 2

        total_cost = cost / 2 * self.m

        return total_cost

    def compute_gradient(self, w, b):
        dj_dw = 0
        dj_db = 0

        for i in range(self.m):
            fwb_i = np.dot(w, self.X[i]) + b

            dj_dw_i = np.dot(self.X[i], (fwb_i - self.Y[i]))
            dj_db_i = fwb_i - self.Y[i]

            dj_dw = dj_dw + dj_dw_i
            dj_db = dj_db + dj_db_i

        dj_dw = dj_dw / self.m
        dj_db = dj_db / self.m

        return dj_dw, dj_db

    def gradient_descent(self, w_in, b_in):

        w = copy.deepcopy(w_in)
        b = b_in

        J_history = []
        Para_history = []

        for i in range(self.num_iters):

            dj_dw, dj_db = self.compute_gradient(w, b)

            if np.all(dj_dw == 0) and np.all(dj_db == 0):
                break

            w = w - np.dot(self.alpha, dj_dw)
            b = b - np.dot(self.alpha, dj_db)

            if i < 100000 and i % 100 == 0:
                J_history.append(self.compute_cost(w, b))
                Para_history.append([w, b])

        return w, b, J_history, Para_history
