import numpy as np
import math, copy


class Linear_Regression:

    # initiating the parameters
    def __init__(self, alpha, num_iters):
        self.alpha = alpha
        self.num_iters = num_iters

    # Accepts dependent and independent variables
    # X = Independent variable (numpy array)
    # Y = Dependent variable (numpy array)
    def fit(self, X, Y, w, b):

        # m = no. of data points
        # n = no. of features per data point
        self.m = X.shape[0]

        # w = weight (as many as features per data point, numpy arrray)
        # b = bias (random single value)
        self.w = w
        self.b = b

        self.X = X
        self.Y = Y

    # compute cost for specific w and b.
    # Ref eq (1)
    def compute_cost(self, w, b):
        cost = 0

        for i in range(self.m):
            f_wb = w * self.X[i] + b
            cost = cost + (f_wb - self.Y[i]) ** 2

        total_cost = cost * (1 / (2 * self.m))
        return total_cost

    # Compute the gradient at sepcific w and b.
    # Ref eq (4) & (5)
    def compute_gradient(self, w, b):

        dj_dw = 0
        dj_db = 0
        for i in range(self.m):
            f_wb = w * self.X[i] + b

            dj_dw_i = self.X[i] * (f_wb - self.Y[i])
            dj_db_i = f_wb - self.Y[i]

            dj_dw = dj_dw + dj_dw_i
            dj_db = dj_db + dj_db_i

        dj_dw = dj_dw / self.m
        dj_db = dj_db / self.m

        return dj_dw, dj_db

    # Implement gradient descent to minimize cost function for a specific input w and b
    # Ref eq (3)
    def gradient_descent(self, w_in, b_in):
        w = copy.deepcopy(w_in)
        b = b_in
        self.J_history = []
        self.p_history = []

        for i in range(self.num_iters):
            dj_dw, dj_db = self.compute_gradient(w, b)

            # Break if convergence reached
            if dj_dw == 0 and dj_db == 0:
                break

            w = w - self.alpha * dj_dw
            b = b - self.alpha * dj_db

            if i < 10000:
                self.J_history.append(self.compute_cost(w, b))
                self.p_history.append([w, b])

        return w, b  # , self.J_history, self.p_history
