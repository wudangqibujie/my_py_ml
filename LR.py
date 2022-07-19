import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

df_train = pd.read_csv("dx_train.csv")
df_test = pd.read_csv("dx_test.csv")
df_mean = df_train.mean()
std_cols = list(df_mean[df_mean > 10].index)
scaler = StandardScaler()
df_train[std_cols] = scaler.fit_transform(df_train[std_cols])

class LR:
    def __init__(self, epoch=5, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        self.X, self.y = np.concatenate([X, ones], axis=1), y
        self.weights = self._init_weights(X)
        for epoch in range(self.epoch):
            out = self._predict()
            loss = self._loss(out, self.y)
            self.grd = self._cal_gradients(out)
            self._update_weight()
            print(epoch, loss)

    def predict(self, X):
        pass

    def predprob(self, x):
        pass

    def _init_weights(self, X):
        col_num = X.shape[-1] + 1
        weights = np.random.normal(size=(col_num, ))
        return weights

    def _sigmoid(self, Z):
        mask = (Z > 0)
        positive_out = np.zeros_like(Z, dtype='float64')
        negative_out = np.zeros_like(Z, dtype='float64')

        # 大于0的情况
        positive_out = 1 / (1 + np.exp(-Z, positive_out, where=mask))
        # 清除对小于等于0元素的影响
        positive_out[~mask] = 0

        # 小于等于0的情况
        expZ = np.exp(Z, negative_out, where=~mask)
        negative_out = expZ / (1 + expZ)
        # 清除对大于0元素的影响
        negative_out[mask] = 0

        return positive_out + negative_out
        # return 1 / (np.exp(x) + 1)

    def _predict(self):
        logits = np.sum(-self.weights * self.X, axis=1)
        return self._sigmoid(logits)

    def _update_weight(self):
        self.weights = self.weights - self.learning_rate * self.grd

    def _loss(self, out, y):
        loss_vals = y * np.log(out + 1e-4) + (1 - y) * np.log((1 - out + 1e-4))
        return -np.mean(loss_vals)

    def _metric(self):
        pass

    def _cal_gradients(self, out):
        grd = (self.y - out) * np.transpose(self.X)
        return grd.sum(axis=1)

if __name__ == '__main__':
    y_train = df_train["是否流失"]
    X_train = df_train.drop(columns=["是否流失"])
    lr = LR(epoch=100)
    lr.fit(X_train.values, y_train.values)