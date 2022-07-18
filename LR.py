import pandas as pd
import numpy as np
import os

df_train = pd.read_csv("dx_train.csv")
df_test = pd.read_csv("dx_test.csv")


class LR:
    def __init__(self, epoch=5, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        self.X, self.y = np.concatenate([X, ones], axis=1), y
        self.weights = self._init_weights(X)
        for epoch in range(self.epoch):
            out = self._predict()
            loss = self._loss(out, self.y)

    def predict(self, X):
        pass

    def predprob(self, x):
        pass

    def _init_weights(self, X):
        col_num = X.shape[-1] + 1
        weights = np.random.normal(size=(col_num, ))
        return weights

    def _sigmoid(self, x):
        return 1 / (np.exp(x) + 1)

    def _predict(self):
        logits = -self.weights * self.X
        return self._sigmoid(logits)

    def _update_weight(self):
        pass

    def _loss(self, out, y):
        sample_num = out.shape[0]
        print(out.shape, y.shape)
        loss_vals = y * np.log(out) + (1 - y) * np.log((1 - out))
        print(loss_vals.shape)

    def _metric(self):
        pass

    def _cal_gradients(self):
        pass



if __name__ == '__main__':
    y_train = df_train["是否流失"]
    X_train = df_train.drop(columns=["是否流失"])
    lr = LR()
    lr.fit(X_train, y_train)





