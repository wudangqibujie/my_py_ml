import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import random

# df_train = pd.read_csv("dx_train.csv")
# df_test = pd.read_csv("dx_test.csv")

df_train = pd.read_csv("dx_train_3cols.csv")
df_test = pd.read_csv("dx_test_3cols.csv")
print(df_train.columns)
df_mean = df_train.mean()
std_cols = list(df_mean[df_mean > 10].index)
scaler = StandardScaler()
df_train[std_cols] = scaler.fit_transform(df_train[std_cols])

class LR:
    def __init__(self, epoch=5, learning_rate=0.001, lambda_=0.1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.lambda_ = lambda_

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        self.X, self.y = np.concatenate([X, ones], axis=1), y
        self.weights = self._init_weights(X)
        for epoch in range(self.epoch):
            out = self._predict()
            # print(out, sum(out))
            loss = self._loss(out, self.y)
            grd = self._cal_gradients(out)
            hessian = self._get_hessian_matrix(out)
            flg = np.linalg.det(hessian)
            print(hessian)
            print(flg)
            # if flg > 0:
            #     self._update_weight_newton(grd, hessian)
            # else:
            self._update_weight(grd)
            if epoch % 100 == 0:
                acores = self.predprob(self.X)
                auc = roc_auc_score(self.y, acores)
                print(epoch, loss, auc)

    def fit_step(self, X, y, batch_size=64):
        idx = [i for i in range(X.shape[0])]
        self.weights = self._init_weights(X)
        for epoch in range(self.epoch):
            random.shuffle(idx)
            block_size = X.shape[0] // batch_size
            loss = 0.
            cnt = 0
            for b in range(block_size):
                batch_idx = idx[b * batch_size: (b + 1) * batch_size]
                batch_X, batch_y = X[batch_idx], y[batch_idx]
                ones = np.ones((batch_X.shape[0], 1))
                self.X, self.y = np.concatenate([batch_X, ones], axis=1), batch_y
                out = self._predict()
                loss_ = self._loss(out, self.y)
                grd = self._cal_gradients(out)
                self._update_weight(grd)
                loss += batch_X.shape[0] * loss_
                cnt += batch_X.shape[0]
            if epoch % 100 == 0:
                ones = np.ones((X.shape[0], 1))
                acores = self.predprob(np.concatenate([X, ones], axis=1))
                auc = roc_auc_score(y, acores)
                print(epoch, loss/cnt, auc)

    def predict(self, X):
        pass

    def predprob(self, x):
        logits = np.sum(-self.weights * x, axis=1)
        return self._sigmoid(logits)

    def _init_weights(self, X):
        col_num = X.shape[-1] + 1
        weights = np.random.uniform(size=(col_num, ))
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

    def _predict(self):
        logits = np.sum(-self.weights * self.X, axis=1)
        return self._sigmoid(logits)

    def _get_hessian_matrix(self, prob):
        val1 , val2 = self.X[:, :, np.newaxis], self.X[:, np.newaxis, :]
        right_val = np.transpose(np.einsum("ijk,ikn->ijn", val1, val2), axes=(1, 2, 0))
        left_val = prob * (1 - prob)
        # print("********** left ***********")
        # print(prob)
        # print(left_val)
        # print("********** right ***********")
        # print(right_val)
        hessian = np.transpose(left_val * right_val, axes=(2, 0, 1)).mean(axis=0)
        return hessian

    def _update_weight_newton(self, grad, hessian):
        self.weights = self.weights - np.matmul(np.linalg.inv(hessian), grad)

    def _update_weight(self, grd):
        # print("************************")
        # print(self.weights)
        # print("************* grd **************")
        # print(self.grd)
        self.weights = self.weights - self.learning_rate * grd
        # print(self.weights)

    def _loss(self, out, y):
        loss_vals = y * np.log(out + 1e-4) + (1 - y) * np.log((1 - out + 1e-4))
        return -np.mean(loss_vals)

    def _metric(self):
        pass

    def _cal_gradients(self, out):
        grd = (self.y - out) * np.transpose(self.X)
        grd_norm = self.lambda_ * self.weights
        grd = grd.mean(axis=1) + grd_norm
        return grd

if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression


    y_train = df_train["是否流失"]
    X_train = df_train.drop(columns=["是否流失"])

    # lr = LogisticRegression(verbose=1, solver="newton-cg")
    # lr.fit(X_train, y_train)
    # scores = lr.predict_proba(X_train)
    # auc = roc_auc_score(y_train, scores[:, 1])
    # print(auc)

    lr = LR(epoch=10000, learning_rate=0.1)
    lr.fit(X_train.values, y_train.values)
    # lr.fit_step(X_train.values, y_train.values, batch_size=1)

    # A = np.ones(shape=(100, 20, 1))
    # B = np.ones(shape=(100, 1, 20))
    # C = np.einsum("ijk,ikn->ijn", A, B)
    # C = np.transpose(C, axes=(1, 2, 0))
    # d = np.random.randn(100)
    # D = np.transpose(d * C, axes=(2, 0, 1))
    #
    # AA = A[:, np.newaxis, :]
    # print(A.shape, AA.shape)