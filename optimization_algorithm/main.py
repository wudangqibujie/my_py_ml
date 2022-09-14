import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func(w1, w2):
    return 10 * w1 ** 2 + w2 ** 3


def cal_gradient(w1, w2):
    return [20 * w1, 3 * (w2 ** 2)]



def simple_gd(w1, w2, learning_rate, epochs, alpha=1.):
    print("******************** sgd **********************")
    history_w1, history_w2 = [], []
    history_target = []
    for epoch in range(1, epochs + 1):
        target = func(w1, w2)
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        w1 -= learning_rate * (gd1 * alpha)
        w2 -= learning_rate * (gd2 * alpha)
    return history_w1, history_w2, history_target


def adagrad(w1, w2, learning_rate, epochs, epi=0.1):
    print("******************** adagrad **********************")
    history_w1, history_w2 = [], []
    history_target = []
    w = np.array([w1, w2])
    for epoch in range(1, epochs + 1):
        target = func(w[0], w[1])
        w1, w2 = w[0], w[1]
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        g = np.array([gd1, gd2])
        Gg = g * g
        w = w - (learning_rate / np.sqrt(Gg + epi)) * g
    return history_w1, history_w2, history_target


def RMSprop(w1, w2, learning_rate, epochs, epi=1e-7, beta=0.9):
    print("******************** RMSprop **********************")
    history_w1, history_w2 = [], []
    history_target = []
    Gt = np.array([0, 0])
    w = np.array([w1, w2])
    for epoch in range(1, epochs + 1):
        target = func(w[0], w[1])
        w1, w2 = w[0], w[1]
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        g = np.array([gd1, gd2])
        Gg = beta * Gt + (1 - beta) * g * g
        w = w - (learning_rate / np.sqrt(Gg + epi)) * g
    return history_w1, history_w2, history_target


def adadelta(w1, w2, epochs, epi=1e-7, beta1=0.9, beta2=0.9):
    print("******************** adadelta **********************")
    history_w1, history_w2 = [], []
    history_target = []
    Gt = np.array([0, 0])
    w = np.array([w1, w2])
    diff_sq_X = np.array([0, 0])
    diff_w_t_1 = np.array([0, 0])
    for epoch in range(1, epochs + 1):
        target = func(w[0], w[1])
        w1, w2 = w[0], w[1]
        # print(w1, w2)
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        g = np.array([gd1, gd2])
        diff_sq_X = diff_sq_X * diff_sq_X * beta2 + (1 - beta2) * diff_w_t_1 * diff_w_t_1
        Gg = beta1 * Gt + (1 - beta1) * g * g
        diff_w = (np.sqrt(diff_sq_X + epi) / np.sqrt(Gg + epi)) * g
        w = w - diff_w
        diff_w_t_1 = diff_w
    return history_w1, history_w2, history_target


def momentum(w1, w2, learning_rate, epochs, beta=0.9, alpha=0.1):
    print("******************** momentum **********************")
    history_w1, history_w2 = [], []
    history_target = []
    diff_w = np.array([0, 0])
    for epoch in range(1, epochs + 1):
        target = func(w1, w2)
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        gd = np.array([gd1, gd2])
        diff_w = beta * diff_w - learning_rate * (gd * alpha)
        diff_w1, diff_w2 = diff_w[0], diff_w[1]
        w1 += diff_w1
        w2 += diff_w2
    return history_w1, history_w2, history_target


def nesterov(w1, w2, learning_rate, epochs, beta=0.9, alpha=0.1):
    print("******************** nesterov **********************")
    history_w1, history_w2 = [], []
    history_target = []
    diff_w = np.array([0, 0])
    w = np.array([w1, w2])
    for epoch in range(1, epochs + 1):
        target = func(w1, w2)
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        gd = np.array([gd1, gd2])
        diff_w = beta * diff_w - learning_rate * (gd * alpha) * (beta * diff_w + w)
        diff_w1, diff_w2 = diff_w[0], diff_w[1]
        w1 += diff_w1
        w2 += diff_w2
        w = np.array([w1, w2])
    return history_w1, history_w2, history_target


def Adam(w1, w2, learning_rate, epochs, epi=1e-7, beta1=0.9, beta2=0.99):
    print("******************** Adam **********************")
    history_w1, history_w2 = [], []
    history_target = []
    Gt = np.array([0, 0])
    M = np.array([0, 0])
    w = np.array([w1, w2])
    for epoch in range(1, epochs + 1):
        target = func(w[0], w[1])
        w1, w2 = w[0], w[1]
        if epoch % verbos == 0:
            print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        g = np.array([gd1, gd2])

        M = beta2 * M + (1 - beta2) * g

        Gg = beta1 * Gt + (1 - beta1) * g * g
        w = w - (learning_rate / np.sqrt(Gg + epi)) * M
    return history_w1, history_w2, history_target


init_w1, init_w2 = 10, 15
epoch_num = 400
learning = 0.01
verbos = 10
# history_w1, history_w2, history_target = adagrad(init_w1, init_w2, learning, epoch_num)


fig = plt.figure()
fig.patch.set_facecolor('black')
ax = Axes3D(fig)
x = np.linspace(start=-15, stop=15, num=100)
y = np.linspace(start=-15, stop=15, num=100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)
ax.plot_wireframe(X, Y, Z, color="white")
ax.set_facecolor('black')
history_w1, history_w2, history_target = RMSprop(init_w1, init_w2, learning, epoch_num)
ax.plot3D(history_w1, history_w2, history_target)
ax.scatter3D(history_w1, history_w2, history_target, c=None)

# history_w1, history_w2, history_target = adagrad(init_w1, init_w2, learning, epoch_num)
# ax.plot3D(history_w1, history_w2, history_target, 'yellow')
# ax.scatter3D(history_w1, history_w2, history_target, c=None, cmap="yellow")

# history_w1, history_w2, history_target = simple_gd(init_w1, init_w2, learning, epoch_num, alpha=0.01)
# ax.plot3D(history_w1, history_w2, history_target)
# ax.scatter3D(history_w1, history_w2, history_target, c=None)

# history_w1, history_w2, history_target = nesterov(init_w1, init_w2, learning, epoch_num, alpha=0.01)
# ax.plot3D(history_w1, history_w2, history_target)
# ax.scatter3D(history_w1, history_w2, history_target, c=None)

# history_w1, history_w2, history_target = momentum(init_w1, init_w2, learning, epoch_num, alpha=0.01)
# ax.plot3D(history_w1, history_w2, history_target)
# ax.scatter3D(history_w1, history_w2, history_target, c=None)

history_w1, history_w2, history_target = Adam(init_w1, init_w2, learning, epoch_num)
ax.plot3D(history_w1, history_w2, history_target)
ax.scatter3D(history_w1, history_w2, history_target, c=None)

# history_w1, history_w2, history_target = adadelta(init_w1, init_w2, epoch_num)
# ax.plot3D(history_w1, history_w2, history_target)
# ax.scatter3D(history_w1, history_w2, history_target, c=None)

plt.xlabel("w1")
plt.ylabel("w2")
plt.show()
