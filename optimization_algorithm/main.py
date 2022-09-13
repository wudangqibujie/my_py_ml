import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func(w1, w2):
    return w1 ** 2 + w2 ** 3


def cal_gradient(w1, w2):
    return [2 * w1, 3 * (w2 ** 2)]


history_w1, history_w2 = [], []
history_target = []
def simple_gd(w1, w2, learning_rate, epochs, alpha=1):
    for epoch in range(1, epochs + 1):
        target = func(w1, w2)
        print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        w1 -= learning_rate * (gd1 * alpha)
        w2 -= learning_rate * (gd2 * alpha)


def adagrad(w1, w2, learning_rate, epochs, epi=0.001):
    w = np.array([w1, w2])
    for epoch in range(1, epochs + 1):
        target = func(w[0], w[1])
        w1, w2 = w[0], w[1]
        print(f"epoch {epoch}, target: {round(target, 4)}, w1: {round(w1, 5)}, w2: {round(w2, 5)}")
        history_w1.append(w1)
        history_w2.append(w2)
        history_target.append(target)
        gd1, gd2 = cal_gradient(w1, w2)
        g = np.array([gd1, gd2])
        Gg = g * g
        w = w - (learning_rate / np.sqrt(Gg + epi) )* g

adagrad(10, 10, 0.01, 100)
# simple_gd(10, 10, 0.01, 100)
history_w1 = np.array(history_w1)
history_w2 = np.array(history_w2)
history_target = np.array(history_target)
fig = plt.figure()
ax = Axes3D(fig)
x = np.linspace(start=-15, stop=15, num=100)
y = np.linspace(start=-15, stop=15, num=100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)
ax.plot_wireframe(X, Y, Z, color="black")
ax.plot3D(history_w1, history_w2, history_target, 'red')
ax.scatter3D(history_w1, history_w2, history_target, c=None, cmap="red")
plt.show()



# ax.plot_surface(X, Y, Z, cmap='viridis')
# # ax.contour3D(X, Y, Z, 50, cmap="binary")
# # ax.plot_wireframe(X, Y, Z, color="black")
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(zline, yline, zline, 'red')
# zdata = 16 * np.random.random(100)
# xdata = np.sin(zdata) + np.random.randn(100)
# ydata = np.cos(zdata) + np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Greens")
# plt.show()





