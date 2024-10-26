import matplotlib.pyplot as plt
import numpy as np
import sys, os
from contextlib import contextmanager
from matplotlib import animation

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def gradient_descent_animation(x0, alpha, n_iter, epsilon=1e-06):
    x = x0
    x_arr = [x]
    y_arr = [x**2]
    for _ in range(n_iter):
        difference = alpha * 2 * x
        if abs(difference) > epsilon:
            x = x - difference
        x_arr.append(x)
        y_arr.append(x**2)

    line_x = np.linspace(-abs(x0)-1, abs(x0)+1, 100)
    line_y = line_x ** 2
    fig, ax = plt.subplots()
    text = ax.text(0.75, 0.05, f'Iterations: {0}',
            transform=ax.transAxes,
            verticalalignment='top')
    ax.plot(line_x, line_y, color="black")
    scat = ax.scatter([], [], c="r", s=30)
    ax.set(xlim=[-abs(x0)-1, abs(x0)+1], ylim=[-1, x0**2+1], xlabel='X', ylabel='Y')
    def update(frame):
        data = np.stack([x_arr[:frame + 1], y_arr[:frame + 1]]).T
        scat.set_offsets(data)
        text.set_text(f'Iterations: {frame}')
        return scat


    ani = animation.FuncAnimation(fig, update, frames=len(x_arr), interval=700)
    return ani

def gradient_descent_lr_animation(X, Y, a, b, alpha, n_iter, epsilon = 1e-06):
    fig, ax = plt.subplots()
    text = ax.text(0.6, 0.05, f'Iterations: {0}',
                   transform=ax.transAxes,
                   verticalalignment='top')
    ax.scatter(X, Y, c="b", s=30)
    ax.set(xlim=[np.min(X) - 1, np.max(X) + 1], ylim=[np.min(Y) - 1, np.max(Y) + 1], xlabel='X', ylabel='Y')
    line, = ax.plot([], [], color="red")
    param = [[a, b, 0]]
    for i in range(n_iter):
        h_a = np.sum(-2 * (Y - a * X - b) * X)
        h_b = np.sum(-2 * (Y - a * X - b))
        if not (abs(h_a) <= epsilon and abs(h_b) <= epsilon):
            a = a - alpha * h_a
            b = b - alpha * h_b
        if (i + 1) % 10000 == 0 or 1 <= (i + 1) <= 10:
            param.append([a, b, i + 1])
    line_x = np.linspace(np.min(X), np.max(X), 100)
    def update(frame):
        text.set_text(f'Iterations: {param[frame][2]}')
        line_y = param[frame][0] * line_x + param[frame][1]
        line.set_data(line_x, line_y)
        return line


    ani = animation.FuncAnimation(fig, update, frames=len(param), interval=150)
    return ani

def stochastic_gradient_descent_lr_animation(X, Y, a, b, alpha, n_iter, epsilon = 1e-06):
    fig, ax = plt.subplots()
    text = ax.text(0.6, 0.05, f'Iterations: {0}',
                   transform=ax.transAxes,
                   verticalalignment='top')
    ax.scatter(X, Y, c="b", s=30)
    ax.set(xlim=[np.min(X) - 1, np.max(X) + 1], ylim=[np.min(Y) - 1, np.max(Y) + 1], xlabel='X', ylabel='Y')
    line, = ax.plot([], [], color="red")
    param = [[a, b, 0]]
    rng = np.random.default_rng(0)
    for i in range(n_iter):
        random_index = rng.integers(0, len(X))
        h_a = np.sum(-2 * (Y[random_index] - a * X[random_index] - b) * X[random_index])
        h_b = np.sum(-2 * (Y[random_index] - a * X[random_index] - b))
        if not (abs(h_a) <= epsilon and abs(h_b) <= epsilon):
            a = a - alpha * h_a
            b = b - alpha * h_b

        if (i + 1) % 10000 == 0 or 1 <= (i + 1) <= 10:
            param.append([a, b, i + 1])

    line_x = np.linspace(np.min(X), np.max(X), 100)
    def update(frame):
        text.set_text(f'Iterations: {param[frame][2]}')
        line_y = param[frame][0] * line_x + param[frame][1]
        line.set_data(line_x, line_y)
        return line

    ani = animation.FuncAnimation(fig, update, frames=len(param), interval=150)
    return ani

def minibatch_stochastic_gradient_descent_lr_animation(X, Y, a, b, alpha, n_iter, batch_size, epsilon = 1e-06):
    fig, ax = plt.subplots()
    text = ax.text(0.6, 0.05, f'Iterations: {0}',
                   transform=ax.transAxes,
                   verticalalignment='top')
    ax.scatter(X, Y, c="b", s=30)
    ax.set(xlim=[np.min(X) - 1, np.max(X) + 1], ylim=[np.min(Y) - 1, np.max(Y) + 1], xlabel='X', ylabel='Y')
    line, = ax.plot([], [], color="red")
    param = [[a, b, 0]]
    rng = np.random.default_rng(0)
    indices = rng.choice(X.shape[0], size=len(X), replace=False)
    shuffled_X = X[indices]
    shuffled_Y = Y[indices]
    start = 0
    end = batch_size
    for i in range(n_iter):
        if end >= len(X):
            indices = rng.choice(X.shape[0], size=len(X), replace=False)
            shuffled_X = X[indices]
            shuffled_Y = Y[indices]
            start = 0
            end = batch_size

        batch_X = shuffled_X[start:end]
        batch_Y = shuffled_Y[start:end]
        h_a = np.sum(-2 * (batch_Y - a * batch_X - b) * batch_X)
        h_b = np.sum(-2 * (batch_Y - a * batch_X - b))
        if not (abs(h_a) <= epsilon and abs(h_b) <= epsilon):
            a = a - alpha * h_a
            b = b - alpha * h_b
            start += batch_size
            end += batch_size

        if (i + 1) % 10000 == 0 or 1 <= (i + 1) <= 10:
            param.append([a, b, i + 1])

    line_x = np.linspace(np.min(X), np.max(X), 100)
    def update(frame):
        text.set_text(f'Iterations: {param[frame][2]}')
        line_y = param[frame][0] * line_x + param[frame][1]
        line.set_data(line_x, line_y)
        return line

    ani = animation.FuncAnimation(fig, update, frames=len(param), interval=150)
    return ani

