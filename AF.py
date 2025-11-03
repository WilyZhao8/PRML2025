import numpy as np
import matplotlib.pyplot as plt

inputs = np.arange(-10, 10, 0.1)

# --- Sigmoid ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- ReLU ---
def ReLU(x):
    return np.maximum(0.05*x, x)

def ReLU_deriv(x):
    return np.where(x > 0, 1, 0.05)

# --- Tanh ---
def Tanh(x):
    return np.tanh(x)

def Tanh_deriv(x):
    y = np.tanh(x)
    return 1 - y ** 2

# --- 绘图 ---
plt.figure(figsize=(10, 10))

plt.subplot(321)
plt.plot(inputs, sigmoid(inputs))
# plt.title("Sigmoid Function")
plt.grid(True)

plt.subplot(322)
plt.plot(inputs, sigmoid_deriv(inputs))
# plt.title("Sigmoid Derivative")
plt.grid(True)

plt.subplot(323)
plt.plot(inputs, ReLU(inputs))
# plt.title("LReLU Function")
plt.grid(True)

plt.subplot(324)
plt.plot(inputs, ReLU_deriv(inputs))
# plt.title("LReLU Derivative")
plt.grid(True)

plt.subplot(325)
plt.plot(inputs, Tanh(inputs))
# plt.title("Tanh Function")
plt.grid(True)

plt.subplot(326)
plt.plot(inputs, Tanh_deriv(inputs))
# plt.title("Tanh Derivative")
plt.grid(True)

plt.tight_layout()
plt.show()

