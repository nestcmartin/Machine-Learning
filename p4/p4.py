import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat

from matplotlib import cm
import matplotlib.pyplot as plt

import displayData as dd
import checkNNGradients as nn

import random


def show_selection(X):
    sel = np.random.permutation(m)
    sel = sel[:100]
    dd.displayData(X[sel, :])
    plt.show()

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def forward(X, theta1, theta2):
    a1 = np.hstack([np.ones([X.shape[0], 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([X.shape[0], 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3

def y_onehot(y, num_etiquetas):
    y = y - 1
    y_onehot = np.zeros((m, num_etiquetas))
    for i in range(m):
        y_onehot[i][y[i]] = 1
    return y_onehot

def cost(X, y, num_etiquetas):
    Y = y_onehot(y, num_etiquetas)
    a1, z2, a2, z3, a3 = forward(X, theta1, theta2)
    return (-Y * np.log(a3) - (1 - Y) * np.log(1 - a3)).sum() / m

def cost_reg(X, y, num_etiquetas, theta1, theta2, Lambda):
    unreg_term = cost(X, y, num_etiquetas)
    reg_term = (Lambda / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return unreg_term + reg_term    

def gradient(X, y, num_etiquetas, theta1, theta2):
    Y = y_onehot(y, num_etiquetas)
    a1, z2, a2, z3, a3 = forward(X, theta1, theta2)
    d3 = a3 - Y
    d2 = np.multiply((d3 @ theta2), sigmoid_derivative(z2)[:, :1])[:, 1:]
    theta1_grad = (1 / m) * (d2.T @ a1)
    theta2_grad = (1 / m) * (d3.T @ a2)
    return theta1_grad, theta2_grad

def gradient_reg(X, y, num_etiquetas, theta1, theta2, Lambda):
    theta1_grad, theta2_grad = gradient(X, y, num_etiquetas, theta1, theta2)
    theta1_grad[:, 1:] = ((Lambda / m) * theta1[:, 1:]) + theta1_grad[:, 1:]
    theta2_grad[:, 1:] = ((Lambda / m) * theta2[:, 1:]) + theta2_grad[:, 1:]
    return theta1_grad, theta2_grad

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1 )))
    J = cost_reg(X, y, num_etiquetas, theta1, theta2, Lambda)
    theta1_grad, theta2_grad = gradient_reg(X, y, num_etiquetas, theta1, theta2, Lambda)
    return (J, theta1_grad, theta2_grad)

def randomize_weights(L_in, L_out):
    #e_ini = math.sqrt(6) / math.sqrt(L_in + L_out)
    e_ini = 0.12
    weights = np.zeros((L_out, 1 + L_in))
    for i in range(L_out):
        for j in range(1 + L_in):
            weights[i,j] = random.uniform(-e_ini, e_ini)
    return weights

def optimize(backprop, params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, Lambda):
    fmin = opt.minimize(fun=backprop, x0=params_rn, args=(num_entradas, num_ocultas, num_etiquetas, X, y, Lambda), method='TNC', jac=True, options={'maxiter': 70})
    return fmin.x

def predict(X):
    predictions = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        predictions[i] = np.argmax(X[i, :]) + 1
    return predictions

def coincidence_percentage(a, b):
    comp = a == b
    return 100 * sum(map(lambda comp : comp == True, comp)) / comp.shape

def neural_training(X, y, weights1, weights2):
    sigmoids = forward(X, weights1, weights2)[4]
    predictions = predict(sigmoids)
    return coincidence_percentage(predictions, y)

#weights = loadmat('ex4weights.mat')
#theta1, theta2 = weights['Theta1'], weights['Theta2']
#params_rn = np.concatenate([theta1.reshape(-1), theta2.reshape(-1)])

data = loadmat('ex4data1.mat')
X, y = data['X'], data['y'].reshape(-1)
m = X.shape[0]

num_etiquetas = 10
num_entradas = 400
num_ocultas = 25
Lambda = 1.0

show_selection(X)
theta1 = randomize_weights(num_entradas, num_ocultas)
theta2 = randomize_weights(num_ocultas, num_etiquetas)
params_rn = np.concatenate([theta1.reshape(-1), theta2.reshape(-1)])

theta_opt = optimize(backprop, params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, Lambda)
theta1_opt = np.reshape(theta_opt[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
theta2_opt = np.reshape(theta_opt[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1 )))
    
percentage = neural_training(X, y, theta1_opt, theta2_opt)
print("Neural network success rate: ", percentage)