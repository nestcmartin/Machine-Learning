import random
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
import matplotlib.pyplot as plt

import displayData as dd
import checkNNGradients as nn

def show_selection(X):
    sel = np.random.permutation(m)
    sel = sel[:100]
    dd.displayData(X[sel, :])
    plt.show()

def convert_y(y, num_etiquetas):
    Y = np.empty((num_etiquetas, y.shape[0]), dtype=bool)
    for i in range(num_etiquetas):
        Y[i, :] = ((y[:, 0] + num_etiquetas - 1) % num_etiquetas == i)    
    Y = Y * 1
    return Y.T

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def feed_forward(X, theta1, theta2):
    a1 = np.hstack([np.ones([X.shape[0], 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([X.shape[0], 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3

def back_propagation(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, Lambda):
    # Unroll thetas (neural network params)
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    # Forward propogation (feed forward)
    A1, Z2, A2, Z3, A3 = feed_forward(X, theta1, theta2)

    # Cost function (without reg term)
    cost_unreg_term = (-Y * np.log(A3) - (1 - Y) * np.log(1 - A3)).sum() / m

    # Cost function (with reg term)
    cost_reg_term = (Lambda / (2 * m)) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    cost = cost_unreg_term + cost_reg_term

    # Numerical gradient (without reg term)
    Theta1_grad = np.zeros(np.shape(theta1))
    Theta2_grad = np.zeros(np.shape(theta2))
    D3 = A3 - Y
    D2 = np.dot(D3, theta2)
    D2 = D2 * (np.hstack([np.ones([Z2.shape[0], 1]), sigmoid_derivative(Z2)]))
    D2 = D2[:, 1:]
    Theta1_grad = Theta1_grad + np.dot(A1.T, D2).T
    Theta2_grad = Theta2_grad + np.dot(A2.T, D3).T

    # Numerical gradient (with reg term)
    Theta1_grad = Theta1_grad * (1 / m)
    Theta2_grad = Theta2_grad * (1 / m)
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (Lambda / m) * theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (Lambda / m) * theta2[:, 1:]
    grad = np.concatenate((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

    return (cost, grad)

def randomize_weights(L_in, L_out):
    epsilon = 0.12
    return np.random.uniform(-epsilon, epsilon, (L_out, 1 + L_in))

def optimize(backprop, params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, Lambda, num_iter):
    result = opt.minimize(fun=backprop, x0=params_rn,
    args=(num_entradas, num_ocultas, num_etiquetas, X, Y, Lambda),
    method='TNC', jac=True, options={'maxiter': num_iter})
    return result.x

def neural_training(y, out):
    max_i = np.argmax(out, axis = 1) + 1
    control = (y[:, 0] == max_i) 
    return 100 * np.size(np.where(control == True)) / y.shape[0]

data = loadmat('ex4data1.mat')
weights = loadmat('ex4weights.mat')
X, y = data['X'], data['y']
theta1, theta2 = weights['Theta1'], weights['Theta2']

Lambda = 1.0
num_iter = 70
m = X.shape[0]
num_entradas = 400
num_ocultas = 25
num_etiquetas = 10

show_selection(X)
theta1 = randomize_weights(theta1.shape[1] - 1, theta1.shape[0])
theta2 = randomize_weights(theta2.shape[1] - 1, theta2.shape[0])
params_rn = np.concatenate([theta1.reshape(-1), theta2.reshape(-1)])
theta_opt = optimize(back_propagation, params_rn, num_entradas, num_ocultas, num_etiquetas, X, convert_y(y, num_etiquetas), Lambda, num_iter)
theta1_opt = np.reshape(theta_opt[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
theta2_opt = np.reshape(theta_opt[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1 )))  
percentage = neural_training(y, feed_forward(X, theta1_opt, theta2_opt)[4])

print("Neural network success rate: {}%".format(percentage))
print(nn.checkNNGradients(back_propagation, Lambda))
# TODO: probar con distintas configuraciones para Lambda y num_iter