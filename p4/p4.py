import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import random

# Display functions ------------

def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0

    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)

def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)

# ------------------------------

# sigmoide de Z
def g(Z):
    return 1 / (1 + np.exp(-Z))

# derivada del sigmoide de Z
def derivative_g(Z):
    return np.multiply(g(Z), (1 - g(Z)))
    
def random_weights(L_in, L_out):
    weights = np.zeros((L_out, 1 + L_in))

    for i in range(L_out):
        for j in range(1 + L_in):
            weights[i, j] = random.uniform(-0.12, 0.12)

    return weights

def forward(theta1, theta2, X):
    m = X.shape[0]

    z2 = X @ theta1.T # a1 = X
    a2 = g(z2)
    a2_ones = np.hstack([np.ones([m, 1]), a2])
    z3 = a2_ones @ theta2.T
    a3 = g(z3) # a3 = ht(X)
    return X, z2, a2_ones, z3, a3 

def y_onehot(y, X, num_etiquetas):
    m = X.shape[0]
    y = y - 1

    y_onehot = np.zeros((m, num_etiquetas))
    for i in range(m):
        y_onehot[i][y[i]] = 1

    return y_onehot

# coste sin regularizar
def cost(theta1, theta2, num_etiquetas, X, y, h):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    cost = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum()
    return cost / m

# coste regularizado
def reg_cost(theta1, theta2, num_etiquetas, X, y, h, reg):
    m = X.shape[0]

    c_reg = cost(theta1, theta2, num_etiquetas, X, y, h) + ((float(reg) / (2 * m)) 
            * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))) 
    
    return c_reg

# deltas # TODO: QUE FUNCIONE 
def deltas(theta1, theta2, a1, z2, a2, z3, h, y):
    m = a1.shape[0] # a1 = X
    delta1, delta2 = np.zeros(theta1.shape), np.zeros(theta2.shape)
    z2_ones =  np.hstack([np.ones([m, 1]), z2])

    d3 = h - y
    aux = theta2.T @ d3.T
    d2 = aux @ derivative_g(z2_ones)
    
    print (a1.shape)
    delta1 += d2.T @ a1
    delta2 += d3.T @ a2

    return delta1 / m, delta2 / m

# backprop devuelve el coste y el gradiente de una red neuronal de dos capas
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1 )))

    a1, z2, a2, z3, h = forward(theta1, theta2, X)

    c = reg_cost(theta1, theta2, num_etiquetas, X, y, h, reg)
    print(c)

    delta1, delta2 = deltas(theta1, theta2, a1, z2, a2, z3, h, y)

def part_one():
    Lambda = 1.0 # reg

    data = loadmat('ex4data1.mat')
    y = data['y']
    X = data['X']
    #print(X.shape, y.shape)

    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    # Theta1 es de dimension 25 x 401
    # Theta2 es de dimension 10 x 26

    num_etiquetas = 10 # 0 es 10
    num_entradas = 400
    num_ocultas = 25
    
    m = np.shape(X)[0]
    X_ones = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X_ones)[1]
    Y_ravel = np.ravel(y)

    Y_oh = y_onehot(Y_ravel, X, num_etiquetas)

    params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    #print(params_rn.shape)

    backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X_ones, Y_oh, Lambda)



part_one()