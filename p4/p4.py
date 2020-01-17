import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

def ht(theta1, theta2, X):
    m = X.shape[0]

    z2 = X @ theta1.T # a1 = X
    a2 = g(z2)
    a2_ones = np.hstack([np.ones([m, 1]), a2])
    z3 = a2_ones @ theta2.T
    a3 = g(z3) # a3 = ht(X)
    return a3 

def y_onehot(y, X, num_etiquetas):
    m = X.shape[0]
    y = y - 1

    y_onehot = np.zeros((m, num_etiquetas))
    for i in range(m):
        y_onehot[i][y[i]] = 1

    return y_onehot

# coste sin regularizar
def cost(theta1, theta2, num_etiquetas, X, y, reg):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    h = ht(theta1, theta2, X)

    cost = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum()
    return cost / m

# coste regularizado
def reg_cost(theta1, theta2, num_etiquetas, X, y, reg):
    m = X.shape[0]

    c_reg = cost(theta1, theta2, num_etiquetas, X, y, reg) + ((float(reg) / (2 * m)) 
            * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))) 
    
    return c_reg

# backprop devuelve el coste y el gradiente de una red neuronal de dos capas
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1 )))

    c = reg_cost(theta1, theta2, num_etiquetas, X, y, reg)
    print(c)

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

	#Lambda = 1.0
	#Theta = np.zeros(n)
	#oneVsAll(X_ones, Y_ravel, 10, Lambda)

part_one()