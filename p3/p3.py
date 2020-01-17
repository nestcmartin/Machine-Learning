import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def load_file(filename):
    data = loadmat(filename)
    return data['X'], data['y']

def load_neural_data(filename):
    weights = loadmat(filename)
    return weights['Theta1'], weights['Theta2']

def g(Z):
    return 1 / (1 + np.exp(-Z))

def coste_reg(Theta, X, Y, Lambda):
    m = np.shape(X)[0]
    cost = -(((np.log(g(X@Theta))).T@Y) + ((np.log(1-g(X@Theta))).T@(1 - Y)))/m
    reg = (Lambda/(2*m))*sum(Theta**2)
    return cost + reg

def gradiente_reg(Theta, X, Y, Lambda):
    m = np.shape(X)[0]
    grad = (X.T@(g(X@Theta)-Y))/m
    aux = np.copy(Theta)
    aux[0] = 0.0
    reg =  (Lambda/m)*aux
    return grad + reg

def oneVsAll(X, y, num_etiquetas, reg):
    Theta = np.zeros(X.shape[1])
    Thetas = np.zeros([num_etiquetas, X.shape[1]])
    
    y_i = np.where(y == 10, 1, 0)
    Thetas[0] = opt.fmin_tnc(func=coste_reg, x0=Theta, fprime=gradiente_reg, args=(X, y_i, reg))[0]
    
    for i in range(1, num_etiquetas):
        y_i = np.where(y == i, 1, 0)
        Thetas[i] = opt.fmin_tnc(func=coste_reg, x0=Theta, fprime=gradiente_reg, args=(X, y_i, reg))[0]
    
    return Thetas

def clasificador(muestra, num_etiquetas, Thetas):
    
    sigmoides = np.zeros(num_etiquetas)
    
    for i in range(num_etiquetas):
        sigmoides[i] = g(np.dot(muestra, Thetas[i, :]))

    return np.argmax(sigmoides)

def porcentajeCoincidencias(a, b):
    comp = a == b
    return 100 * sum(map(lambda comp : comp == True, comp)) / comp.shape

def entrenamiento(X, y, num_etiquetas, reg):
    Thetas = oneVsAll(X, y, num_etiquetas, reg)    
    y_ = np.zeros(X.shape[0])
    y = np.where(y == 10, 0, y)    
    for i in range(X.shape[0]):
        y_[i] = clasificador(X[i, :], num_etiquetas, Thetas)        
    return porcentajeCoincidencias(y_, y)

def propagacion(X, theta1, theta2):
    a1 = np.hstack([np.ones([X.shape[0], 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([X.shape[0], 1]), g(z2)])
    z3 = np.dot(a2, theta2.T)
    return g(z3)

def predictor(sigmoides) :
    y = np.zeros(sigmoides.shape[0])
    for i in range(sigmoides.shape[0]):
        y[i] = np.argmax(sigmoides[i, :]) + 1 
    return y

def entrenamiento_neural(X, y, theta1, theta2) :
    sigmoides = propagacion(X, theta1, theta2)
    y_ = predictor(sigmoides)
    return porcentajeCoincidencias(y_, y)

def first_test():
    X, y = load_file('ex3data1.mat')
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')

def second_test():
    Lambda = 0.1
    num_etiquetas = 10
    
    X, y = load_file('ex3data1.mat')
    X_ones = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    Y_ravel = np.ravel(y)
    
    percentage = entrenamiento(X_ones, Y_ravel, num_etiquetas, Lambda)
    print("Porcentaje reg logística: ", percentage)

def third_test():
    X, y = load_file('ex3data1.mat')
    Y_ravel = np.ravel(y)
    
    theta1, theta2 = load_neural_data('ex3weights.mat')
    percentage = entrenamiento_neural(X, Y_ravel, theta1, theta2)
    print("Porcentaje red neuronal: ", percentage)

first_test()
second_test()
third_test()