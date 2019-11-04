import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def g(Z):
	return 1 / (1 + np.exp(-Z))

def cost(Theta, X, Y, Lambda):
	m = np.shape(X)[0]
	cost = -(((np.log(g(X@Theta))).T@Y) + ((np.log(1-g(X@Theta))).T@(1 - Y)))/m
	reg = (Lambda/(2*m))*sum(Theta**2)
	return cost + reg 

def gradient(Theta, X, Y, Lambda):
	m = np.shape(X)[0]
	grad = (X.T@(g(X@Theta)-Y))/m
	aux = np.copy(Theta)
	aux[0] = 0.0
	reg =  (Lambda/m)*aux
	return grad + reg

def oneVsAll(X, y, num_etiquetas, Lambda):
	n = np.shape(X)[1]	
	theta_opt = np.zeros([num_etiquetas, n])

	for i in range(num_etiquetas):
		y = np.where(y == i, 1, 0)
		Theta = np.zeros(n)
		result = opt.fmin_tnc(func=cost, x0=Theta, 
			fprime=gradient, args=(X, y, Lambda))
		theta_opt[i] = result[0]
	return theta_opt


#def draw_sample(X):
#	sample = np.random.choice(X.shape[0], 10)
#	plt.imshow(X[sample, :].reshape(-1, 20).T)
#	plt.axis('off')
#	plt.show()

def part_one():
	data = loadmat('ex3data1.mat')
	y = data['y']
	X = data['X']

	m = np.shape(X)[0]
	X_ones = np.hstack([np.ones([m, 1]), X])
	n = np.shape(X_ones)[1]
	Y_ravel = np.ravel(y)

	Lambda = 1.0
	Theta = np.zeros(n)
	oneVsAll(X_ones, Y_ravel, 10, Lambda)

part_one()