import numpy as np 
from pandas.io.parsers import read_csv

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    return valores.astype(float)

def gradient_descent(X, Y, a):
	m = np.shape(X)[0]
	n = np.shape(X)[1]
	Thetas = np.ndarray((1500, n))
	Theta = np.zeros(n)
	costes = np.ndarray(1500)
	for i in range(1500):
		Thetas[i] = gradient(X, Y, Theta, a)
		costes[i] = j(X, Y, Thetas[i])
	return Thetas, costes


def gradient(X, Y, Z, a):
	z_new = Z
	m = np.shape(X)[0]
	n = np.shape(X)[1]
	H = np.dot(X, Z)
	Aux = (H - Y)

	for i in range(n):
		Aux_i = Aux * X[:, i]
		z_new[i] -= (a / m) * Aux_i.sum()
	
	return z_new

def j(X, Y, Z):
	H = np.dot(X, Z)
	Aux = (H - Y) ** 2
	return Aux.sum() / (2 * len(X))

def normalize(X):
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	for ix, iy in np.ndindex(X.shape):
		X[ix, iy] = (X[ix,iy] - mu[iy]) / sigma[iy] 
	return X, mu, sigma

def normal_equation(X, Y):
	return np.linalg.inv(X.T @ X) @ X.T @ Y

def make_data(Z0_range, Z1_range, X, Y):
	step = 0.1
	z0 = np.arange(Z0_range[0], Z0_range[1], step)
	z1 = np.arange(Z1_range[0], Z1_range[1], step)
	Z0, Z1 = np.meshgrid(z0, z1)
	
	J = np.empty_like(Z0)
	for ix, iy in np.ndindex(Z0.shape):
		J[ix, iy] = j(X, Y, [Z0[ix, iy], Z1[ix, iy]])

	return [Z0, Z1, J]

def draw_data_surface(Z0, Z1, J):	
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(Z0, Z1, J, cmap=cm.coolwarm, linewidths=0)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

def draw_data_contour(Z0, Z1, J):
	plt.contour(Z0, Z1, J, np.logspace(-2, 3, 20), colors="blue")
	plt.show()

def part_one():
	data = carga_csv("ex1data1.csv")
	X = data[:, :-1]      
	Y = data[:, -1]

	m = np.shape(X)[0]	
	X_ones = np.hstack([np.ones([m, 1]), X])
	
	a = 0.01
	Thetas, costes = gradient_descent(X_ones, Y, a)

	x_sample = np.array([np.amin(X), np.amax(X)])
	x_aux = np.hstack([np.ones([2, 1]), x_sample.reshape(2, 1)])
	y_sample = np.dot(x_aux, Thetas[1499])

	plt.scatter(X, Y, color='red', marker='x')
	plt.plot(x_sample, y_sample, color='blue')
	plt.show()

	Z0, Z1, J = make_data([-10, 10], [-1, 4], X, Y)
	draw_data_contour(Z0, Z1, J)
	draw_data_surface(Z0, Z1, J)

def part_two_a():
	data = carga_csv("ex1data2.csv")
	X = data[:, :-1]      
	Y = data[:, -1]
	
	m = np.shape(X)[0]	
	X_norm, mu, sigma = normalize(X)
	X_norm = np.hstack([np.ones([m, 1]), X_norm])
	
	a = 0.01
	Thetas, costes = gradient_descent(X_norm, Y, a)
	print(Thetas[1499])
	print(np.dot([1.0, (1650.0 - mu[0]) / sigma[0], (3.0 - mu[1]) / sigma[1]], Thetas[1499]))
	
	a = 0.3
	Thetas, costes = gradient_descent(X_norm, Y, a)
	plt.plot(costes, color='red')
	
	a = 0.1
	Thetas, costes = gradient_descent(X_norm, Y, a)
	plt.plot(costes, color='yellow')
	
	a = 0.03
	Thetas, costes = gradient_descent(X_norm, Y, a)
	plt.plot(costes, color='green')
	
	a = 0.01
	Thetas, costes = gradient_descent(X_norm, Y, a)
	plt.plot(costes, color='blue')
	
	a = 0.003
	Thetas, costes = gradient_descent(X_norm, Y, a)
	plt.plot(costes, color='magenta')
	plt.show()

def part_two_b():
	data = carga_csv("ex1data2.csv")
	X = data[:, :-1]      
	Y = data[:, -1]
	
	m = np.shape(X)[0]
	n = np.shape(X)[1]

	X = np.hstack([np.ones([m, 1]), X])

	Theta = normal_equation(X, Y)
	print(Theta)
	print(np.dot([1.0, 1650.0, 3.0], Theta))

part_one()
part_two_a()
part_two_b()