import math
import time
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def f(x):
	return np.sin(x)

def integrate_mc(f, a, b, num_points=100):

	tic = time.time()

	x = np.linspace(a, b, num_points)
	y = f(x)
	
	max_f = max(y)
	min_f = 0
	
	random_x = np.random.uniform(a, b, num_points)
	random_y = np.random.uniform(min_f, max_f, num_points)
	f_random_x = f(random_x)
	
	#plt.scatter(random_x, random_y, 'x', c='red')
	#plt.plot(eje_x, eje_y, '-')
	#plt.show()

	below = sum(random_y < f_random_x)
	area = abs(b - a) * abs(max_f - min_f)
	area *= (below / num_points)

	toc = time.time()

	return area, 1000 * (toc - tic)

def integrate_mc_slow(f, a, b, N=10000):
    
    tic = time.time()
    x = np.linspace(a, b, N)
    y = f(x)

    M = np.amax(y)
    
    x_sample = np.random.uniform(a, b, N)
    y_sample = np.random.uniform(0, M, N)

    #plt.scatter(x_sample, y_sample, 'x', c='red')
	#plt.plot(x, y, '-')
	#plt.show()
   
    num_points_below = 0
    for i in range(N):
        if f(x_sample[i] >= y_sample[i]):
            num_points_below += 1
    
    rect_area = (b - a) * M
    I = (num_points_below / N) * rect_area
    
    toc = time.time()
    return I, 1000 * (toc - tic)

def compara_tiempos():
    points = np.arange(100, 10000, 100)
    times_mc_slow = []
    times_mc = []

    for num_points in points:
        times_mc += [integrate_mc(f, 0, math.pi, num_points)[1]]
        times_mc_slow += [integrate_mc_slow(f, 0, math.pi, num_points)[1]]

    plt.scatter(points, times_mc_slow, c='blue', label='loop')
    plt.scatter(points, times_mc, c='red', label='vector')
    plt.show()

compara_tiempos();