import numpy as np
import matplotlib.pyplot as plt

# Sigmoide
def g(z):
    return 1 / (1 + np.exp(-z))

def plot_frontier(Theta, X, Y):
	plt.figure()

	pos = np.where(Y == 1);
	neg = np.where(Y == 0);
	pts_pos = plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
	pts_neg = plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')

	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
	
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
						   np.linspace(x2_min, x2_max))

	h = g(np.c_[np.ones((xx1.ravel().shape[0], 1)),
				xx1.ravel(),
				xx2.ravel()].dot(Theta))

	h = h.reshape(xx1.shape)

	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
	plt.legend((pts_pos, pts_neg), ('Admitted', 'Not Admitted'))
	plt.show()

def plot_frontier_3D(Theta, X, Y):
	fig = plt.figure()

	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
	
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
						   np.linspace(x2_min, x2_max))

	h = g(np.c_[np.ones((xx1.ravel().shape[0], 1)),
				xx1.ravel(),
				xx2.ravel()].dot(Theta))

	h = h.reshape(xx1.shape)

	ax = Axes3D(fig)
	surf = ax.plot_surface(xx1, xx2, h, cmap=cm.Spectral, linewidths=0)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

def accuracy_percentage(Theta, X, Y):
	Z = g(X@Theta)
	pos = np.where(Y == 1)
	neg = np.where(Y == 0)
	z_pos = np.where(Z >= 0.5)
	z_neg = np.where(Z < 0.5)

	pos_perc = np.shape(z_pos)[1] * 100 / np.shape(pos)[1]
	neg_perc = np.shape(z_neg)[1] * 100 / np.shape(neg)[1]	
	if pos_perc > 100: pos_perc = 200 - pos_perc
	if neg_perc > 100: neg_perc = 200 - neg_perc
	
	print("{}% of positives accuracy.".format(pos_perc))
	print("{}% of negatives accuracy.".format(neg_perc))

def plot_decisionboundary(Theta, X, Y, poly):
    plt.figure()

    pos = np.where(Y == 1);
    neg = np.where(Y == 0);
    pts_pos = plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
    pts_neg = plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), 
                           np.linspace(x2_min, x2_max))

    h = g(poly.fit_transform(np.c_[xx1.ravel(),
                                   xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.show()

def plot_decisionboundary_3D(Theta, X, Y, poly):
    fig = plt.figure()

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), 
                           np.linspace(x2_min, x2_max))

    h = g(poly.fit_transform(np.c_[xx1.ravel(),
                                   xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)

    ax = Axes3D(fig)
    surf = ax.plot_surface(xx1, xx2, h, cmap=cm.Spectral, linewidths=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()