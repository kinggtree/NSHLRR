import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.linalg import norm, svd
from scipy.sparse import diags
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from Lh import compute_Lh


def svt(A, epsilon):
    U, s, VT = svd(A, full_matrices=False)
    s = np.maximum(s - epsilon, 0)
    return U @ diags(s) @ VT

def soft_thresholding(x, epsilon):
    return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)


def clustering_optimization(Y, num_neighbors, lambda_=0.02, beta=1.0, gamma=0.1, mu=1e-6, mu_max=1e6, epsilon1=1e-6, epsilon2=1e-2):
    n = Y.shape[1]

    Lh = compute_Lh(Y.T, num_neighbors)
    Lh_norm = norm(Lh, 2)

    Z = J = M2 = np.zeros((n, n))
    E = np.zeros(Y.shape)
    M1 = np.zeros(Y.shape)

    rho = mu
    while True:
        Z_old = Z.copy()
        J_old = J.copy()
        E_old = E.copy()

        # Update Z
        gradient_Z = beta * (Z @ (Lh.T + Lh)) + mu * (Z - J + M2/mu) + mu * Y.T @ (Y @ Z - Y + E - M1/mu)
        eta1 = 2 * beta * Lh_norm + mu * (1 + norm(Y, 2)**2)
        Z = svt(Z - gradient_Z / eta1, 1/eta1)

        # Update E
        E = soft_thresholding(Y - Y @ Z + M1/mu, gamma/mu)

        # Update J
        J = np.maximum(soft_thresholding(Z + M2/mu, lambda_/mu), 0)

        # Update M1 and M2
        M1 = M1 + mu * (Y - Y @ Z - E)
        M2 = M2 + mu * (Z - J)

        # Update mu
        if max(eta1 * norm(Z - Z_old, 'fro'), mu * norm(J - J_old, 'fro'), mu * norm(E - E_old, 'fro')) <= epsilon2:
            rho = mu
        else:
            rho = 1
        mu = min(mu_max, rho * mu)

        # Check convergence
        if norm(Y - Y @ Z - E, 'fro') / norm(Y, 'fro') < epsilon1 and max(eta1 * norm(Z - Z_old, 'fro'), mu * norm(J - J_old, 'fro'), mu * norm(E - E_old, 'fro')) < epsilon2:
            break

    return Z, E

def moon():
    # 生成半月形数据
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    # 平移和旋转数据
    X[y == 0, :] = X[y == 0, :] + [0.5, -0.25]
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = X @ rotation_matrix

    # 假设labels是你的聚类结果
    num_neighbors = 10
    Z, E = clustering_optimization(X, num_neighbors)

    # 基于Z的谱聚类
    spectral_clustering = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="discretize")
    labels = spectral_clustering.fit_predict(np.abs(Z))

    # 绘制聚类后的数据
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

moon()