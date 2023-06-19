import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from scipy.sparse import diags

# 设置参数
lambda_ = 0.1
gamma = 0.1
beta = 0.1
num_neighbors = 5

def get_hyperedges(Y, num_neighbors):
    # 使用 sklearn 的 NearestNeighbors 找到每个点的 K 个最近邻
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(Y)
    distances, indices = nbrs.kneighbors(Y)

    # 生成超边
    hyperedges = []
    for idx_list in indices:
        hyperedge = set(idx_list)
        hyperedges.append(hyperedge)

    return hyperedges


def soft_thresholding_operator(x, epsilon):
    return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)

def singular_value_thresholding_operator(A, epsilon):
    U, s, V = np.linalg.svd(A, full_matrices=False)
    return U @ np.diag(soft_thresholding_operator(s, epsilon)) @ V

def get_neighbors(data, num_neighbors):
    nbrs = NearestNeighbors(n_neighbors=num_neighbors+1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices  # 返回所有数据点最近的 num_neighbors 个点的索引



def compute_H(Y, num_neighbors):
    n, d = Y.shape

    hyperedges = get_hyperedges(Y, num_neighbors)
    m = len(hyperedges)

    H = np.zeros((n, m))
    for e in range(m):
        # 每一个超边是一个顶点的集合
        for v in hyperedges[e]:
            H[v, e] = 1

    return H

def compute_Lh(Y, epsilon):
    Y=Y.T
    n, d = Y.shape
    W_epsilon = np.zeros((n, n))

    # 计算所有数据点的近邻
    all_neighbors = get_neighbors(Y, num_neighbors)

    # 更新 W_epsilon
    for i in range(n):
        neighbors = all_neighbors[i, 1:]  # 选择出第 i 个数据点的近邻
        for j in neighbors:
            W_epsilon[i, j] = np.exp(-np.linalg.norm(Y[i] - Y[j])**2 / epsilon)
    
    Dv = diags(W_epsilon.sum(axis=1))
    De = diags(W_epsilon.sum(axis=0))
    De_inv = diags(1.0 / W_epsilon.sum(axis=0))

    # 需要 H 矩阵的定义和计算
    H = compute_H(Y, num_neighbors)

    Lh = Dv - H @ W_epsilon @ De_inv @ H.T


    return Lh


def ladmap(Y, lambda_, beta, gamma, k_nearest, max_iter=100, tol=1e-6):
    # Initialization
    n = Y.shape[1]
    Lh=compute_Lh(Y, 0.1)
    Z = np.zeros((n, n))
    E = np.zeros(Y.shape)
    J = np.zeros((n, n))
    M1 = np.zeros(Y.shape)
    M2 = np.zeros((n, n))
    mu = 0.02
    rho = 1.0
    mu_max = 1e6
    epsilon1 = 1e-6
    epsilon2 = 1e-2

    # Iteration
    for _ in range(max_iter):
        Z_old = Z.copy()
        E_old = E.copy()
        J_old = J.copy()

        # Update Z
        gradient_Z = beta * (Z @ Lh.T + Z @ Lh) + mu * (Z - J + M2 / mu) + mu * Y.T @ (Y @ Z - Y + E - M1 / mu)
        eta1 = 2 * beta * np.linalg.norm(Lh, 'fro')**2 + mu * (1 + np.linalg.norm(Y, 'fro')**2)
        Z = singular_value_thresholding_operator(Z - gradient_Z / eta1, 1 / eta1)

        # Update E
        E = soft_thresholding_operator(Y - Y @ Z + M1 / mu, gamma / mu)

        # Update J
        J = np.maximum(soft_thresholding_operator(Z + M2 / mu, lambda_ / mu), 0)

        print(Y.shape)    # 输出 Y 的形状
        temp = Y @ Z      # 计算 Y @ Z
        print(temp.shape) # 输出 Y @ Z 的形状
        temp = Y - temp   # 计算 Y - (Y @ Z)
        print(temp.shape) # 输出 Y - (Y @ Z) 的形状
        temp = temp - E   # 计算 Y - (Y @ Z) - E
        print("E",temp.shape) # 输出 Y - (Y @ Z) - E 的形状

        # Update Lagrange multipliers M1 and M2
        M1 += mu * (Y - Y @ Z - E)
        M2 += mu * (Z - J)

        # Update mu
        mu = min(mu_max, rho * mu)

        # Check convergence
        if np.linalg.norm(Y - Y @ Z - E, 'fro') / np.linalg.norm(Y, 'fro') < epsilon1 and \
           max(eta1 * np.linalg.norm(Z - Z_old, 'fro'), mu * np.linalg.norm(J - J_old, 'fro'), mu * np.linalg.norm(E - E_old, 'fro')) < epsilon2:
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
    Z,E = ladmap(X.T, lambda_=0.02, beta=1.0, gamma=5.0, k_nearest=10)

    # 使用谱聚类将表示系数矩阵转换为聚类标签
    clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(Z)
    labels = clustering.labels_

    # 绘制聚类后的数据
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()


moon()
