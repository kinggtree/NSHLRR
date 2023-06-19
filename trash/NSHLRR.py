import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 设置参数
lambda_ = 0.1
gamma = 0.1
beta = 0.1
num_neighbors = 5


# 定义我们的优化目标函数和约束条件
def objective(ZE, n, d, lambda_, gamma, beta, Y, W):
    Z = ZE[:n*d].reshape(n, d)
    E = ZE[n*d:].reshape(n, d)
    
    term1 = np.linalg.norm(Z, 'nuc')  # nuclear norm of Z
    term2 = lambda_ * np.linalg.norm(Z, 1)  # l1 norm of Z
    term3 = gamma * np.linalg.norm(E, 1)  # l1 norm of E
    
    term4 = 0
    for e in range(n):
        for i in range(d):
            for j in range(d):
                if i != j:
                    term4 += np.linalg.norm(Z[:, i] - Z[:, j])**2 * W[e] / d
    
    term4 *= beta
    
    return term1 + term2 + term3 + term4

def constraint(ZE, n, d, Y):
    Z = ZE[:n*d].reshape(n, d)
    E = ZE[n*d:].reshape(n, d)
    return Y - np.dot(Y, Z) - E

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

def get_neighbors(data, point_id, num_neighbors):
    nbrs = NearestNeighbors(n_neighbors=num_neighbors+1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices[point_id, 1:]  # 返回最近的 num_neighbors 个点的索引，我们不包括点自身，所以使用了1:来进行切片

def compute_weight_matrix(X, num_neighbors):
    n = X.shape[0]
    W = np.zeros((n, n))
    
    for i in range(n):
        neighbors_i = get_neighbors(X, i, num_neighbors)
        for j in range(n):
            neighbors_j = get_neighbors(X, j, num_neighbors)
            
            if i in neighbors_j or j in neighbors_i:
                W[i, j] = W[j, i] = 1
    
    return W


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

def compute_Z(Y, W):
  n, d = Y.shape

  # 初始化 Z 和 E
  Z0 = np.random.rand(n, d)
  E0 = np.random.rand(n, d)

  # 将 Z 和 E 拼接成一个向量，因为 scipy 的 minimize 函数只接受向量输入
  ZE0 = np.concatenate((Z0.ravel(), E0.ravel()))

  # 定义约束条件
  cons = {'type': 'eq', 'fun': constraint, 'args': (n, d, Y)}

  # 进行优化
  res = minimize(objective, ZE0, args=(n, d, lambda_, gamma, beta, Y, W), constraints=cons, method='SLSQP')

  # 提取结果
  ZE_opt = res.x
  Z_opt = ZE_opt[:n*d].reshape(n, d)
  E_opt = ZE_opt[n*d:].reshape(n, d)

  return Z_opt

def compute_Lh(Y, epsilon):
    n, d = Y.shape
    W_epsilon = np.zeros((n, n))

    # 计算近邻并且更新 W_epsilon
    for i in range(n):
        # 这里需要一个获取近邻的函数
        neighbors = get_neighbors(Y, i, num_neighbors)
        for j in neighbors:
            W_epsilon[i, j] = np.exp(-np.linalg.norm(Y[i] - Y[j])**2 / epsilon)
    
    Dv = diags(W_epsilon.sum(axis=1))
    De = diags(W_epsilon.sum(axis=0))
    De_inv = diags(1.0 / W_epsilon.sum(axis=0))

    # 需要 H 矩阵的定义和计算
    H = compute_H(Y,num_neighbors)

    Lh = Dv - H @ W_epsilon @ De_inv @ H.T
    return Lh



def moon():
    # 生成半月形数据
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    # 平移和旋转数据
    X[y == 0, :] = X[y == 0, :] + [0.5, -0.25]
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = X @ rotation_matrix

    # 计算近邻数和 epsilon
    num_neighbors = 5  # 这个数值可能需要调整
    epsilon = 0.1  # 这个数值可能需要调整
    
    # 计算 Lh
    Lh = compute_Lh(X, epsilon)
    print("Lh")
    
    # 计算 W
    W = compute_weight_matrix(X, num_neighbors)
    print("W")

    # 计算 Z
    Z = compute_Z(X, W.T)
    print("Z")

    # 使用KMeans计算聚类标签
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(Z)
    labels = kmeans.labels_

    # 绘制聚类后的数据
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

    # 这里假设 Z 是你的聚类结果，你可能需要对 Z 进行进一步的处理以得到正确的聚类结果
    labels = Z

    # 绘制聚类后的数据
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

moon()

