import numpy as np
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist


def compute_Lh(Y, num_neighbors):
    n = Y.shape[1]
    W_epsilon = np.zeros((n, n))

    # 计算近邻并且更新 W_epsilon
    #for i in range(d):
        # 这里需要一个获取近邻的函数
    #    neighbors = get_neighbors(Y.T, i, num_neighbors)
    #    for j in neighbors:
    #        W_epsilon[i, j] = np.exp(-np.linalg.norm(Y[:, i] - Y[:, j])**2 / epsilon)
    # 计算k近邻图
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(Y.T)
    distances, indices = nbrs.kneighbors(Y.T)

    # 计算高斯核函数的带宽
    distances = pdist(Y.T, 'euclidean')
    sigma = np.median(distances)

    # 创建加权邻接矩阵
    n = Y.shape[1]
    W = np.zeros((n, n))
    for i in range(n):
        for j in indices[i]:
            W[i, j] = np.exp(-np.linalg.norm(Y[:, i] - Y[:, j])**2 / (2. * sigma**2))

    # 计算超边的权重矩阵W_e
    hyperedges=get_nearest_neighbor_mask(Y.T, num_neighbors)
    W_epsilon = np.diag([W[i, hyperedges[i]].mean() for i in range(n)])

    Dv = diags(W_epsilon.sum(axis=1))
    De = diags(W_epsilon.sum(axis=0))
    De_inv = diags(1.0 / W_epsilon.sum(axis=0))

    # 计算 H 矩阵
    H = compute_H(Y,num_neighbors)

    Lh = Dv - H @ W_epsilon @ De_inv @ H.T
    return Lh

def compute_H(Y, num_neighbors):
    n = Y.shape[1]

    hyperedges = get_hyperedges(Y.T, num_neighbors)

    H = np.zeros((n, n))
    for e in range(n):
        # 每一个超边是一个顶点的集合
        for v in hyperedges[e]:
            H[v, e] = 1

    return H

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

def get_nearest_neighbor_mask(Y, num_neighbors):
    # 使用 sklearn 的 NearestNeighbors 找到每个点的 K 个最近邻
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(Y)
    distances, indices = nbrs.kneighbors(Y)

    # 创建布尔数组
    bool_array = np.zeros((Y.shape[0], Y.shape[0]), dtype=bool)
    
    # 对于每个点，标记它的最近邻
    for idx, idx_list in enumerate(indices):
        bool_array[idx, idx_list] = True

    return bool_array