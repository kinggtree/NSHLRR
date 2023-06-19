import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from numpy.linalg import norm, svd
from sklearn.decomposition import PCA
from usps import usps
from Lh import compute_Lh
from usps import usps

def soft_thresholding_operator(x, epsilon):
    is_matrix = isinstance(x, np.matrix)
    if is_matrix:
        x = np.array(x)
    
    result = np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)

    if is_matrix:
        result = np.matrix(result)
    
    return result


def singular_value_thresholding_operator(A, epsilon):
    U, s, V = svd(A, full_matrices=False)
    return U @ np.diag(soft_thresholding_operator(s, epsilon)) @ V


def ladmap(Y, lambda_, beta, gamma, k_nearest, max_iter=100, tol=1e-6):
    print("Computing Ladmap...")
    # Initialization
    n = Y.shape[1]
    Lh = compute_Lh(Y,k_nearest)
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
    for ind in range(max_iter):
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

        # Update Lagrange multipliers M1 and M2
        M1 += mu * (Y - Y @ Z - E)
        M2 += mu * (Z - J)

        # Update mu
        if max(eta1 * norm(Z - Z_old, 'fro'), mu * norm(J - J_old, 'fro'), mu * norm(E - E_old, 'fro')) <= epsilon2:
            rho = mu
        else:
            rho = 1
        mu = min(mu_max, rho * mu)

        show_scale=100
        if ind%show_scale==0:
            print("current: ",ind/show_scale, "target: ", max_iter/show_scale)
            print("Judge1: ", norm(Y - Y @ Z - E, 'fro') / np.linalg.norm(Y, 'fro'), "<", epsilon1)
            print("Judge2: ",max(eta1 * np.linalg.norm(Z - Z_old, 'fro'), mu * np.linalg.norm(J - J_old, 'fro'), mu * np.linalg.norm(E - E_old, 'fro')), "<", epsilon2)
            print()

        # Check convergence
        if norm(Y - Y @ Z - E, 'fro') / np.linalg.norm(Y, 'fro') < epsilon1 and \
            max(eta1 * np.linalg.norm(Z - Z_old, 'fro'), mu * np.linalg.norm(J - J_old, 'fro'), mu * np.linalg.norm(E - E_old, 'fro')) < epsilon2:
            break

    print("Ladmap Done!")
    return Z, E

def moon():
    # 生成半月形数据
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    # 平移和旋转数据
    X[y == 0, :] = X[y == 0, :] + [0.5, -0.25]
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = X @ rotation_matrix

    #labels为聚类结果
    Z,E = ladmap(X.T, lambda_=0.02, beta=1.0, gamma=5.0, k_nearest=10, max_iter=2000)

    # 使用谱聚类将表示系数矩阵转换为聚类标签
    Z=np.array(Z)
    clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(Z)
    labels = clustering.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

def compute_usps():
    # Load the USPS dataset
    matrix, matrix_label=usps(2, 100)
    # Use PCA to reduce the dimensionality to 2
    pca = PCA(n_components=2)
    matrix_pca = pca.fit_transform(matrix)

    Z,E = ladmap(matrix.T, lambda_=0.02, beta=1.0, gamma=5.0, k_nearest=50, max_iter=100)

    # 使用谱聚类将表示系数矩阵转换为聚类标签
    Z=np.array(Z)
    clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(Z)
    labels = clustering.labels_

    # 创建一个包含两个子图的图像对象
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 在第一个子图上绘制原图像
    ax1.scatter(matrix_pca[:, 0], matrix_pca[:, 1], c=matrix_label)
    ax1.set_title('Original Data')

    # 在第二个子图上绘制聚类后的图像
    ax2.scatter(Z[:, 0], Z[:, 1], c=labels)
    ax2.set_title('Clustered Data')

    # 显示图像
    plt.show()

moon()
#compute_usps()

