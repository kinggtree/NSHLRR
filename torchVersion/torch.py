import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np

# Your custom module
from Lh import compute_Lh


# Change numpy specific code to PyTorch
def soft_thresholding_operator(x, epsilon):
    return torch.max(torch.abs(x) - epsilon, torch.tensor(0.0, device=x.device)) * torch.sign(x)


# Change numpy specific code to PyTorch
def singular_value_thresholding_operator(A, epsilon):
    U, s, V = torch.svd(A)
    return U @ torch.diag(soft_thresholding_operator(s, epsilon)) @ V.t()


def ladmap(Y, lambda_, beta, gamma, k_nearest, max_iter=3500, tol=1e-6):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cast numpy array to PyTorch tensor and send to appropriate device
    Y = torch.from_numpy(Y).float().to(device)

    n = Y.shape[1]

    # You have to adjust this function for PyTorch
    Lh = compute_Lh(Y, k_nearest)
    Z = torch.zeros((n, n), device=device)
    E = torch.zeros(Y.shape, device=device)
    J = torch.zeros((n, n), device=device)
    M1 = torch.zeros(Y.shape, device=device)
    M2 = torch.zeros((n, n), device=device)

    # Same code but using PyTorch syntax
    mu = 0.02
    rho = 1.0
    mu_max = 1e6
    epsilon1 = 1e-6
    epsilon2 = 1e-2

    for ind in range(max_iter):
        Z_old = Z.clone()
        E_old = E.clone()
        J_old = J.clone()

        # Update Z
        gradient_Z = beta * (Z @ Lh.t() + Z @ Lh) + mu * (Z - J + M2 / mu) + mu * Y.t() @ (Y @ Z - Y + E - M1 / mu)
        eta1 = 2 * beta * torch.norm(Lh, 'fro')**2 + mu * (1 + torch.norm(Y, 'fro')**2)
        Z = singular_value_thresholding_operator(Z - gradient_Z / eta1, 1 / eta1)

        # Update E
        E = soft_thresholding_operator(Y - Y @ Z + M1 / mu, gamma / mu)

        # Update J
        J = torch.max(soft_thresholding_operator(Z + M2 / mu, lambda_ / mu), torch.tensor(0., device=device))

        # Update Lagrange multipliers M1 and M2
        M1 += mu * (Y - Y @ Z - E)
        M2 += mu * (Z - J)

        # Update mu
        if max(eta1 * torch.norm(Z - Z_old, 'fro'), mu * torch.norm(J - J_old, 'fro'), mu * torch.norm(E - E_old, 'fro')) <= epsilon2:
            rho = mu
        else:
            rho = 1
        mu = min(mu_max, rho * mu)

        if ind%1000==0:
            print("current: ",ind/1000, "target: ", max_iter/1000)
            print("Judge1: ", torch.norm(Y - Y @ Z - E, 'fro') / torch.norm(Y, 'fro'), "<", epsilon1)
            print("Judge2: ",max(eta1 * torch.norm(Z - Z_old, 'fro'), mu * torch.norm(J - J_old, 'fro'), mu * torch.norm(E - E_old, 'fro')), "<", epsilon2)
            print()

        # Check convergence
        if torch.norm(Y - Y @ Z - E, 'fro') / torch.norm(Y, 'fro') < epsilon1 and \
            max(eta1 * torch.norm(Z - Z_old, 'fro'), mu * torch.norm(J - J_old, 'fro'), mu * torch.norm(E - E_old, 'fro')) < epsilon2:
            break

    return Z.cpu().numpy(), E.cpu().numpy()  # convert the results back to numpy arrays if you prefer working with numpy


def moon():
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
    X[_ == 0, :] = X[_ == 0, :] + [0.5, -0.25]
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = X @ rotation_matrix

    Z, E = ladmap(X.T, lambda_=0.02, beta=1.0, gamma=5.0, k_nearest=10)
    Z = np.array(Z)

    clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(Z)
    labels = clustering.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()


moon()
