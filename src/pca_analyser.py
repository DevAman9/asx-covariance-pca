
import numpy as np

from risk_engine import compute_ledoit_wolf, compute_ewma_covariance
from returns_calculator import calculate_log_returns
from data_loader import download_price_data


def compute_pca(cov_matrix, n_days, n_assets):

    # first we get the eigen values and eigen vectors of our covariance matrix. Initially we would have 5 of each. but we will filter them out later using
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # reversing the order, so largest values are frist. and the array is in descending order.
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]


    # this is our Marchenko-Pastur threshold. basically it gives us a value to use as a filter threshold to differentiate important signals from statistical noise.
    sigma_squared = np.mean(np.diag(cov_matrix))


    ratio = n_assets / n_days

    # we are using this formula (λ_max = σ² · (1 + √(n_assets/n_days))²)
    lambda_max = sigma_squared * (1 + np.sqrt(ratio)) ** 2

    filtered_eigenvalues = eigenvalues.copy()

    for i , value in enumerate(filtered_eigenvalues):
        if value < lambda_max:
            filtered_eigenvalues[i] = 0

    cleaned_cov = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

    return eigenvalues, eigenvectors, filtered_eigenvalues, cleaned_cov


def explained_variance_ratio(eigenvalues):
    total = sum(eigenvalues)
    ratios = eigenvalues / total
    return ratios

if __name__ == "__main__":
    tickers = ["CBA.AX", "BHP.AX", "WES.AX", "ANZ.AX", "RIO.AX"]

    close_prices = download_price_data(tickers,  "2019-01-01" , "2024-01-01")

    returns =  calculate_log_returns(close_prices)
    cov_matrix = compute_ewma_covariance(returns)
    lp_cov_matrix = compute_ledoit_wolf(returns)

    n_days, n_assets = returns.shape

    eigenvalues, eigenvectors, filtered_eigenvalues, cleaned_cov = compute_pca(cov_matrix, n_days, n_assets)

    eigen_ratio = explained_variance_ratio(eigenvalues)

    print(f"Cleaned covariance matrix after PCA analysis and filtering using Marchenko pestur: \n {cleaned_cov}")



    # print(f"EWMA Covariance Matrix dimension: {cov_matrix.shape}")
    # print(cov_matrix)
    # print("\n\n")
    #
    # print(f"Ledoit Wolf stabilised Covariance Matrix dimension: {lp_cov_matrix.shape}")
    # print(lp_cov_matrix)