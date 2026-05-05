
import numpy as np
import pandas as pd
import sklearn.covariance

from returns_calculator import calculate_log_returns
from data_loader import download_price_data
from sklearn.covariance import LedoitWolf

from sklearn.covariance import LedoitWolf

#this code is still in progress. it works. but i need to complete a few testings and checks. that's why its got loose comments and print statements all over. i will make sure to remove them before finalising my project.

def compute_ewma_covariance(returns, lambda_ = 0.94):

    # before and after. went from pandas dataframe to numpy array.
    # pandas dataframe is very good to present the data in a tabular format. but for calculation and analysis, numpy is much more efficient.

    data = returns.values


    n_days , n_assets = data.shape

    # now i will just make a basic covariance matrix using the first 30 rows just to mark a starting point or as a standard average. we would also have to transpose the matrix since we need the assets as rows unlike currently where they are columns.
    cov_matrix = np.cov(data[0:30].T)

    print(cov_matrix.shape)

    # testing the wrong way. what if we use the EWMA formula from day 1. what will happen. just curious
    # for t in range(0, n_days):
    #     r_t = data[t]
    #     outer = np.outer(r_t, r_t)
    #     #forgot to move lambda outside the loop. reminder for myself
    #     lambda_ = 0.94
    #     cov_matrix2 = lambda_ * cov_matrix + (1 - lambda_) * outer
    #
    # print(f"Bad covariance matrix: {cov_matrix2}")
    # okay so the above matrix (cov_matrix2) has been observed to have smaller values than the one below, proving the point that we need to have a standard average to standardise the EWMA or it will start from absolute 0 and mess up the results.

    for t in range (30, n_days):
        r_t = data[t]
        outer = np.outer(r_t, r_t)
        cov_matrix = lambda_ * cov_matrix + (1 - lambda_) * outer

    return cov_matrix

def compute_ledoit_wolf(returns):

    # we get the raw returns directly and not from the EWMA function because, ledoit wolf function internatlly calculates its own covariance matrix with the shrinkage applied to it.
    # this is intentionally different to EWMA. as this is a different method to it and not just an extension. Same inputs but they produce different 5x5 matrix.
    #converting the Dataframe into numpy array
    data = returns.values


    lp = LedoitWolf()

    lp.fit(data)

    #getting the shrunked stabilised matrix from the standardised data.
    cov_matrix = lp.covariance_

    return cov_matrix


if __name__ == "__main__":
    tickers = ["CBA.AX", "BHP.AX", "WES.AX", "ANZ.AX", "RIO.AX"]

    close_prices = download_price_data(tickers,  "2019-01-01" , "2024-01-01")

    returns =  calculate_log_returns(close_prices)
    cov_matrix = compute_ewma_covariance(returns)
    lp_cov_matrix = compute_ledoit_wolf(returns)


    print(f"EWMA Covariance Matrix dimension: {cov_matrix.shape}")
    print(cov_matrix)
    print("\n\n")

    print(f"Ledoit Wolf stabilised Covariance Matrix dimension: {lp_cov_matrix.shape}")
    print(lp_cov_matrix)