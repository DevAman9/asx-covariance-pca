import pandas as pd
import numpy as np
from data_loader import download_price_data


def calculate_log_returns(price_data):
    # maths formula = (ln(todays price) - ln (yesterdays price)) * number of rows
    log_returns = price_data.apply(np.log).diff()
    # using dropna instead of just ignoring the first row. dropna is better. because it will not count any missing values at all.
    log_returns = log_returns.dropna()
    return log_returns


# testing the above function.
# reminder for myself: Add an ifstatement to run these with __name__
# tickers = ["CBA.AX", "BHP.AX", "WES.AX", "ANZ.AX", "RIO.AX"]
# close_price = download_price_data(tickers, "2020-01-01", "2020-12-12")
# returns_calc = calculate_log_returns(close_price)
#
# print(f"Dataframe with returns: \n\n {returns_calc}")
