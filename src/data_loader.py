import yfinance as yf


def download_price_data(tickers, start_date, end_date):

    #using auto adjust to account for the dividends given to the shareholders.
    # it automatically adjusts the raw close column to adjusted closes.
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    close_data = data["Close"]
    clean_close_data = close_data.dropna()
    return clean_close_data


# using Apple as the experiment ticker from 1st of jan 2020 to current date 10th Apr 2026
# just did this outside the function to actually see the outputs and experiment.

# asx_data = yf.download("AAPL", start="2020-01-01", end="2026-04-10", auto_adjust=True)
# print(asx_data)
#
# #printed just to see what the data looks like
#
# asx_close = asx_data["Close"]
#
# clean_asx_close = asx_close.dropna()
#
# print(clean_asx_close)
