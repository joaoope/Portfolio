import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm


portfolio =  {'XLF': 0.25,
              'SPY': 0.25,
              'XLE': 0.25,
              'XLK': 0.25}

portfolio = pd.DataFrame.from_dict(portfolio, orient='index', columns=['Weight'])

start_date = '2018-01-01'
end_date = '2023-09-23'

prices = pd.DataFrame()

for stock in list(portfolio.index):
    prices[stock] = yf.download(stock).loc[start_date:end_date]['Adj Close']

returns = prices.pct_change().dropna()
pfolio_returns = (returns * np.array(portfolio['Weight'])).sum(axis=1)

def calculate_var(returns_df,alpha=0.05,type='normal'):
    if type == 'normal':
        var_values = returns_df.quantile(alpha)
    elif type == 'parametric':
        mean_returns = returns_df.mean()
        std_dev_returns = returns_df.std()
        z_score = norm.ppf(1 - alpha)
        var_values = mean_returns - z_score * std_dev_returns
    else:
        raise ValueError('type must be normal or parametric')
    
    return var_values

def calculate_cvar(returns_df, alpha=0.05,type='normal'):
    var_values = calculate_var(returns_df, alpha, type)
    cvar_values = returns_df[returns_df <= var_values].mean()
    return cvar_values

print(calculate_var(pfolio_returns,type='normal'))
print(calculate_cvar(pfolio_returns,type='normal'))
print(calculate_var(pfolio_returns,type='parametric'))
print(calculate_cvar(pfolio_returns,type='parametric'))