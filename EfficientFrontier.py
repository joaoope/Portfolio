import yfinance as yf
import pandas as pd
import numpy as np
from PortfolioOptimization import PortfolioSimpleOptimization
from scipy.optimize import minimize
import matplotlib.pyplot as plt

portfolio =  {'XLF': 0.25,
              'SPY': 0.25,
              'XLE': 0.25,
              'XLK': 0.25}

start_date = '2018-01-01'
end_date = '2023-09-23'

weight_change = 1

return_rows = 10 ** 4

prices = pd.DataFrame()

for stock in list(portfolio.keys()):
    prices[stock] = yf.download(stock).loc[start_date:end_date]['Adj Close']

returns = prices.pct_change().dropna()

expected_covariance = returns.cov()
expected_returns = returns.mean()

def objective(wts):
  return np.sum((expected_returns * wts * 252))

initial_weights = np.array(list(portfolio.values()))

bounds = []
for wt in initial_weights:
    bounds.append((max(-1, wt - weight_change), min(1, wt + weight_change)))

min_ret_port = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: 1 - sum(x)}).x
max_ret_port = minimize(lambda x: -1 * objective(x), initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: 1 - sum(x)}).x

efficient_frontier = pd.DataFrame()

efficient_frontier['Expected Return'] = np.linspace(objective(min_ret_port), objective(max_ret_port), return_rows)

for row in efficient_frontier.index:
    target_return = efficient_frontier.loc[row, 'Expected Return'] / 252
    pfolio_weights = np.array(PortfolioSimpleOptimization(expected_returns.to_dict(),portfolio,expected_covariance,'MinRisk',target_return,weight_change)['Optimal Weights'])
    efficient_frontier.loc[row, 'Expected Volatility'] = np.sqrt(np.dot(pfolio_weights.T, np.dot(expected_covariance * 252, pfolio_weights)))
    efficient_frontier['Sharpe'] = efficient_frontier['Expected Return'] / efficient_frontier['Expected Volatility']

print(efficient_frontier)

max_sharpe_portfolio = efficient_frontier.loc[efficient_frontier['Sharpe'].idxmax()]
min_volatility_portfolio = efficient_frontier.loc[efficient_frontier['Expected Volatility'].idxmin()]

# Plot the efficient frontier
plt.scatter(efficient_frontier['Expected Volatility'], efficient_frontier['Expected Return'], c=efficient_frontier['Sharpe'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Highlight the portfolios with max Sharpe ratio and min volatility
plt.scatter(max_sharpe_portfolio['Expected Volatility'], max_sharpe_portfolio['Expected Return'], c='red', marker='*', s=100)
plt.scatter(min_volatility_portfolio['Expected Volatility'], min_volatility_portfolio['Expected Return'], c='blue', marker='*', s=100)

plt.title('Efficient Frontier')
plt.show()