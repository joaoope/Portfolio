import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from ValueAtRisk import calculate_cvar

# Portfolio initial weights:
portfolio = {
    'XLF': 0.25,
    'SPY': 0.25,
    'XLE': 0.25,
    'XLK': 0.25
}

start_date, end_date = '2018-01-01', '2023-09-23'  # Date range
weight_change = 1  # Maximum weight change for each asset
alpha = 0.05  # Confidence level for VaR
return_rows = 10 ** 3  # Number of return points on the efficient frontier

# Download stock price data
prices = pd.DataFrame()
for stock in list(portfolio.keys()):
    prices[stock] = yf.download(stock).loc[start_date:end_date]['Adj Close']

returns = prices.pct_change().dropna()

# Calculate expected covariance and returns
expected_covariance = returns.cov()
expected_returns = returns.mean()

# Define the objective function to minimize VaR
def objective(wts):
    pfolio_returns = (returns * wts).sum(axis=1)
    portfolio_cvar = calculate_cvar(pfolio_returns, alpha)
    return -portfolio_cvar

initial_weights = np.array(list(portfolio.values()))

# Define bounds for portfolio weights
bounds = []
for wt in initial_weights:
    bounds.append((max(-1, wt - weight_change), min(1, wt + weight_change)))
 
# Define the objective function to maximize return so we can optimize for maximum and minimum portfolio
def objective_returns(wts):
  return np.sum((expected_returns * wts * 252))

# Optimize for minimum and maximum return portfolios
min_ret_port = ((minimize(objective_returns, initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: 1 - sum(x)}).x) * expected_returns).sum()
max_ret_port = ((minimize(lambda x: -1 * objective_returns(x), initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: 1 - sum(x)}).x) * expected_returns).sum()

# Create a range of target returns
efficient_frontier = pd.DataFrame()
efficient_frontier['Expected Return'] = np.linspace(min_ret_port, max_ret_port, return_rows)

# Calculate the efficient frontier using VaR as the risk measure
for row in efficient_frontier.index:
    target_return = efficient_frontier.loc[row, 'Expected Return']
    
    # Define the constraint to enforce the target return
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights constraint
                   {'type': 'eq', 'fun': lambda w: np.sum(w * expected_returns) - target_return}]
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    pfolio_weights = result.x
    pfolio_daily_returns = (pfolio_weights * returns).sum(axis=1)
    portfolio_returns = pfolio_daily_returns.mean() * 252
    efficient_frontier.loc[row, 'Portfolio Returns'] = portfolio_returns.round(4)
    efficient_frontier.loc[row, 'Portfolio CVaR'] = calculate_cvar(pfolio_daily_returns, alpha) * -1

# Find portfolios with max Sharpe ratio and min volatility
max_return_portfolio = efficient_frontier.loc[efficient_frontier['Portfolio Returns'].idxmax()]
min_cvar_portfolio = efficient_frontier.loc[efficient_frontier['Portfolio CVaR'].idxmin()]

# Plot the efficient frontier
plt.scatter(efficient_frontier['Portfolio CVaR'], efficient_frontier['Portfolio Returns'])
plt.xlabel('Value-at-Risk')
plt.ylabel('Return')

# Highlight the portfolios with max Sharpe ratio and min volatility
plt.scatter(max_return_portfolio['Portfolio CVaR'], max_return_portfolio['Portfolio Returns'], c='red', marker='*', s=100)
plt.scatter(min_cvar_portfolio['Portfolio CVaR'], min_cvar_portfolio['Portfolio Returns'], c='blue', marker='*', s=100)

plt.title('Efficient CVaR Portfolios')
plt.show()



