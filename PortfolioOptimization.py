import pandas as pd
import numpy as np 
from scipy.optimize import minimize

returns = {'XLF': 0.0315,
           'SPY': 0.0250,
           'XLE': 0.045,
           'XLK': 0.0132,
           'XTN': 0.0023,
           'XLY': 0.0056,
           'EWG': 0.045,
           'TLT': 0.003}

initial_weights = {'XLF': 0.125,
                   'SPY': 0.125,
                   'XLE': 0.125,
                   'XLK': 0.125,
                   'XTN': 0.125,
                   'XLY': 0.125,
                   'EWG': 0.125,
                   'TLT': 0.125}


covariance = {'Tickers': ['XLF', 'SPY', 'XLE', 'XLK', 'XTN', 'XLY', 'EWG', 'TLT'],
              'XLF': [0.0010, 0.0013, -0.0006, -0.0007,	0.0001,	0.0001,	-0.0004, -0.0004],
              'SPY': [0.0013, 0.0073, -0.0013, -0.0006,	-0.0022, -0.0010, 0.0014, -0.0015],
              'XLE': [-0.0006, -0.0013,	0.0599,	0.0276,	0.0635,	0.0230,	0.0330,	0.0480],
              'XLK': [-0.0007, -0.0006,	0.0276,	0.0296,	0.0266,	0.0215,	0.0207,	0.0299],
              'XTN': [0.0001, -0.0022, 0.0635, 0.0266, 0.1025, 0.0427, 0.0399, 0.0660],
              'XLY': [0.0001, -0.0010, 0.0230, 0.0215, 0.0427, 0.0321, 0.0199, 0.0322],
              'EWG': [-0.0004, 0.0014, 0.0330, 0.0207, 0.0399, 0.0199, 0.0284, 0.0351],
              'TLT': [-0.0004, -0.0015,	0.0480,	0.0299,	0.0660,	0.0322,	0.0351,	0.0800]
}

covariance_df = pd.DataFrame(covariance)
covariance_df = covariance_df.set_index('Tickers')

def PortfolioSimpleOptimization(returns,initial_weights,covariance_df,optimization,target=None,weight_change=None):
    if optimization == 'MaxReturn':
        def objective(weights):
            pfolio_return = -np.sum(np.array(list(returns.values())) * weights)
            return pfolio_return

        def constraint(weights):
            pfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_df.values, weights)))
            return target - pfolio_vol  
    
        def constraint_weights_sum(weights):
            return 1 - np.sum(weights)  
    
        constraints = [
            {'type': 'eq', 'fun': constraint_weights_sum},
            {'type': 'ineq', 'fun': constraint}
            ]

    elif optimization == 'MinRisk':
        def objective(weights):
            pfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_df.values, weights)))
            return pfolio_vol

        def constraint(weights):
            pfolio_return = np.sum(np.array(list(returns.values())) * weights)
            return pfolio_return - target
    
        def constraint_weights_sum(weights):
            return 1 - np.sum(weights)  
    
        constraints = [
            {'type': 'eq', 'fun': constraint_weights_sum},
            {'type': 'ineq', 'fun': constraint}
            ]
    
    elif optimization == 'MaxSharpe':
        def objective(weights):
            pfolio_return = np.sum(np.array(list(returns.values())) * weights)
            pfolio_vol = 100 * np.sqrt(np.dot(weights.T, np.dot(covariance_df.values, weights)))
            sharpe_ratio = pfolio_return / pfolio_vol
            return -sharpe_ratio 
    
        def constraint_weights_sum(weights):
            return 1 - np.sum(weights)  
    
        constraints = [{'type': 'eq', 'fun': constraint_weights_sum}]

    else:
        raise ValueError('Optimization must be MaxReturn, MinRisk or MaxSharpe')
 
    initial_weights = list(initial_weights.values())

    if weight_change is None:
        bounds = [(-1, 1)] * len(returns)
    else:
        bounds = []
        for wt in initial_weights:
            bounds.append((round(wt-weight_change,2), round(wt + weight_change,2)))
        bounds = tuple(bounds)
        
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds,  constraints=constraints)
    optimal_weights = result.x
    optimal_portfolio_return = np.sum(np.array(list(returns.values())) * optimal_weights)
    optimal_portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_df.values, optimal_weights)))
    optimal_sharpe_ratio = optimal_portfolio_return / optimal_portfolio_volatility

    return {
        "Optimal Weights": list(np.round(optimal_weights, decimals=3)),
        "Optimal Portfolio Return": optimal_portfolio_return,
        "Optimal Portfolio Volatility": optimal_portfolio_volatility,
        "Optimal Sharpe Ratio": optimal_sharpe_ratio
     }

#print(PortfolioSimpleOptimization(returns,initial_weights,covariance_df,'MinRisk',0.05,20))
#print(PortfolioSimpleOptimization(returns,initial_weights,covariance_df,'MaxReturn',0.2,20))
#print(PortfolioSimpleOptimization(returns,initial_weights,covariance_df,'MaxSharpe',weight_change=20))

def SharpeOptimalPortfolio(returns,initial_weights,covariance_df,tau,riskfree_return,optimization,weight_change=None): 

    def objective(weights):
        pfolio_return = np.sum(np.array(list(returns.values())) * weights) + (1 - weights.sum()) * riskfree_return
        pfolio_vol = 100 * np.sqrt(np.dot(weights.T, np.dot(covariance_df.values, weights)))
        pfolio_riskadjusted_return = pfolio_return - tau*(pfolio_vol**2)
        return -pfolio_riskadjusted_return 
     
    initial_weights = list(initial_weights.values())

    if weight_change is None:
        bounds = [(-1, 1)] * len(returns)
    else:
        bounds = []
    for wt in initial_weights:
        bounds.append((round(wt-weight_change,2), round(wt + weight_change,2)))
    bounds = tuple(bounds)
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds)
    optimal_weights = result.x
    optimal_portfolio_return = np.sum(np.array(list(returns.values())) * optimal_weights) + (1 - optimal_weights.sum()) * riskfree_return
    optimal_portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_df.values, optimal_weights)))
    optimal_riskadjusted_return = optimal_portfolio_return - tau*(optimal_portfolio_volatility**2)

    if optimization == 'Risk-Adjusted Maximization':
        return ({
            "Optimal Weights": list(np.round(optimal_weights, decimals=3)),
            "Optimal Portfolio Return": optimal_portfolio_return,
            "Optimal Portfolio Volatility": optimal_portfolio_volatility,
            "Optimal RiskAdjusted Ratio": optimal_riskadjusted_return
        })
    elif optimization == 'Sharpe Portfolio Calculation':
        sharpe_optimal_weights = optimal_weights / optimal_weights.sum()
        optimal_sharpe_ratio = ((np.sum(np.array(list(returns.values())) * optimal_weights) + (1 - optimal_weights.sum()) * riskfree_return) - riskfree_return) / optimal_portfolio_volatility
    
        return ({
            "Sharpe Optimal Weights": list(np.round(sharpe_optimal_weights, decimals=3)),
            "Optimal Sharpe Ratio": optimal_sharpe_ratio
        }) 
    else:
        raise ValueError('Optimization must be Risk-Adjusted Maximization or Sharpe Portfolio Calculation')

#print(SharpeOptimalPortfolio(returns,initial_weights,covariance_df,0.01,0.015,'Risk-Adjusted Maximization',20))
#print(SharpeOptimalPortfolio(returns,initial_weights,covariance_df,0.01,0.015,'Sharpe Portfolio Calculation',20))