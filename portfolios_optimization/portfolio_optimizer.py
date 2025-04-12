import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_markowitz_portfolio(returns, risk_aversion=1.0, period_ret=252):
    """
    Optimize a portfolio using Markowitz mean-variance optimization
    
    Parameters:
        returns (pd.DataFrame): DataFrame of asset returns
        risk_aversion (float): Risk aversion parameter (higher = more conservative)
        period_ret (int): Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
    
    Returns:
        tuple: (weights, expected_return, expected_volatility, sharpe_ratio)
    """
    # Number of assets
    n_assets = len(returns.columns)
    
    # Initial weights (equal allocation)
    init_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Bounds for weights (0 to 1)
    bounds = tuple([(0, 1) for _ in range(n_assets)])
    
    # Constraint (sum of weights = 1)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Mean returns and covariance matrix
    mean_returns = returns.mean() * period_ret
    cov_matrix = returns.cov() * period_ret
    
    # Objective function to minimize (negative Sharpe ratio or portfolio volatility)
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if risk_aversion == 1.0:
            # Maximize Sharpe ratio
            return -portfolio_return / portfolio_volatility
        else:
            # Mean-variance utility
            return -portfolio_return + risk_aversion * portfolio_volatility
    
    # Optimize the portfolio
    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Get the optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    expected_return = np.sum(mean_returns * weights)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = expected_return / expected_volatility
    
    return weights, expected_return, expected_volatility, sharpe_ratio

def get_efficient_frontier(returns, n_points=100, period_ret=252):
    """
    Generate points along the efficient frontier
    
    Parameters:
        returns (pd.DataFrame): DataFrame of asset returns
        n_points (int): Number of points to generate
        period_ret (int): Annualization factor
    
    Returns:
        tuple: (returns, volatilities, weights) for points along the frontier
    """
    # Number of assets
    n_assets = len(returns.columns)
    
    # Initial weights (equal allocation)
    init_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Bounds for weights (0 to 1)
    bounds = tuple([(0, 1) for _ in range(n_assets)])
    
    # Constraint (sum of weights = 1)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Mean returns and covariance matrix
    mean_returns = returns.mean() * period_ret
    cov_matrix = returns.cov() * period_ret
    
    # Function to minimize volatility
    def minimize_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Get the range of possible returns
    # First, find the minimum volatility portfolio
    min_vol_result = minimize(minimize_volatility, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    min_vol_weights = min_vol_result['x']
    min_vol_return = np.sum(mean_returns * min_vol_weights)
    
    # Find return of each asset to know the range
    asset_returns = np.array(mean_returns)
    min_return = min_vol_return
    max_return = asset_returns.max()
    
    # Generate target returns
    target_returns = np.linspace(min_return, max_return, n_points)
    
    # For each target return, find the portfolio with minimum volatility
    frontier_returns = []
    frontier_volatilities = []
    frontier_weights = []
    
    for target_return in target_returns:
        # Add constraint for target return
        return_constraint = {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return}
        all_constraints = constraints + [return_constraint]
        
        # Optimize
        result = minimize(minimize_volatility, init_weights, method='SLSQP', bounds=bounds, constraints=all_constraints)
        
        if result['success']:
            weights = result['x']
            volatility = minimize_volatility(weights)
            
            frontier_returns.append(target_return)
            frontier_volatilities.append(volatility)
            frontier_weights.append(weights)
    
    return frontier_returns, frontier_volatilities, frontier_weights

def generate_random_portfolios(returns, n_portfolios=10000, period_ret=252):
    """
    Generate random portfolios for efficient frontier visualization
    
    Parameters:
        returns (pd.DataFrame): DataFrame of asset returns
        n_portfolios (int): Number of random portfolios to generate
        period_ret (int): Annualization factor
    
    Returns:
        tuple: (returns, volatilities, sharpe_ratios) for random portfolios
    """
    # Number of assets
    n_assets = len(returns.columns)
    
    # Mean returns and covariance matrix
    mean_returns = returns.mean() * period_ret
    cov_matrix = returns.cov() * period_ret
    
    # Arrays to store results
    portfolio_returns = np.zeros(n_portfolios)
    portfolio_volatilities = np.zeros(n_portfolios)
    portfolio_sharpe_ratios = np.zeros(n_portfolios)
    
    # Generate random portfolios
    for i in range(n_portfolios):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate Sharpe ratio
        portfolio_sharpe_ratio = portfolio_return / portfolio_volatility
        
        # Store results
        portfolio_returns[i] = portfolio_return
        portfolio_volatilities[i] = portfolio_volatility
        portfolio_sharpe_ratios[i] = portfolio_sharpe_ratio
    
    return portfolio_returns, portfolio_volatilities, portfolio_sharpe_ratios 