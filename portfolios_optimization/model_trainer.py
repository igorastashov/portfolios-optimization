import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from scipy.optimize import minimize

def train_model(model_type, assets, price_data, train_start_date, train_end_date, **kwargs):
    """
    Train a portfolio optimization model
    
    Parameters:
        model_type (str): Type of model ('markowitz', 'a2c', 'ppo', 'ddpg', 'sac')
        assets (list): List of assets to include in the portfolio
        price_data (pd.DataFrame): Historical price data
        train_start_date (datetime): Start date for training
        train_end_date (datetime): End date for training
        **kwargs: Additional parameters specific to each model type
        
    Returns:
        dict: Trained model information
    """
    # Filter data for training period and selected assets
    train_data = price_data.loc[train_start_date:train_end_date, assets]
    
    # Calculate returns
    returns = train_data.pct_change().dropna()
    
    if model_type.lower() == 'markowitz':
        # Train Markowitz model
        model_info = train_markowitz(returns, **kwargs)
    elif model_type.lower() in ['a2c', 'ppo', 'ddpg', 'sac']:
        # Train RL model (placeholder for actual implementation)
        model_info = train_rl_model(model_type, returns, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Add model metadata
    model_info.update({
        'model_type': model_type,
        'assets': assets,
        'train_start_date': train_start_date.strftime('%Y-%m-%d'),
        'train_end_date': train_end_date.strftime('%Y-%m-%d'),
        'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return model_info

def train_markowitz(returns, window_size=180, rebalance_period=7, risk_aversion=1.0, **kwargs):
    """
    Train a Markowitz portfolio optimization model
    
    Parameters:
        returns (pd.DataFrame): Asset returns
        window_size (int): Lookback window in days
        rebalance_period (int): Rebalance period in days
        risk_aversion (float): Risk aversion parameter
        **kwargs: Additional parameters
        
    Returns:
        dict: Trained model information
    """
    # Number of assets
    n_assets = len(returns.columns)
    assets = returns.columns.tolist()
    
    # Initialize portfolio weights
    init_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Store optimal weights for each rebalance date
    optimal_weights = {}
    
    # Get unique rebalance dates
    all_dates = returns.index
    rebalance_dates = []
    
    # Convert rebalance_period from days to number of rows (approximation)
    rows_per_day = len(returns) / (returns.index[-1] - returns.index[0]).days
    rebalance_rows = int(rebalance_period * rows_per_day)
    
    # Select rebalance dates
    for i in range(0, len(all_dates), rebalance_rows):
        if i < len(all_dates):
            rebalance_dates.append(all_dates[i])
    
    # For each rebalance date, calculate optimal weights
    for date in rebalance_dates:
        # Get historical data for the window
        window_end = date
        window_start_idx = max(0, all_dates.get_loc(date) - window_size)
        window_start = all_dates[window_start_idx]
        
        # Window data
        window_returns = returns.loc[window_start:window_end]
        
        # Calculate optimal weights
        weights = optimize_portfolio(window_returns, risk_aversion)
        
        # Store weights
        optimal_weights[date.strftime('%Y-%m-%d')] = {
            asset: weight for asset, weight in zip(assets, weights)
        }
    
    # Create model info
    model_info = {
        'weights': optimal_weights,
        'parameters': {
            'window_size': window_size,
            'rebalance_period': rebalance_period,
            'risk_aversion': risk_aversion
        }
    }
    
    return model_info

def train_rl_model(model_type, returns, total_timesteps=100000, learning_rate=0.001, **kwargs):
    """
    Train a reinforcement learning model for portfolio optimization
    
    Parameters:
        model_type (str): RL model type ('a2c', 'ppo', 'ddpg', 'sac')
        returns (pd.DataFrame): Asset returns
        total_timesteps (int): Total training timesteps
        learning_rate (float): Learning rate
        **kwargs: Additional parameters
        
    Returns:
        dict: Trained model information
    """
    # This is a placeholder for actual RL model training
    # In a real application, you would use libraries like FinRL or Stable Baselines
    
    # Create a placeholder model info
    model_info = {
        'weights': {},  # Would contain weights per date
        'parameters': {
            'model_type': model_type,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate
        }
    }
    
    # In a real implementation:
    # 1. Set up the RL environment with the asset returns
    # 2. Define the RL agent (A2C, PPO, DDPG, or SAC)
    # 3. Train the agent
    # 4. Extract the optimal policy (weights)
    
    return model_info

def optimize_portfolio(returns, risk_aversion=1.0):
    """
    Optimize a portfolio using mean-variance optimization
    
    Parameters:
        returns (pd.DataFrame): Asset returns
        risk_aversion (float): Risk aversion parameter
        
    Returns:
        np.array: Optimal portfolio weights
    """
    # Number of assets
    n_assets = len(returns.columns)
    
    # Initial weights (equal allocation)
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Bounds for weights (0 to 1)
    bounds = tuple([(0, 1) for _ in range(n_assets)])
    
    # Constraint (sum of weights = 1)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 252  # Annualize
    cov_matrix = returns.cov() * 252  # Annualize
    
    # Define objective function (minimize negative Sharpe ratio)
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if risk_aversion == 1.0:
            # Maximize Sharpe ratio (minimize negative Sharpe)
            return -portfolio_return / portfolio_std_dev
        else:
            # Mean-variance utility
            return -portfolio_return + risk_aversion * portfolio_std_dev
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Return optimal weights
    return result['x']

def load_trained_model(model_type, model_name):
    """
    Load a trained model
    
    Parameters:
        model_type (str): Type of model ('markowitz', 'a2c', 'ppo', 'ddpg', 'sac')
        model_name (str): Name of the model to load
        
    Returns:
        dict: Loaded model information
    """
    # Define path to the model
    model_path = os.path.join("notebooks", "trained_models", model_type, model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # For demonstration purposes, return a placeholder
    # In a real application, you would load the actual model
    
    model_info = {
        'model_type': model_type,
        'model_name': model_name,
        'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            # Would contain actual model parameters
        },
        'weights': {
            # Would contain actual model weights
        }
    }
    
    return model_info

def save_model(model_info, model_type, model_name):
    """
    Save a trained model
    
    Parameters:
        model_info (dict): Model information to save
        model_type (str): Type of model ('markowitz', 'a2c', 'ppo', 'ddpg', 'sac')
        model_name (str): Name for the saved model
        
    Returns:
        str: Path to the saved model
    """
    # Create model directory if it doesn't exist
    model_dir = os.path.join("notebooks", "trained_models", model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Model path
    model_path = os.path.join(model_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # For demonstration purposes, we're not actually saving the model
    # In a real application, you would save model files
    
    return model_path 