import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from .portfolio_optimizer import get_efficient_frontier, generate_random_portfolios

def calculate_metrics(returns):
    """
    Calculate performance metrics for portfolio returns
    
    Parameters:
        returns (pd.DataFrame): DataFrame of portfolio returns
    
    Returns:
        pd.DataFrame: DataFrame with performance metrics
    """
    # Convert to cumulative returns
    cum_returns = returns.cumsum()
    
    # Calculate metrics
    metrics = pd.DataFrame({
        "Total Return": cum_returns.iloc[-1],
        "Annual Return": (1 + cum_returns.iloc[-1]) ** (365 / len(cum_returns)) - 1,
        "Sharpe Ratio": cum_returns.iloc[-1] / cum_returns.std(),
        "Volatility": cum_returns.std(),
        "Max Drawdown": cum_returns.apply(lambda x: (x.cummax() - x).max())
    })
    
    # Format percentages
    for col in ["Total Return", "Annual Return", "Max Drawdown"]:
        metrics[col] = metrics[col].apply(lambda x: f"{x*100:.2f}%")
    
    for col in ["Sharpe Ratio", "Volatility"]:
        metrics[col] = metrics[col].apply(lambda x: f"{x:.4f}")
    
    return metrics

def calculate_drawdown(returns):
    """
    Calculate drawdown series for portfolio returns
    
    Parameters:
        returns (pd.DataFrame): DataFrame of portfolio returns
    
    Returns:
        pd.DataFrame: DataFrame with drawdown series
    """
    # Convert to cumulative returns
    cum_returns = returns.cumsum()
    
    # Calculate drawdowns
    drawdowns = {}
    for column in cum_returns.columns:
        cum_ret = cum_returns[column]
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        drawdowns[column] = drawdown
    
    return pd.DataFrame(drawdowns)

def calculate_rolling_metrics(returns, window=30):
    """
    Calculate rolling performance metrics
    
    Parameters:
        returns (pd.DataFrame): DataFrame of portfolio returns
        window (int): Rolling window size
    
    Returns:
        tuple: (rolling_returns, rolling_volatility, rolling_sharpe)
    """
    # Calculate rolling metrics
    rolling_returns = returns.rolling(window=window).mean() * 252  # Annualize
    rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualize
    rolling_sharpe = rolling_returns / rolling_volatility
    
    return rolling_returns, rolling_volatility, rolling_sharpe

def plot_efficient_frontier(returns, optimal_weights=None, period_ret=252, n_random=1000):
    """
    Plot the efficient frontier with random portfolios
    
    Parameters:
        returns (pd.DataFrame): DataFrame of asset returns
        optimal_weights (array): Weights of the optimal portfolio
        period_ret (int): Annualization factor
        n_random (int): Number of random portfolios to generate
    
    Returns:
        plotly.graph_objects.Figure: Efficient frontier plot
    """
    # Generate random portfolios
    rand_returns, rand_volatilities, rand_sharpe = generate_random_portfolios(
        returns, n_portfolios=n_random, period_ret=period_ret
    )
    
    # Generate efficient frontier
    ef_returns, ef_volatilities, _ = get_efficient_frontier(
        returns, n_points=50, period_ret=period_ret
    )
    
    # Create figure
    fig = go.Figure()
    
    # Add random portfolios scatter plot
    fig.add_trace(
        go.Scatter(
            x=rand_volatilities,
            y=rand_returns,
            mode='markers',
            marker=dict(
                size=5,
                color=rand_sharpe,
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                showscale=True
            ),
            name='Random Portfolios'
        )
    )
    
    # Add efficient frontier line
    fig.add_trace(
        go.Scatter(
            x=ef_volatilities,
            y=ef_returns,
            mode='lines',
            line=dict(color='red', width=3),
            name='Efficient Frontier'
        )
    )
    
    # Add optimal portfolio if provided
    if optimal_weights is not None:
        # Calculate optimal portfolio metrics
        mean_returns = returns.mean() * period_ret
        cov_matrix = returns.cov() * period_ret
        
        opt_return = np.sum(mean_returns * optimal_weights)
        opt_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        
        fig.add_trace(
            go.Scatter(
                x=[opt_volatility],
                y=[opt_return],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    symbol='star'
                ),
                name='Optimal Portfolio'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Annualized)',
        yaxis_title='Return (Annualized)',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='closest'
    )
    
    return fig

def analyze_portfolio(weights, returns, portfolio_name="Portfolio"):
    """
    Analyze a portfolio with given weights
    
    Parameters:
        weights (array): Portfolio weights
        returns (pd.DataFrame): DataFrame of asset returns
        portfolio_name (str): Name of the portfolio
    
    Returns:
        dict: Dictionary with portfolio analysis results
    """
    # Asset names
    assets = returns.columns
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(np.dot(returns, weights), index=returns.index, name=portfolio_name)
    
    # Calculate cumulative returns
    cum_returns = portfolio_returns.cumsum()
    
    # Calculate drawdown
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    
    # Calculate metrics
    total_return = cum_returns.iloc[-1]
    annual_return = (1 + total_return) ** (252 / len(cum_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility
    max_drawdown = drawdown.min()
    
    # Create results dictionary
    results = {
        "weights": dict(zip(assets, weights)),
        "returns": portfolio_returns,
        "cumulative_returns": cum_returns,
        "drawdown": drawdown,
        "metrics": {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    }
    
    return results 