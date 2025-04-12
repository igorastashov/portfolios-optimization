import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_portfolio_performance(returns, title="Portfolio Performance"):
    """
    Plot portfolio performance over time
    
    Parameters:
        returns (pd.DataFrame): DataFrame of portfolio returns
        title (str): Plot title
    
    Returns:
        plotly.graph_objects.Figure: Performance plot
    """
    # Calculate cumulative returns
    cum_returns = returns.cumsum()
    
    # Create figure
    fig = px.line(
        cum_returns,
        x=cum_returns.index,
        y=cum_returns.columns,
        title=title,
        labels={"value": "Return", "variable": "Portfolio"}
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        hovermode="x unified"
    )
    
    return fig

def plot_drawdown(returns, title="Portfolio Drawdown"):
    """
    Plot portfolio drawdown over time
    
    Parameters:
        returns (pd.DataFrame): DataFrame of portfolio returns
        title (str): Plot title
    
    Returns:
        plotly.graph_objects.Figure: Drawdown plot
    """
    # Calculate cumulative returns
    cum_returns = returns.cumsum()
    
    # Calculate drawdowns
    drawdowns = {}
    for column in cum_returns.columns:
        cum_ret = cum_returns[column]
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        drawdowns[column] = drawdown
    
    drawdown_df = pd.DataFrame(drawdowns)
    
    # Create figure
    fig = px.line(
        drawdown_df,
        x=drawdown_df.index,
        y=drawdown_df.columns,
        title=title,
        labels={"value": "Drawdown", "variable": "Portfolio"}
    )
    
    # Add horizontal lines
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.add_hline(y=-0.1, line_width=1, line_dash="dash", line_color="orange")
    fig.add_hline(y=-0.2, line_width=1, line_dash="dash", line_color="red")
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        legend=dict(x=0.01, y=0.01, bordercolor="Black", borderwidth=1),
        hovermode="x unified"
    )
    
    return fig

def plot_rolling_metrics(returns, window=30, title="Rolling Metrics"):
    """
    Plot rolling performance metrics
    
    Parameters:
        returns (pd.DataFrame): DataFrame of portfolio returns
        window (int): Rolling window size
        title (str): Plot title
    
    Returns:
        plotly.graph_objects.Figure: Rolling metrics plot
    """
    # Calculate rolling metrics (annualized)
    rolling_returns = returns.rolling(window=window).mean() * 252
    rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_volatility
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Rolling Return (Annualized)", "Rolling Volatility (Annualized)", "Rolling Sharpe Ratio"],
        vertical_spacing=0.1
    )
    
    # Add rolling returns
    for column in rolling_returns.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns[column],
                name=f"{column} Return",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add rolling volatility
    for column in rolling_volatility.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_volatility.index,
                y=rolling_volatility[column],
                name=f"{column} Volatility",
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Add rolling Sharpe ratio
    for column in rolling_sharpe.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe[column],
                name=f"{column} Sharpe",
                showlegend=True
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{title} ({window}-Day Window)",
        height=900,
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        hovermode="x unified"
    )
    
    # Update y-axis formats
    fig.update_yaxes(title_text="Return", row=1, col=1, tickformat=".1%")
    fig.update_yaxes(title_text="Volatility", row=2, col=1, tickformat=".1%")
    fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
    
    # Update x-axis
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def plot_asset_allocation(weights, title="Portfolio Allocation"):
    """
    Plot asset allocation
    
    Parameters:
        weights (dict or array): Portfolio weights
        title (str): Plot title
    
    Returns:
        plotly.graph_objects.Figure: Asset allocation plot
    """
    # Convert weights to dict if array
    if not isinstance(weights, dict):
        assets = [f"Asset {i+1}" for i in range(len(weights))]
        weights_dict = dict(zip(assets, weights))
    else:
        weights_dict = weights
    
    # Create DataFrame
    df = pd.DataFrame({
        'Asset': list(weights_dict.keys()),
        'Weight': list(weights_dict.values())
    })
    
    # Sort by weight (descending)
    df = df.sort_values('Weight', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Asset',
        y='Weight',
        title=title,
        text_auto='.1%'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Asset",
        yaxis_title="Allocation",
        yaxis_tickformat=".1%"
    )
    
    return fig

def plot_asset_correlations(returns, title="Asset Correlations"):
    """
    Plot asset correlation heatmap
    
    Parameters:
        returns (pd.DataFrame): DataFrame of asset returns
        title (str): Plot title
    
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="equal",
        color_continuous_scale='RdBu_r',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Asset",
        yaxis_title="Asset"
    )
    
    return fig 