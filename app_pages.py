import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolios_optimization.portfolio_optimizer import optimize_markowitz_portfolio
from portfolios_optimization.portfolio_analysis import calculate_metrics, plot_efficient_frontier
from portfolios_optimization.visualization import plot_portfolio_performance, plot_asset_allocation
from portfolios_optimization.model_trainer import train_model, load_trained_model

def render_dashboard(price_data, model_returns, model_actions, assets):
    """Отображение страницы Dashboard"""
    st.header("Portfolio Dashboard")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Portfolio Performance")
        
        # Выбор моделей для отображения
        models = st.multiselect(
            "Select models to compare",
            options=model_returns.columns.tolist() if not model_returns.empty else [],
            default=model_returns.columns.tolist()[:2] if not model_returns.empty and len(model_returns.columns) > 1 else []
        )
        
        if models:
            # Расчет накопленной доходности
            cum_returns = model_returns[models].cumsum()
            
            # График доходности
            fig = px.line(
                cum_returns, 
                x=cum_returns.index, 
                y=cum_returns.columns,
                title="Cumulative Returns",
                labels={"value": "Return", "variable": "Model"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
            # Таблица метрик
            metrics = pd.DataFrame({
                "Total Return": cum_returns.iloc[-1],
                "Sharpe Ratio": cum_returns.iloc[-1] / cum_returns.std(),
                "Max Drawdown": cum_returns.apply(lambda x: (x.cummax() - x).max())
            })
            
            st.table(metrics)
    
    with col2:
        st.subheader("Latest Allocations")
        
        # Получение списка моделей с данными распределения
        available_models = list(model_actions.keys())
        
        if available_models:
            # Выбор модели
            selected_model = st.selectbox(
                "Select model",
                options=available_models,
                index=0
            )
            
            if selected_model and not model_actions[selected_model].empty:
                # Получение последнего распределения
                latest_allocation = model_actions[selected_model].iloc[-1]
                
                # Круговая диаграмма
                fig = px.pie(
                    values=latest_allocation.values,
                    names=latest_allocation.index,
                    title=f"Latest {selected_model.upper()} Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation data available. Train models first or load existing models.")
    
    # Показатели активов
    st.subheader("Asset Performance")
    
    if assets:
        # Выбор активов для отображения
        selected_assets = st.multiselect(
            "Select assets",
            options=assets,
            default=assets[:5] if len(assets) > 5 else assets
        )
        
        if selected_assets:
            # Расчет нормализованных цен
            normalized_prices = price_data[selected_assets] / price_data[selected_assets].iloc[0]
            
            # График цен активов
            fig = px.line(
                normalized_prices, 
                x=normalized_prices.index, 
                y=normalized_prices.columns,
                title="Normalized Asset Prices",
                labels={"value": "Normalized Price", "variable": "Asset"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No asset data available. Please check data sources.")

def render_portfolio_optimization(price_data, assets):
    """Отображение страницы оптимизации портфеля"""
    st.header("Portfolio Optimization")
    
    if not assets:
        st.error("No asset data available. Please check data sources.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimization Parameters")
            
            # Выбор активов для портфеля
            portfolio_assets = st.multiselect(
                "Select assets for your portfolio",
                options=assets,
                default=assets[:7] if len(assets) > 7 else assets
            )
            
            # Выбор периода оптимизации
            lookback_period = st.slider(
                "Lookback period (days)",
                min_value=30,
                max_value=365,
                value=180,
                step=30
            )
            
            # Параметр неприятия риска
            risk_aversion = st.slider(
                "Risk aversion (higher means more conservative)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            # Кнопка оптимизации
            optimize_button = st.button("Optimize Portfolio")
        
        with col2:
            if portfolio_assets and optimize_button:
                st.subheader("Optimization Results")
                
                # Фильтрация данных для выбранных активов и периода
                if not price_data.empty:
                    end_date = price_data.index[-1]
                    start_date = end_date - timedelta(days=lookback_period)
                    filtered_data = price_data.loc[start_date:end_date, portfolio_assets]
                    
                    # Расчет доходностей
                    returns = filtered_data.pct_change().dropna()
                    
                    # Оптимизация портфеля
                    weights, expected_return, expected_volatility, sharpe_ratio = optimize_markowitz_portfolio(
                        returns, risk_aversion=risk_aversion
                    )
                    
                    # Отображение результатов
                    results = pd.DataFrame({
                        'Asset': portfolio_assets,
                        'Weight': weights
                    })
                    
                    # Метрики
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Expected Return (Annual)", f"{expected_return*100:.2f}%")
                    metrics_col2.metric("Expected Volatility (Annual)", f"{expected_volatility*100:.2f}%")
                    metrics_col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    # Веса портфеля
                    st.subheader("Optimal Portfolio Weights")
                    
                    # Столбчатая диаграмма весов
                    fig = px.bar(
                        results,
                        x='Asset',
                        y='Weight',
                        title="Portfolio Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Эффективная граница
                    st.subheader("Efficient Frontier")
                    frontier_fig = plot_efficient_frontier(returns, weights)
                    st.plotly_chart(frontier_fig, use_container_width=True)
                    
                    # Добавление кнопки "Сохранить в мой портфель"
                    if st.button("Сохранить в мой портфель"):
                        # Здесь будет логика сохранения в портфель пользователя
                        # Этот функционал будет добавлен в файле auth_app.py
                        st.success("Портфель сохранен в ваш личный кабинет!")
                else:
                    st.error("No price data available.")
            
            elif not portfolio_assets:
                st.info("Please select assets for your portfolio.")
            elif not optimize_button:
                st.info("Click 'Optimize Portfolio' to see the results.")

def render_model_training(price_data, assets):
    """Отображение страницы обучения моделей"""
    st.header("Model Training & Selection")
    
    tab1, tab2 = st.tabs(["Choose Pretrained Model", "Train New Model"])
    
    with tab1:
        st.subheader("Choose Pretrained Model")
        
        # Получение списка типов моделей из директории
        model_types = ["markowitz", "a2c", "ppo", "ddpg", "sac"]
        
        # Выбор типа модели
        selected_model_type = st.selectbox(
            "Select model type",
            options=model_types,
            index=0
        )
        
        # Получение доступных обученных моделей выбранного типа
        trained_models_path = os.path.join("notebooks", "trained_models", selected_model_type) 
        
        # Проверка существования директории
        if not os.path.exists(trained_models_path):
            st.warning(f"No trained models found for {selected_model_type}. Directory {trained_models_path} does not exist.")
            trained_models = []
        else:
            # Список директорий (каждая директория - обученная модель)
            trained_models = [d for d in os.listdir(trained_models_path) 
                             if os.path.isdir(os.path.join(trained_models_path, d))]
        
        if trained_models:
            # Выбор конкретной модели
            selected_model = st.selectbox(
                "Select trained model",
                options=trained_models,
                index=0
            )
            
            # Отображение информации о модели
            st.info(f"Selected model: {selected_model_type}/{selected_model}")
            
            # Кнопка загрузки модели
            if st.button("Load Model"):
                model_path = os.path.join(trained_models_path, selected_model)
                
                # Заглушка для загрузки модели (фактическая реализация зависела бы от типа модели)
                st.success(f"Model loaded from {model_path}")
                
                # Здесь вы бы фактически загрузили модель и, возможно, отобразили ее параметры
                st.json({
                    "model_type": selected_model_type,
                    "model_name": selected_model,
                    "trained_date": "2023-01-01",  # Пример, был бы получен из метаданных модели
                    "performance": {
                        "sharpe_ratio": 1.5,
                        "return": "15.2%",
                        "volatility": "10.1%"
                    }
                })
        else:
            st.warning(f"No trained models found for {selected_model_type}.")
    
    with tab2:
        st.subheader("Train New Model")
        
        # Выбор типа модели
        model_type = st.selectbox(
            "Select model type to train",
            options=["markowitz", "a2c", "ppo", "ddpg", "sac"],
            index=0
        )
        
        # Выбор активов
        training_assets = st.multiselect(
            "Select assets for training",
            options=assets,
            default=assets[:7] if len(assets) > 7 else assets
        )
        
        # Параметры обучения
        col1, col2 = st.columns(2)
        
        with col1:
            train_start_date = st.date_input(
                "Training Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
            
            train_end_date = st.date_input(
                "Training End Date",
                value=datetime.now(),
                min_value=train_start_date,
                max_value=datetime.now()
            )
        
        with col2:
            if model_type == "markowitz":
                # Параметры модели Марковица
                rebalance_period = st.slider(
                    "Rebalance Period (days)",
                    min_value=1,
                    max_value=30,
                    value=7
                )
                
                window_size = st.slider(
                    "Lookback Window (days)",
                    min_value=30,
                    max_value=365,
                    value=180
                )
            else:
                # Параметры RL-моделей
                total_timesteps = st.number_input(
                    "Total Training Timesteps",
                    min_value=10000,
                    max_value=1000000,
                    value=100000,
                    step=10000
                )
                
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    format="%f"
                )
        
        # Имя модели
        model_name = st.text_input(
            "Model Name",
            value=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Кнопка обучения
        train_button = st.button("Train Model")
        
        if train_button:
            if not training_assets:
                st.error("Please select assets for training.")
            else:
                # Отображение прогресса
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Симуляция обучения
                for i in range(1, 101):
                    # Обновление прогресс-бара
                    progress_bar.progress(i)
                    status_text.text(f"Training in progress... {i}%")
                    
                    # Симуляция некоторого времени обработки
                    if i % 10 == 0:
                        st.info(f"Training step {i}/100 completed...")
                    
                    # Задержка для симуляции обработки
                    import time
                    time.sleep(0.1)
                
                # Отображение завершения
                st.success(f"Model {model_name} training completed!")
                
                # Здесь вы бы фактически сохранили модель и ее параметры
                st.json({
                    "model_type": model_type,
                    "model_name": model_name,
                    "trained_date": datetime.now().strftime("%Y-%m-%d"),
                    "parameters": {
                        "assets": training_assets,
                        "train_start": train_start_date.strftime("%Y-%m-%d"),
                        "train_end": train_end_date.strftime("%Y-%m-%d"),
                        "learning_rate": learning_rate if model_type != "markowitz" else "N/A",
                        "total_timesteps": total_timesteps if model_type != "markowitz" else "N/A",
                        "rebalance_period": rebalance_period if model_type == "markowitz" else "N/A",
                        "window_size": window_size if model_type == "markowitz" else "N/A"
                    }
                })

def render_model_comparison(model_returns, model_actions):
    """Отображение страницы сравнения моделей"""
    st.header("Model Comparison")
    
    if model_returns.empty:
        st.warning("No model return data available. Please train or load models first.")
    else:
        # Выбор моделей для сравнения
        models = st.multiselect(
            "Select models to compare",
            options=model_returns.columns.tolist(),
            default=model_returns.columns.tolist()
        )
        
        if models:
            # Расчет накопленной доходности
            cum_returns = model_returns[models].cumsum()
            
            # График доходности
            fig = px.line(
                cum_returns, 
                x=cum_returns.index, 
                y=cum_returns.columns,
                title="Cumulative Returns Comparison",
                labels={"value": "Return", "variable": "Model"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Метрики доходности
            st.subheader("Performance Metrics")
            
            metrics = pd.DataFrame({
                "Total Return": cum_returns.iloc[-1],
                "Annual Return": (1 + cum_returns.iloc[-1]) ** (365 / len(cum_returns)) - 1,
                "Sharpe Ratio": cum_returns.iloc[-1] / cum_returns.std(),
                "Volatility": cum_returns.std(),
                "Max Drawdown": cum_returns.apply(lambda x: (x.cummax() - x).max())
            })
            
            # Форматирование процентов
            for col in ["Total Return", "Annual Return"]:
                metrics[col] = metrics[col].apply(lambda x: f"{x*100:.2f}%")
            
            st.table(metrics)
            
            # Сравнение распределения активов моделей
            st.subheader("Asset Allocation Comparison")
            
            # Получение списка моделей с данными распределения
            models_with_allocations = [m for m in models if m.lower() in model_actions]
            
            if models_with_allocations:
                # Подготовка списка дат для выбора
                available_dates = []
                for model in models_with_allocations:
                    if model.lower() in model_actions and not model_actions[model.lower()].empty:
                        available_dates.extend(model_actions[model.lower()].index.tolist())
                
                # Удаление дубликатов и сортировка
                available_dates = sorted(list(set(available_dates)))
                
                if available_dates:
                    # Выбор даты для сравнения
                    selected_date = st.selectbox(
                        "Select date for allocation comparison",
                        options=available_dates,
                        index=len(available_dates)-1
                    )
                    
                    if selected_date:
                        try:
                            # Создание DataFrame с распределениями от всех моделей
                            allocations = pd.DataFrame({
                                model: model_actions[model.lower()].loc[selected_date] 
                                for model in models_with_allocations if model.lower() in model_actions
                                and selected_date in model_actions[model.lower()].index
                            })
                            
                            if not allocations.empty:
                                # График распределений
                                fig = px.bar(
                                    allocations,
                                    title=f"Model Allocations on {selected_date}",
                                    barmode="group"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No allocation data found for the selected date: {selected_date}")
                        except KeyError as e:
                            st.error(f"Error accessing allocation data: {e}")
                else:
                    st.warning("No allocation dates available.")
            else:
                st.warning("None of the selected models have allocation data.")

def render_backtest(model_returns):
    """Отображение страницы бэктестирования"""
    st.header("Backtest Results")
    
    if model_returns.empty:
        st.warning("No model return data available. Please train or load models first.")
    else:
        # Выбор моделей для бэктестирования
        selected_models = st.multiselect(
            "Select models",
            options=model_returns.columns.tolist(),
            default=model_returns.columns.tolist()[:2] if len(model_returns.columns) > 1 else model_returns.columns.tolist()
        )
        
        # Выбор периода бэктестирования
        col1, col2 = st.columns(2)
        
        # Приведение индекса к datetime при необходимости
        if model_returns.index.dtype == 'object':
            model_returns.index = pd.to_datetime(model_returns.index)
        
        # Получение минимальной и максимальной дат
        min_date = model_returns.index.min()
        max_date = model_returns.index.max()
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        if selected_models and start_date and end_date:
            # Фильтрация данных для выбранного периода
            mask = (model_returns.index >= pd.Timestamp(start_date)) & \
                  (model_returns.index <= pd.Timestamp(end_date))
            backtest_returns = model_returns.loc[mask, selected_models]
            
            if not backtest_returns.empty:
                # Расчет накопленной доходности
                cum_returns = backtest_returns.cumsum()
                
                # График доходности
                fig = px.line(
                    cum_returns, 
                    x=cum_returns.index, 
                    y=cum_returns.columns,
                    title="Backtest Results",
                    labels={"value": "Return", "variable": "Model"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Расчет и отображение метрик доходности
                st.subheader("Performance Metrics")
                
                metrics = calculate_metrics(backtest_returns)
                st.table(metrics)
                
                # Отображение просадок
                st.subheader("Drawdowns")
                
                # Расчет просадок
                drawdowns = {}
                for model in selected_models:
                    cum_ret = cum_returns[model]
                    peak = cum_ret.cummax()
                    drawdown = (cum_ret - peak) / peak
                    drawdowns[model] = drawdown
                
                drawdown_df = pd.DataFrame(drawdowns)
                
                # График просадок
                fig = px.line(
                    drawdown_df,
                    x=drawdown_df.index,
                    y=drawdown_df.columns,
                    title="Drawdowns",
                    labels={"value": "Drawdown", "variable": "Model"}
                )
                fig.update_layout(yaxis_tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected period.")

def render_about():
    """Отображение страницы 'About'"""
    st.header("About")
    
    st.markdown("""
    ## Development of a System for Monitoring and Optimization of Investment Portfolios
    
    This application provides tools for monitoring and optimizing investment portfolios using various algorithms:
    
    ### Features:
    
    - **Dashboard**: Overview of portfolio performance and asset allocation
    - **Portfolio Optimization**: Optimize your portfolio using the Markowitz model
    - **Model Training**: Train new models or select from pretrained models
    - **Model Comparison**: Compare performance of different portfolio optimization models
    - **Backtest Results**: Analyze historical performance of different strategies
    
    ### Optimization Models:
    
    1. **Markowitz Model**: Traditional mean-variance optimization
    2. **Reinforcement Learning Models**:
       - A2C (Advantage Actor-Critic)
       - PPO (Proximal Policy Optimization)
       - DDPG (Deep Deterministic Policy Gradient)
       - SAC (Soft Actor-Critic)
    
    ### Data:
    
    The system uses historical cryptocurrency price data from Binance, including:
    - BNBUSDT, BTCUSDT, CAKEUSDT, ETHUSDT, LTCUSDT, SOLUSDT, STRKUSDT, TONUSDT, USDCUSDT, XRPUSDT, PEPEUSDT, HBARUSDT, APTUSDT, LDOUSDT, JUPUSDT
    
    ### Implementation:
    
    The system is developed using Python with the following key libraries:
    - Streamlit for the web application
    - Pandas and NumPy for data processing
    - Matplotlib and Plotly for visualization
    - SciPy for optimization algorithms
    - FinRL for reinforcement learning models
    """) 