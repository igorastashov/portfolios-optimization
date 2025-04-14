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
from portfolios_optimization.authentication import get_user_portfolios, get_user_portfolio

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

def render_account_dashboard(username, price_data, assets):
    """Отображение страницы аккаунта в стиле Bybit"""
    st.header("Единый торговый аккаунт")
    
    # Получение данных о портфелях пользователя
    from portfolios_optimization.authentication import get_user_portfolios, get_user_portfolio
    
    # Обновление времени
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Последнее обновление: {current_time}")
    
    # Получение портфелей пользователя
    portfolios = get_user_portfolios(username)
    
    if not portfolios:
        st.info("У вас пока нет созданных портфелей. Создайте портфель в разделе 'Мой кабинет'.")
        return
    
    # Объединение всех активов из портфелей пользователя
    all_assets = {}
    total_balance = 0
    
    for portfolio_name in portfolios:
        portfolio_data = get_user_portfolio(username, portfolio_name)
        if portfolio_data and "assets" in portfolio_data:
            # Принимаем, что у нас есть условная сумма в 100,000 USD, которые распределяются по активам
            portfolio_capital = 100000  # Например, 100K USD на портфель
            
            for asset, weight in portfolio_data["assets"].items():
                asset_value = portfolio_capital * weight
                if asset in all_assets:
                    all_assets[asset] += asset_value
                else:
                    all_assets[asset] = asset_value
                total_balance += asset_value
    
    # Основные показатели аккаунта
    col1, col2, col3, col4 = st.columns(4)
    
    # Расчет P&L за сегодня (демонстрационные данные)
    today_pnl = np.random.uniform(-0.02, 0.03) * total_balance
    daily_pnl_percent = today_pnl / total_balance * 100
    
    with col1:
        st.metric("Общий баланс", f"${total_balance:,.2f}")
    
    with col2:
        arrow = "▲" if today_pnl >= 0 else "▼"
        color = "green" if today_pnl >= 0 else "red"
        st.metric("P&L за сегодня", f"{arrow} ${abs(today_pnl):,.2f}", 
                 f"{arrow} {abs(daily_pnl_percent):.2f}%",
                 delta_color="normal" if today_pnl >= 0 else "inverse")
    
    with col3:
        # Предположим, что у нас есть процентное изменение за 7 дней
        week_change = np.random.uniform(-0.05, 0.08) * 100
        arrow_week = "▲" if week_change >= 0 else "▼"
        st.metric("P&L за 7 дней", f"{arrow_week} ${abs(week_change * total_balance / 100):,.2f}", 
                 f"{arrow_week} {abs(week_change):.2f}%",
                 delta_color="normal" if week_change >= 0 else "inverse")
    
    with col4:
        # Маржа (для демонстрации)
        margin_used = total_balance * 0.3  # 30% используется как маржа
        st.metric("Используемая маржа", f"${margin_used:,.2f}", f"{margin_used/total_balance*100:.1f}%")
    
    # Таблица с активами
    st.subheader("Ваши активы")
    
    # Создаем датафрейм с активами
    assets_data = []
    for asset, value in all_assets.items():
        # Получаем текущую цену актива (для демонстрации берем последнюю из исторических данных)
        current_price = price_data[asset].iloc[-1] if asset in price_data.columns else np.nan
        
        # Расчет изменения за 24 часа (для демонстрации)
        price_24h_ago = price_data[asset].iloc[-2] if asset in price_data.columns and len(price_data) > 1 else current_price
        change_24h = (current_price - price_24h_ago) / price_24h_ago * 100 if not np.isnan(current_price) and price_24h_ago != 0 else 0
        
        assets_data.append({
            "Актив": asset,
            "Цена (USD)": current_price,
            "Капитал (USD)": value,
            "Изменение 24ч (%)": change_24h,
            "P&L 24ч (USD)": value * change_24h / 100
        })
    
    # Сортировка по капиталу (по убыванию)
    assets_df = pd.DataFrame(assets_data)
    assets_df = assets_df.sort_values("Капитал (USD)", ascending=False)
    
    # Форматирование таблицы
    formatted_df = assets_df.copy()
    formatted_df["Цена (USD)"] = formatted_df["Цена (USD)"].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else "N/A")
    formatted_df["Капитал (USD)"] = formatted_df["Капитал (USD)"].apply(lambda x: f"${x:,.2f}")
    formatted_df["Изменение 24ч (%)"] = formatted_df["Изменение 24ч (%)"].apply(
        lambda x: f"**+{x:.2f}%**" if x > 0 else (f"**{x:.2f}%**" if x < 0 else "0.00%")
    )
    formatted_df["P&L 24ч (USD)"] = formatted_df["P&L 24ч (USD)"].apply(
        lambda x: f"**+${x:,.2f}**" if x > 0 else (f"**-${abs(x):,.2f}**" if x < 0 else "$0.00")
    )
    
    # Стилизация таблицы с использованием HTML и CSS
    st.markdown(
        formatted_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    
    # График распределения активов
    st.subheader("Распределение капитала")
    fig = px.pie(
        assets_df,
        values="Капитал (USD)",
        names="Актив",
        title="Распределение капитала по активам"
    )
    st.plotly_chart(fig)
    
    # Секция P&L статистики
    st.header("P&L Аккаунта")
    
    # Выбор временного интервала
    interval = st.radio(
        "Выберите интервал:",
        ["7d", "30d", "60d", "180d"],
        horizontal=True
    )
    
    # Определение количества дней для выбранного интервала
    days_map = {"7d": 7, "30d": 30, "60d": 60, "180d": 180}
    days = days_map[interval]
    
    # Генерация демонстрационных данных для P&L
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Генерация случайных данных для P&L
    np.random.seed(42)  # Для воспроизводимости результатов
    daily_pnl_values = np.random.normal(0, total_balance * 0.01, len(date_range))
    
    # Накопленный P&L
    cumulative_pnl = np.cumsum(daily_pnl_values)
    
    # Процент P&L относительно начального капитала
    percentage_pnl = cumulative_pnl / total_balance * 100
    
    pnl_data = pd.DataFrame({
        'Дата': date_range,
        'Суточный P&L': daily_pnl_values,
        'Суммарный P&L': cumulative_pnl,
        'Суммарный P&L (%)': percentage_pnl
    })
    
    # Отображение суммарных метрик
    col1, col2 = st.columns(2)
    
    with col1:
        final_pnl = cumulative_pnl[-1]
        arrow = "▲" if final_pnl >= 0 else "▼"
        st.metric(
            f"Суммарный P&L за {interval}",
            f"{arrow} ${abs(final_pnl):,.2f}",
            delta_color="normal" if final_pnl >= 0 else "inverse"
        )
    
    with col2:
        final_pnl_pct = percentage_pnl[-1]
        arrow = "▲" if final_pnl_pct >= 0 else "▼"
        st.metric(
            f"Суммарный P&L (%) за {interval}",
            f"{arrow} {abs(final_pnl_pct):.2f}%",
            delta_color="normal" if final_pnl_pct >= 0 else "inverse"
        )
    
    # Графики P&L
    # Создаем subplot с двумя графиками
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Суммарный P&L и P&L (%)", "Суточный P&L"),
                        vertical_spacing=0.12,
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    
    # Линия суммарного P&L
    fig.add_trace(
        go.Scatter(
            x=pnl_data['Дата'],
            y=pnl_data['Суммарный P&L'],
            name="Суммарный P&L (USD)",
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Линия процентного P&L на вторичной оси Y
    fig.add_trace(
        go.Scatter(
            x=pnl_data['Дата'],
            y=pnl_data['Суммарный P&L (%)'],
            name="Суммарный P&L (%)",
            line=dict(color='purple', width=2, dash='dot')
        ),
        row=1, col=1,
        secondary_y=True
    )
    
    # Столбчатый график суточного P&L
    daily_colors = ['green' if x >= 0 else 'red' for x in pnl_data['Суточный P&L']]
    
    fig.add_trace(
        go.Bar(
            x=pnl_data['Дата'],
            y=pnl_data['Суточный P&L'],
            name="Суточный P&L",
            marker_color=daily_colors
        ),
        row=2, col=1
    )
    
    # Обновление макета
    fig.update_layout(
        height=700,
        title_text=f"P&L за {interval}",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Обновление подписей осей
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    
    # Отображение графика
    st.plotly_chart(fig, use_container_width=True)

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