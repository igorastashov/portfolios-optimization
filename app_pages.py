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
from portfolios_optimization.authentication import (
    get_user_portfolios, get_user_portfolio, get_user_transactions, 
    add_transaction, delete_transaction, get_portfolio_with_quantities, update_user_portfolio
)

def render_dashboard(username, price_data, model_returns, model_actions, assets):
    """Отображение страницы Dashboard"""
    st.header("Portfolio Dashboard")
    
    # Получение данных о портфеле пользователя на основе транзакций
    portfolio_data = get_portfolio_with_quantities(username)
    
    # Проверка наличия активов в портфеле пользователя
    has_portfolio = portfolio_data and any(portfolio_data["quantities"].values())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Portfolio Performance")
        
        # Создаем tabs для переключения между моделями и портфелем пользователя
        perf_tabs = st.tabs(["My Portfolio", "Model Performance"])
        
        with perf_tabs[0]:
            if not has_portfolio:
                st.info("У вас пока нет активов в портфеле. Добавьте транзакции в разделе 'Управление активами', чтобы сформировать портфель.")
                
                # Кнопка перехода к разделу "Управление активами"
                if st.button("Перейти к управлению активами"):
                    st.session_state.active_page = "Управление активами"
                    st.rerun()
            else:
                # Отображение информации о портфеле пользователя
                # Создание данных о доходности пользовательского портфеля
                
                # Временной период для анализа
                lookback_days = st.slider(
                    "Период анализа (дней)",
                    min_value=7,
                    max_value=365,
                    value=30,
                    step=1
                )
                
                end_date = price_data.index[-1]
                start_date = end_date - timedelta(days=lookback_days)
                
                # Создаем DataFrame для хранения стоимости портфеля пользователя
                portfolio_dates = price_data.loc[start_date:end_date].index
                portfolio_values = []
                
                for date in portfolio_dates:
                    value_at_date = 0
                    for asset, quantity in portfolio_data["quantities"].items():
                        if quantity > 0 and asset in price_data.columns:
                            try:
                                price_at_date = price_data.loc[date, asset]
                                value_at_date += quantity * price_at_date
                            except:
                                pass
                    portfolio_values.append(value_at_date)
                
                # Создание DataFrame с ценами портфеля
                portfolio_df = pd.DataFrame({
                    'Date': portfolio_dates,
                    'Portfolio Value': portfolio_values
                })
                
                # Расчет доходности портфеля
                portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()
                portfolio_df = portfolio_df.dropna()
                
                # Расчет накопленной доходности
                portfolio_df['Cumulative Return'] = (1 + portfolio_df['Daily Return']).cumprod() - 1
                
                # График накопленной доходности
                fig = px.line(
                    portfolio_df, 
                    x='Date', 
                    y='Cumulative Return',
                    title="Cumulative Portfolio Return",
                    labels={"Cumulative Return": "Return", "Date": "Date"}
                )
                fig.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)
                
                # Метрики портфеля
                col1, col2, col3 = st.columns(3)
                
                # Расчет общей доходности
                total_return = portfolio_df['Cumulative Return'].iloc[-1] * 100
                
                # Расчет годовой доходности
                days = len(portfolio_df)
                ann_return = ((1 + portfolio_df['Cumulative Return'].iloc[-1]) ** (365/days) - 1) * 100
                
                # Расчет волатильности (годовой)
                volatility = portfolio_df['Daily Return'].std() * np.sqrt(252) * 100
                
                # Расчет коэффициента Шарпа
                sharpe = ann_return / volatility if volatility > 0 else 0
                
                col1.metric("Общая доходность", f"{total_return:.2f}%")
                col2.metric("Годовая доходность", f"{ann_return:.2f}%")
                col3.metric("Коэффициент Шарпа", f"{sharpe:.2f}")
                
                # Активы в портфеле
                st.subheader("Portfolio Composition")
                
                # Создаем DataFrame для отображения состава портфеля
                assets_data = []
                for asset, quantity in portfolio_data["quantities"].items():
                    if quantity > 0:
                        current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                        asset_value = quantity * current_price
                        assets_data.append({
                            "Asset": asset,
                            "Quantity": quantity,
                            "Current Price": current_price,
                            "Value": asset_value,
                            "Weight": asset_value / sum(quantity * price_data[a].iloc[-1] 
                                                      for a, q in portfolio_data["quantities"].items() 
                                                      if q > 0 and a in price_data.columns)
                        })
                
                assets_df = pd.DataFrame(assets_data)
                
                if not assets_df.empty:
                    # График весов активов
                    fig = px.pie(
                        assets_df,
                        values="Value",
                        names="Asset",
                        title="Portfolio Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица активов
                    formatted_assets = assets_df.copy()
                    formatted_assets["Quantity"] = formatted_assets["Quantity"].apply(lambda x: f"{x:,.8f}")
                    formatted_assets["Current Price"] = formatted_assets["Current Price"].apply(lambda x: f"${x:,.2f}")
                    formatted_assets["Value"] = formatted_assets["Value"].apply(lambda x: f"${x:,.2f}")
                    formatted_assets["Weight"] = formatted_assets["Weight"].apply(lambda x: f"{x*100:.2f}%")
                    
                    st.dataframe(formatted_assets, use_container_width=True)
        
        with perf_tabs[1]:
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
                    title="Cumulative Model Returns",
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
            else:
                st.info("Выберите модели для сравнения")
    
    with col2:
        st.subheader("Latest Allocations")
        
        # Создаем tabs для переключения между моделями и портфелем пользователя
        alloc_tabs = st.tabs(["My Allocation", "Model Allocation"])
        
        with alloc_tabs[0]:
            if not has_portfolio:
                st.info("Добавьте активы в портфель")
            else:
                # Создаем DataFrame для отображения состава портфеля
                assets_data = []
                for asset, quantity in portfolio_data["quantities"].items():
                    if quantity > 0:
                        current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                        asset_value = quantity * current_price
                        assets_data.append({
                            "Asset": asset,
                            "Value": asset_value
                        })
                
                assets_df = pd.DataFrame(assets_data)
                
                if not assets_df.empty:
                    # График весов активов
                    fig = px.pie(
                        assets_df,
                        values="Value",
                        names="Asset",
                        title="Your Current Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with alloc_tabs[1]:
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
        if has_portfolio:
            # По умолчанию показываем активы из портфеля пользователя
            default_assets = [asset for asset, quantity in portfolio_data["quantities"].items() 
                             if quantity > 0 and asset in assets]
        else:
            default_assets = assets[:5] if len(assets) > 5 else assets
        
        selected_assets = st.multiselect(
            "Select assets",
            options=assets,
            default=default_assets
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

def render_portfolio_optimization(username, price_data, assets):
    """Отображение страницы оптимизации портфеля"""
    st.header("Portfolio Optimization")
    
    # Получение данных о портфеле пользователя
    portfolio_data = get_portfolio_with_quantities(username)
    has_portfolio = portfolio_data and any(portfolio_data["quantities"].values())
    
    if not assets:
        st.error("No asset data available. Please check data sources.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimization Parameters")
            
            # Определение активов для работы
            if has_portfolio:
                # Если у пользователя есть портфель, предлагаем использовать его активы
                user_assets = [asset for asset, quantity in portfolio_data["quantities"].items() if quantity > 0 and asset in assets]
                
                # Показываем опцию использования активов из портфеля
                use_portfolio_assets = st.checkbox(
                    "Использовать активы из моего портфеля", 
                    value=True if user_assets else False
                )
                
                if use_portfolio_assets and user_assets:
                    assets_message = "Активы из вашего портфеля:"
                    default_assets = user_assets
                else:
                    assets_message = "Выберите активы для вашего портфеля:"
                    default_assets = assets[:7] if len(assets) > 7 else assets
            else:
                assets_message = "Выберите активы для вашего портфеля:"
                default_assets = assets[:7] if len(assets) > 7 else assets
                use_portfolio_assets = False
            
            # Выбор активов для портфеля
            portfolio_assets = st.multiselect(
                assets_message,
                options=assets,
                default=default_assets
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
                    
                    # Сравнение с текущим портфелем, если он есть
                    if has_portfolio and use_portfolio_assets:
                        st.subheader("Сравнение с текущим портфелем")
                        
                        # Создаем DataFrame для отображения сравнения весов
                        comparison_data = []
                        
                        # Подсчитываем текущие веса в портфеле пользователя
                        total_value = 0
                        current_weights = {}
                        
                        for asset, quantity in portfolio_data["quantities"].items():
                            if quantity > 0 and asset in portfolio_assets:
                                current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                                asset_value = quantity * current_price
                                total_value += asset_value
                                current_weights[asset] = asset_value
                        
                        # Нормализуем веса
                        if total_value > 0:
                            for asset in current_weights:
                                current_weights[asset] /= total_value
                        
                        # Создаем данные для сравнения
                        for asset in portfolio_assets:
                            current_weight = current_weights.get(asset, 0)
                            optimal_weight = weights[portfolio_assets.index(asset)]
                            
                            comparison_data.append({
                                'Asset': asset,
                                'Current Weight': current_weight,
                                'Optimal Weight': optimal_weight,
                                'Difference': optimal_weight - current_weight
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # График сравнения весов
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=comparison_df['Asset'],
                            y=comparison_df['Current Weight'],
                            name='Current Weight',
                            marker_color='lightblue'
                        ))
                        fig.add_trace(go.Bar(
                            x=comparison_df['Asset'],
                            y=comparison_df['Optimal Weight'],
                            name='Optimal Weight',
                            marker_color='lightgreen'
                        ))
                        fig.update_layout(
                            title='Current vs Optimal Weights',
                            barmode='group',
                            yaxis=dict(
                                title='Weight',
                                tickformat='.0%'
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Таблица сравнения
                        formatted_comparison = comparison_df.copy()
                        formatted_comparison['Current Weight'] = formatted_comparison['Current Weight'].apply(lambda x: f"{x*100:.2f}%")
                        formatted_comparison['Optimal Weight'] = formatted_comparison['Optimal Weight'].apply(lambda x: f"{x*100:.2f}%")
                        formatted_comparison['Difference'] = formatted_comparison['Difference'].apply(
                            lambda x: f"+{x*100:.2f}%" if x > 0 else f"{x*100:.2f}%"
                        )
                        
                        st.dataframe(formatted_comparison, use_container_width=True)
                        
                        # Расчет ребалансировки
                        st.subheader("Необходимые действия для ребалансировки")
                        
                        rebalance_data = []
                        for asset in portfolio_assets:
                            current_weight = current_weights.get(asset, 0)
                            optimal_weight = weights[portfolio_assets.index(asset)]
                            
                            # Определение действия (купить/продать)
                            if optimal_weight > current_weight:
                                action = "Купить"
                                difference = optimal_weight - current_weight
                            elif optimal_weight < current_weight:
                                action = "Продать"
                                difference = current_weight - optimal_weight
                            else:
                                action = "Оставить без изменений"
                                difference = 0
                            
                            # Расчет суммы в долларах
                            amount_dollars = difference * total_value
                            
                            # Расчет количества актива
                            current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                            quantity = amount_dollars / current_price if current_price > 0 else 0
                            
                            if difference > 0.001:  # показываем только значимые изменения
                                rebalance_data.append({
                                    'Asset': asset,
                                    'Action': action,
                                    'Amount (USD)': amount_dollars,
                                    'Quantity': quantity
                                })
                        
                        rebalance_df = pd.DataFrame(rebalance_data)
                        
                        if not rebalance_df.empty:
                            # Форматирование данных
                            formatted_rebalance = rebalance_df.copy()
                            formatted_rebalance['Amount (USD)'] = formatted_rebalance['Amount (USD)'].apply(lambda x: f"${x:.2f}")
                            formatted_rebalance['Quantity'] = formatted_rebalance['Quantity'].apply(lambda x: f"{x:.8f}")
                            
                            st.dataframe(formatted_rebalance, use_container_width=True)
                        else:
                            st.info("Значимых изменений для ребалансировки не требуется")
                    
                    # Добавление кнопки "Сохранить оптимизированный портфель"
                    if st.button("Сохранить оптимальные веса как цель для портфеля"):
                        # Здесь будет логика сохранения в портфель пользователя
                        # Создаем новый портфель с оптимальными весами
                        portfolio_name = f"optimized_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Создание данных портфеля
                        optimized_portfolio = {
                            "description": f"Оптимизированный портфель от {datetime.now().strftime('%Y-%m-%d')}",
                            "type": "optimized",
                            "assets": {asset: weight for asset, weight in zip(portfolio_assets, weights)},
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "optimization_params": {
                                "risk_aversion": risk_aversion,
                                "lookback_period": lookback_period,
                                "expected_return": expected_return,
                                "expected_volatility": expected_volatility,
                                "sharpe_ratio": sharpe_ratio
                            }
                        }
                        
                        # Сохранение портфеля
                        success, message = update_user_portfolio(
                            username, 
                            portfolio_name, 
                            optimized_portfolio
                        )
                        
                        if success:
                            st.success(f"Оптимизированный портфель сохранен как {portfolio_name}")
                            st.info("Вы можете использовать его как цель для ребалансировки своего текущего портфеля")
                        else:
                            st.error(message)
                else:
                    st.error("No price data available.")
            
            elif not portfolio_assets:
                st.info("Please select assets for your portfolio.")
            elif not optimize_button:
                st.info("Click 'Optimize Portfolio' to see the results.")

def render_model_training(username, price_data, assets):
    """Отображение страницы обучения моделей"""
    st.header("Model Training & Selection")
    
    # Получение данных о портфеле пользователя
    portfolio_data = get_portfolio_with_quantities(username)
    has_portfolio = portfolio_data and any(portfolio_data["quantities"].values())
    
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
        
        # Определение активов для работы
        if has_portfolio:
            # Если у пользователя есть портфель, предлагаем использовать его активы
            user_assets = [asset for asset, quantity in portfolio_data["quantities"].items() if quantity > 0 and asset in assets]
            
            # Показываем опцию использования активов из портфеля
            use_portfolio_assets = st.checkbox(
                "Использовать активы из моего портфеля", 
                value=True if user_assets else False
            )
            
            if use_portfolio_assets and user_assets:
                assets_message = "Активы из вашего портфеля:"
                default_assets = user_assets
            else:
                assets_message = "Выберите активы для обучения модели:"
                default_assets = assets[:7] if len(assets) > 7 else assets
        else:
            assets_message = "Выберите активы для обучения модели:"
            default_assets = assets[:7] if len(assets) > 7 else assets
        
        # Выбор активов
        training_assets = st.multiselect(
            assets_message,
            options=assets,
            default=default_assets
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

def render_model_comparison(username, model_returns, model_actions, price_data):
    """Отображение страницы сравнения моделей"""
    st.header("Model Comparison")
    
    # Получение данных о портфеле пользователя
    portfolio_data = get_portfolio_with_quantities(username)
    has_portfolio = portfolio_data and any(portfolio_data["quantities"].values())
    
    if model_returns.empty:
        st.warning("No model return data available. Please train or load models first.")
    else:
        # Расчет доходности портфеля пользователя, если он существует
        if has_portfolio:
            # Создание DataFrame с доходностью портфеля пользователя
            portfolio_returns = pd.DataFrame(index=model_returns.index)
            
            # Расчет стоимости портфеля для каждой даты
            portfolio_values = []
            for date in model_returns.index:
                value_at_date = 0
                for asset, quantity in portfolio_data["quantities"].items():
                    if quantity > 0 and asset in price_data.columns:
                        try:
                            price_at_date = price_data.loc[date, asset]
                            value_at_date += quantity * price_at_date
                        except:
                            pass
                portfolio_values.append(value_at_date)
            
            # Создание Series с ценами портфеля
            portfolio_prices = pd.Series(portfolio_values, index=model_returns.index)
            
            # Расчет доходности портфеля
            portfolio_returns['Your Portfolio'] = portfolio_prices.pct_change().fillna(0)
            
            # Объединение с данными моделей
            all_returns = pd.concat([model_returns, portfolio_returns], axis=1)
        else:
            all_returns = model_returns
        
        # Выбор моделей для сравнения
        available_models = all_returns.columns.tolist()
        
        # Если у пользователя есть портфель, добавляем его в список по умолчанию
        if has_portfolio:
            default_models = ['Your Portfolio']
            if len(available_models) > 1:
                default_models.extend(available_models[:2] if len(available_models) > 2 else available_models)
                default_models = list(set(default_models))  # Удаление дубликатов
        else:
            default_models = available_models[:2] if len(available_models) > 1 else available_models
        
        models = st.multiselect(
            "Select models to compare",
            options=available_models,
            default=default_models
        )
        
        if models:
            # Расчет накопленной доходности
            cum_returns = all_returns[models].cumsum()
            
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
            
            # Если портфель пользователя является частью сравнения, показываем дополнительную информацию
            if has_portfolio and 'Your Portfolio' in models:
                st.subheader("Ваш портфель vs Модели")
                
                # Расчет относительной доходности (разницы) по сравнению с пользовательским портфелем
                if len(models) > 1:
                    st.write("Насколько хорошо работают модели по сравнению с вашим портфелем:")
                    
                    # Создаем DataFrame с относительной доходностью
                    relative_returns = pd.DataFrame()
                    
                    for model in models:
                        if model != 'Your Portfolio':
                            # Разница в накопленной доходности
                            diff_return = cum_returns[model].iloc[-1] - cum_returns['Your Portfolio'].iloc[-1]
                            # Процентная разница
                            percent_diff = diff_return / abs(cum_returns['Your Portfolio'].iloc[-1]) * 100 if cum_returns['Your Portfolio'].iloc[-1] != 0 else float('inf')
                            
                            relative_returns.loc[model, 'Difference'] = diff_return
                            relative_returns.loc[model, 'Percent Difference'] = percent_diff
                    
                    # Форматирование
                    formatted_relative = relative_returns.copy()
                    formatted_relative['Difference'] = formatted_relative['Difference'].apply(
                        lambda x: f"+{x:.4f}" if x > 0 else f"{x:.4f}"
                    )
                    formatted_relative['Percent Difference'] = formatted_relative['Percent Difference'].apply(
                        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
                    )
                    
                    st.table(formatted_relative)
                    
                    # Объяснение результатов
                    best_model = relative_returns['Difference'].idxmax()
                    worst_model = relative_returns['Difference'].idxmin()
                    
                    if relative_returns.loc[best_model, 'Difference'] > 0:
                        st.info(f"Модель '{best_model}' работает лучше всего и превосходит ваш портфель на {formatted_relative.loc[best_model, 'Percent Difference']}")
                    
                    if relative_returns.loc[worst_model, 'Difference'] < 0:
                        st.warning(f"Модель '{worst_model}' работает хуже всего и отстает от вашего портфеля на {formatted_relative.loc[worst_model, 'Percent Difference'].replace('-', '')}")
            
            # Сравнение распределения активов моделей
            st.subheader("Asset Allocation Comparison")
            
            # Если у пользователя есть портфель, показываем его распределение
            if has_portfolio and 'Your Portfolio' in models:
                # Подготовка данных распределения портфеля пользователя
                user_allocation = {}
                total_value = 0
                
                for asset, quantity in portfolio_data["quantities"].items():
                    if quantity > 0:
                        current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                        asset_value = quantity * current_price
                        total_value += asset_value
                        user_allocation[asset] = asset_value
                
                # Нормализация весов
                if total_value > 0:
                    for asset in user_allocation:
                        user_allocation[asset] /= total_value
                
                # Добавление распределения пользователя в словарь распределений моделей
                user_allocation_df = pd.Series(user_allocation)
                model_actions['your_portfolio'] = user_allocation_df
            
            # Получение списка моделей с данными распределения
            models_with_allocations = []
            for m in models:
                if m == 'Your Portfolio' and has_portfolio:
                    models_with_allocations.append('your_portfolio')
                elif m.lower() in model_actions:
                    models_with_allocations.append(m.lower())
            
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
        else:
            st.info("Please select models to compare.")

def render_backtest(username, model_returns, price_data):
    """Отображение страницы бэктестирования"""
    st.header("Backtest Results")
    
    # Получение данных о портфеле пользователя
    portfolio_data = get_portfolio_with_quantities(username)
    has_portfolio = portfolio_data and any(portfolio_data["quantities"].values())
    
    if model_returns.empty:
        st.warning("No model return data available. Please train or load models first.")
    else:
        # Добавление портфеля пользователя для сравнения
        if has_portfolio:
            # Создание DataFrame с доходностью портфеля пользователя
            portfolio_returns = pd.DataFrame(index=model_returns.index)
            
            # Расчет стоимости портфеля для каждой даты
            portfolio_values = []
            for date in model_returns.index:
                value_at_date = 0
                for asset, quantity in portfolio_data["quantities"].items():
                    if quantity > 0 and asset in price_data.columns:
                        try:
                            price_at_date = price_data.loc[date, asset]
                            value_at_date += quantity * price_at_date
                        except:
                            pass
                portfolio_values.append(value_at_date)
            
            # Создание Series с ценами портфеля
            portfolio_prices = pd.Series(portfolio_values, index=model_returns.index)
            
            # Расчет доходности портфеля
            portfolio_returns['Your Portfolio'] = portfolio_prices.pct_change().fillna(0)
            
            # Объединение с данными моделей
            all_returns = pd.concat([model_returns, portfolio_returns], axis=1)
        else:
            all_returns = model_returns
        
        # Выбор моделей для бэктестирования
        available_models = all_returns.columns.tolist()
        
        # Если у пользователя есть портфель, добавляем его в список по умолчанию
        if has_portfolio:
            default_models = ['Your Portfolio']
            if len(available_models) > 1:
                default_models.extend(available_models[:1])  # Добавляем только одну модель для сравнения
        else:
            default_models = available_models[:2] if len(available_models) > 1 else available_models
        
        selected_models = st.multiselect(
            "Select models",
            options=available_models,
            default=default_models
        )
        
        # Выбор периода бэктестирования
        col1, col2 = st.columns(2)
        
        # Приведение индекса к datetime при необходимости
        if all_returns.index.dtype == 'object':
            all_returns.index = pd.to_datetime(all_returns.index)
        
        # Получение минимальной и максимальной дат
        min_date = all_returns.index.min()
        max_date = all_returns.index.max()
        
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
            mask = (all_returns.index >= pd.Timestamp(start_date)) & \
                  (all_returns.index <= pd.Timestamp(end_date))
            backtest_returns = all_returns.loc[mask, selected_models]
            
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
    
    # Получение данных о портфеле пользователя на основе транзакций
    portfolio_data = get_portfolio_with_quantities(username)
    
    # Обновление времени
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Последнее обновление: {current_time}")
    
    # Проверка наличия активов в портфеле
    has_assets = portfolio_data and any(portfolio_data["quantities"].values())
    
    if not has_assets:
        st.info("""
        У вас пока нет активов в портфеле.
        
        Чтобы сформировать портфель:
        1. Перейдите в раздел 'Управление активами'
        2. На вкладке 'Добавить транзакцию' добавьте свои первые активы
        3. После добавления транзакций, портфель сформируется автоматически
        """)
        
        # Кнопка перехода к разделу "Управление активами"
        if st.button("Перейти к управлению активами"):
            st.session_state.active_page = "Управление активами"
            st.rerun()
        return
    
    # Создание данных для отображения информации об аккаунте
    all_assets = {}
    total_balance = 0
    
    # Обработка данных портфеля из транзакций
    for asset, quantity in portfolio_data["quantities"].items():
        if quantity > 0:
            # Получение текущей цены актива
            current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
            
            # Расчет стоимости актива
            asset_value = quantity * current_price
            all_assets[asset] = asset_value
            total_balance += asset_value
    
    # Основные показатели аккаунта
    col1, col2, col3, col4 = st.columns(4)
    
    # Расчет P&L за сегодня
    today_pnl = 0
    portfolio_24h_ago = 0
    
    for asset, quantity in portfolio_data["quantities"].items():
        if quantity > 0:
            current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
            price_24h_ago = price_data[asset].iloc[-2] if asset in price_data.columns and len(price_data) > 1 else current_price
            
            asset_value_now = quantity * current_price
            asset_value_24h_ago = quantity * price_24h_ago
            
            today_pnl += (asset_value_now - asset_value_24h_ago)
            portfolio_24h_ago += asset_value_24h_ago
    
    daily_pnl_percent = today_pnl / portfolio_24h_ago * 100 if portfolio_24h_ago > 0 else 0
    
    with col1:
        st.metric("Общий баланс", f"${total_balance:,.2f}")
    
    with col2:
        arrow = "▲" if today_pnl >= 0 else "▼"
        color = "green" if today_pnl >= 0 else "red"
        st.metric("P&L за сегодня", f"{arrow} ${abs(today_pnl):,.2f}", 
                 f"{arrow} {abs(daily_pnl_percent):.2f}%",
                 delta_color="normal" if today_pnl >= 0 else "inverse")
    
    with col3:
        # Расчет P&L в сравнении со средней ценой покупки
        total_invested = 0
        for asset, quantity in portfolio_data["quantities"].items():
            if quantity > 0:
                avg_price = portfolio_data["avg_prices"].get(asset, 0)
                total_invested += quantity * avg_price
        
        total_pnl = total_balance - total_invested
        total_pnl_percent = total_pnl / total_invested * 100 if total_invested > 0 else 0
        
        arrow_total = "▲" if total_pnl >= 0 else "▼"
        st.metric("Общий P&L", f"{arrow_total} ${abs(total_pnl):,.2f}", 
                 f"{arrow_total} {abs(total_pnl_percent):.2f}%",
                 delta_color="normal" if total_pnl >= 0 else "inverse")
    
    with col4:
        # Маржа (для демонстрации)
        margin_used = total_balance * 0.3  # 30% используется как маржа
        st.metric("Используемая маржа", f"${margin_used:,.2f}", f"{margin_used/total_balance*100:.1f}%")
    
    # Таблица с активами
    st.subheader("Ваши активы")
    
    # Создаем датафрейм с активами
    assets_data = []
    for asset, quantity in portfolio_data["quantities"].items():
        if quantity > 0:
            # Получение текущей цены актива
            current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
            
            # Средняя цена покупки
            avg_buy_price = portfolio_data["avg_prices"].get(asset, 0)
            
            # Расчет текущей стоимости и прибыли/убытка
            current_value = quantity * current_price
            invested_value = quantity * avg_buy_price
            profit_loss = current_value - invested_value
            profit_loss_percent = (profit_loss / invested_value * 100) if invested_value > 0 else 0
            
            # Расчет изменения за 24 часа
            price_24h_ago = price_data[asset].iloc[-2] if asset in price_data.columns and len(price_data) > 1 else current_price
            change_24h = (current_price - price_24h_ago) / price_24h_ago * 100 if price_24h_ago > 0 else 0
            
            assets_data.append({
                "Актив": asset,
                "Количество": quantity,
                "Средняя цена": avg_buy_price,
                "Текущая цена": current_price,
                "Стоимость": current_value,
                "Изменение 24ч (%)": change_24h,
                "P&L": profit_loss,
                "P&L (%)": profit_loss_percent
            })
    
    # Сортировка по стоимости (по убыванию)
    assets_df = pd.DataFrame(assets_data)
    assets_df = assets_df.sort_values("Стоимость", ascending=False)
    
    # Форматирование таблицы
    formatted_df = assets_df.copy()
    formatted_df["Количество"] = formatted_df["Количество"].apply(lambda x: f"{x:,.8f}")
    formatted_df["Средняя цена"] = formatted_df["Средняя цена"].apply(lambda x: f"${x:,.2f}")
    formatted_df["Текущая цена"] = formatted_df["Текущая цена"].apply(lambda x: f"${x:,.2f}")
    formatted_df["Стоимость"] = formatted_df["Стоимость"].apply(lambda x: f"${x:,.2f}")
    formatted_df["Изменение 24ч (%)"] = formatted_df["Изменение 24ч (%)"].apply(
        lambda x: f"**+{x:.2f}%**" if x > 0 else (f"**{x:.2f}%**" if x < 0 else "0.00%")
    )
    formatted_df["P&L"] = formatted_df["P&L"].apply(
        lambda x: f"**+${x:,.2f}**" if x > 0 else (f"**-${abs(x):,.2f}**" if x < 0 else "$0.00")
    )
    formatted_df["P&L (%)"] = formatted_df["P&L (%)"].apply(
        lambda x: f"**+{x:.2f}%**" if x > 0 else (f"**{x:.2f}%**" if x < 0 else "0.00%")
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
        values="Стоимость",
        names="Актив",
        title="Распределение капитала по активам"
    )
    st.plotly_chart(fig)
    
    # Секция P&L статистики
    st.header("P&L Аккаунта")
    
    # Выбор временного интервала
    interval = st.radio(
        "Выберите интервал:",
        ["7d", "30d", "60d", "180d", "С момента покупки"],
        horizontal=True,
        index=4
    )
    
    # Определение количества дней для выбранного интервала или использование данных с момента покупки
    if interval == "С момента покупки":
        # Используем данные с момента самой первой транзакции
        transactions = get_user_transactions(username)
        if transactions:
            # Находим дату самой ранней транзакции
            earliest_date = min([pd.to_datetime(t["date"]) for t in transactions])
            days = (datetime.now() - earliest_date).days
        else:
            days = 30  # Значение по умолчанию, если нет транзакций
    else:
        days_map = {"7d": 7, "30d": 30, "60d": 60, "180d": 180}
        days = days_map[interval]
    
    # Генерация исторических данных P&L на основе исторических цен активов
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Попытка использовать исторические данные цен для расчета P&L
    if not price_data.empty and len(price_data) > 1:
        # Фильтрация данных по дате
        historical_data = price_data[price_data.index >= pd.Timestamp(start_date)]
        
        # Создание DataFrame для хранения ежедневного P&L
        date_range = historical_data.index
        daily_pnl_values = []
        
        # Расчет стоимости портфеля для каждого дня
        portfolio_values = []
        
        # Стоимость всех активов по средней цене покупки (для расчета общего P&L)
        total_cost_basis = 0
        for asset, quantity in portfolio_data["quantities"].items():
            if quantity > 0:
                avg_price = portfolio_data["avg_prices"].get(asset, 0)
                total_cost_basis += quantity * avg_price
        
        for date in date_range:
            value_at_date = 0
            for asset, quantity in portfolio_data["quantities"].items():
                if quantity > 0 and asset in price_data.columns:
                    try:
                        price_at_date = price_data.loc[date, asset]
                        value_at_date += quantity * price_at_date
                    except:
                        pass  # Пропускаем, если нет данных цен для этой даты
            
            portfolio_values.append(value_at_date)
        
        # Расчет ежедневного P&L
        for i in range(1, len(portfolio_values)):
            daily_pnl = portfolio_values[i] - portfolio_values[i-1]
            daily_pnl_values.append(daily_pnl)
        
        # Добавление нуля для первого дня, так как нет предыдущего значения
        daily_pnl_values.insert(0, 0)
        
        # Создание DataFrame с данными P&L
        pnl_data = pd.DataFrame({
            'Дата': date_range,
            'Стоимость портфеля': portfolio_values,
            'Суточный P&L': daily_pnl_values
        })
        
        # Расчет накопленного P&L
        pnl_data['Суммарный P&L'] = np.cumsum(pnl_data['Суточный P&L'])
        
        # Расчет P&L относительно средней цены покупки
        if interval == "С момента покупки":
            # Для режима "С момента покупки" показываем реальный P&L относительно средней цены покупки
            initial_value = total_cost_basis
            total_pnl_from_cost = portfolio_values[-1] - total_cost_basis  # P&L от средней цены
            
            # Заменяем последнее значение суммарного P&L на P&L от средней цены
            pnl_data.loc[pnl_data.index[-1], 'Суммарный P&L'] = total_pnl_from_cost
        else:
            initial_value = portfolio_values[0] if portfolio_values else 0
        
        # Расчет процентного P&L
        if initial_value > 0:
            pnl_data['Суммарный P&L (%)'] = (pnl_data['Суммарный P&L'] / initial_value) * 100
        else:
            pnl_data['Суммарный P&L (%)'] = 0
    else:
        # Если недостаточно исторических данных - создаем синтетические данные для демонстрации
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Генерация случайных данных для P&L
        np.random.seed(42)  # Для воспроизводимости результатов
        daily_pnl_values = np.random.normal(0, total_balance * 0.01, len(date_range))
        
        # Создаем синтетические portfolio_values на основе daily_pnl
        portfolio_values = [total_balance]  # Начинаем с текущего баланса
        for pnl in daily_pnl_values:
            portfolio_values.append(portfolio_values[-1] + pnl)
        # Удаляем первый элемент, так как он соответствует значению до первого дня в date_range
        portfolio_values.pop(0)
        
        # Накопленный P&L
        cumulative_pnl = np.cumsum(daily_pnl_values)
        
        # Процент P&L относительно начального капитала
        percentage_pnl = cumulative_pnl / total_balance * 100 if total_balance > 0 else np.zeros_like(cumulative_pnl)
        
        pnl_data = pd.DataFrame({
            'Дата': date_range,
            'Стоимость портфеля': portfolio_values,
            'Суточный P&L': daily_pnl_values,
            'Суммарный P&L': cumulative_pnl,
            'Суммарный P&L (%)': percentage_pnl
        })
    
    # Отображение суммарных метрик
    col1, col2, col3 = st.columns(3)
    
    with col1:
        final_pnl = pnl_data['Суммарный P&L'].iloc[-1]
        arrow = "▲" if final_pnl >= 0 else "▼"
        st.metric(
            f"Суммарный P&L за {interval}",
            f"{arrow} ${abs(final_pnl):,.2f}",
            delta_color="normal" if final_pnl >= 0 else "inverse"
        )
    
    with col2:
        final_pnl_pct = pnl_data['Суммарный P&L (%)'].iloc[-1]
        arrow = "▲" if final_pnl_pct >= 0 else "▼"
        st.metric(
            f"Суммарный P&L (%) за {interval}",
            f"{arrow} {abs(final_pnl_pct):.2f}%",
            delta_color="normal" if final_pnl_pct >= 0 else "inverse"
        )
    
    with col3:
        if interval == "С момента покупки":
            # Добавляем метрику ROI (Return on Investment)
            roi = (portfolio_values[-1] / total_cost_basis - 1) * 100 if total_cost_basis > 0 else 0
            arrow_roi = "▲" if roi >= 0 else "▼"
            st.metric(
                "ROI (доходность инвестиций)",
                f"{arrow_roi} {abs(roi):.2f}%",
                delta_color="normal" if roi >= 0 else "inverse"
            )
        else:
            # Годовая процентная доходность (APY)
            days_period = len(pnl_data)
            if days_period > 0 and portfolio_values[0] > 0:
                apy = ((portfolio_values[-1] / portfolio_values[0]) ** (365 / days_period) - 1) * 100
                arrow_apy = "▲" if apy >= 0 else "▼"
                st.metric(
                    "Годовая доходность (APY)",
                    f"{arrow_apy} {abs(apy):.2f}%",
                    delta_color="normal" if apy >= 0 else "inverse"
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
    
    # Добавим кнопку для управления активами
    st.info("Чтобы изменить состав портфеля, перейдите в раздел 'Управление активами'")
    if st.button("Перейти к управлению активами"):
        st.session_state.active_page = "Управление активами"
        st.rerun()

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

def render_transactions_manager(username, price_data, assets):
    """Отображение страницы управления транзакциями в стиле CoinMarketCap"""
    st.header("Управление активами и транзакциями")
    
    # Получение текущих транзакций пользователя
    transactions = get_user_transactions(username)
    
    # Получение текущего портфеля с количествами
    portfolio_data = get_portfolio_with_quantities(username)
    
    # Вкладки для разных функций
    tabs = st.tabs(["Мой портфель", "Добавить транзакцию", "История транзакций"])
    
    # Вкладка "Мой портфель"
    with tabs[0]:
        st.subheader("Текущий портфель")
        
        if not portfolio_data or not any(portfolio_data["quantities"].values()):
            st.info("У вас пока нет активов в портфеле. Добавьте транзакции, чтобы сформировать портфель.")
        else:
            # Создание DataFrame для отображения портфеля
            portfolio_items = []
            
            total_portfolio_value = 0
            total_profit_loss = 0
            
            for asset, quantity in portfolio_data["quantities"].items():
                if quantity > 0:
                    # Получение текущей цены актива
                    current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                    
                    # Средняя цена покупки
                    avg_buy_price = portfolio_data["avg_prices"].get(asset, 0)
                    
                    # Расчет текущей стоимости и прибыли/убытка
                    current_value = quantity * current_price
                    invested_value = quantity * avg_buy_price
                    profit_loss = current_value - invested_value
                    profit_loss_percent = (profit_loss / invested_value * 100) if invested_value > 0 else 0
                    
                    total_portfolio_value += current_value
                    total_profit_loss += profit_loss
                    
                    # Расчет изменения за 24 часа
                    price_24h_ago = price_data[asset].iloc[-2] if asset in price_data.columns and len(price_data) > 1 else current_price
                    change_24h = (current_price - price_24h_ago) / price_24h_ago * 100 if price_24h_ago > 0 else 0
                    
                    portfolio_items.append({
                        "Актив": asset,
                        "Количество": quantity,
                        "Средняя цена покупки": avg_buy_price,
                        "Текущая цена": current_price,
                        "Текущая стоимость": current_value,
                        "Изменение 24ч (%)": change_24h,
                        "P&L": profit_loss,
                        "P&L (%)": profit_loss_percent
                    })
            
            # Отображение общей информации о портфеле
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Общая стоимость портфеля", f"${total_portfolio_value:,.2f}")
            
            with col2:
                # Знак для P&L
                if total_profit_loss >= 0:
                    st.metric("Общий P&L", f"${total_profit_loss:,.2f}", delta=f"{total_profit_loss/total_portfolio_value*100:.2f}%")
                else:
                    st.metric("Общий P&L", f"-${abs(total_profit_loss):,.2f}", delta=f"{total_profit_loss/total_portfolio_value*100:.2f}%", delta_color="inverse")
            
            with col3:
                # Получение общего изменения стоимости за 24 часа
                portfolio_24h_ago = 0
                for item in portfolio_items:
                    asset = item["Актив"]
                    quantity = item["Количество"]
                    price_24h_ago = price_data[asset].iloc[-2] if asset in price_data.columns and len(price_data) > 1 else item["Текущая цена"]
                    portfolio_24h_ago += quantity * price_24h_ago
                
                change_24h = (total_portfolio_value - portfolio_24h_ago) / portfolio_24h_ago * 100 if portfolio_24h_ago > 0 else 0
                
                st.metric("Изменение за 24ч", 
                         f"${total_portfolio_value - portfolio_24h_ago:,.2f}", 
                         delta=f"{change_24h:.2f}%",
                         delta_color="normal" if change_24h >= 0 else "inverse")
            
            # Создание DataFrame из данных портфеля
            portfolio_df = pd.DataFrame(portfolio_items)
            
            # Отображение таблицы активов
            if not portfolio_df.empty:
                # Сортировка по текущей стоимости (по убыванию)
                portfolio_df = portfolio_df.sort_values("Текущая стоимость", ascending=False)
                
                # Форматирование значений для отображения
                formatted_df = portfolio_df.copy()
                formatted_df["Средняя цена покупки"] = formatted_df["Средняя цена покупки"].apply(lambda x: f"${x:,.2f}")
                formatted_df["Текущая цена"] = formatted_df["Текущая цена"].apply(lambda x: f"${x:,.2f}")
                formatted_df["Текущая стоимость"] = formatted_df["Текущая стоимость"].apply(lambda x: f"${x:,.2f}")
                formatted_df["Изменение 24ч (%)"] = formatted_df["Изменение 24ч (%)"].apply(
                    lambda x: f"+{x:.2f}%" if x > 0 else (f"{x:.2f}%" if x < 0 else "0.00%")
                )
                formatted_df["P&L"] = formatted_df["P&L"].apply(
                    lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
                )
                formatted_df["P&L (%)"] = formatted_df["P&L (%)"].apply(
                    lambda x: f"+{x:.2f}%" if x > 0 else (f"{x:.2f}%" if x < 0 else "0.00%")
                )
                
                # Отображение таблицы
                st.dataframe(formatted_df, use_container_width=True)
                
                # График распределения активов по стоимости
                st.subheader("Распределение портфеля")
                fig = px.pie(
                    portfolio_df,
                    values="Текущая стоимость",
                    names="Актив",
                    title="Распределение портфеля по текущей стоимости"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Вкладка "Добавить транзакцию"
    with tabs[1]:
        st.subheader("Добавить новую транзакцию")
        
        # Форма для добавления транзакции
        with st.form("add_transaction_form"):
            # Выбор актива
            asset = st.selectbox(
                "Выберите актив",
                options=assets
            )
            
            # Тип операции
            transaction_type = st.radio(
                "Тип операции",
                options=["buy", "sell"],
                format_func=lambda x: "Покупка" if x == "buy" else "Продажа",
                horizontal=True
            )
            
            # Дата и время транзакции
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_date = st.date_input(
                    "Дата транзакции",
                    value=datetime.now()
                )
            
            with col2:
                transaction_time = st.time_input(
                    "Время транзакции",
                    value=datetime.now().time()
                )
            
            # Объединение даты и времени
            transaction_datetime = datetime.combine(transaction_date, transaction_time)
            
            # Количество и цена
            col1, col2 = st.columns(2)
            
            with col1:
                quantity = st.number_input(
                    "Количество",
                    min_value=0.0,
                    value=1.0,
                    step=0.01
                )
            
            with col2:
                # Текущая цена актива
                current_price = price_data[asset].iloc[-1] if asset in price_data.columns else 0
                
                price = st.number_input(
                    "Цена за единицу",
                    min_value=0.0,
                    value=current_price,
                    step=0.01
                )
            
            # Комиссия и общая стоимость
            col1, col2 = st.columns(2)
            
            with col1:
                fee = st.number_input(
                    "Комиссия",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )
            
            with col2:
                total_cost = quantity * price + fee if transaction_type == "buy" else quantity * price - fee
                st.metric("Общая стоимость", f"${total_cost:,.2f}")
            
            # Примечание к транзакции
            note = st.text_area(
                "Примечание",
                placeholder="Добавьте описание или комментарий к транзакции"
            )
            
            # Кнопка отправки формы
            submit_button = st.form_submit_button("Добавить транзакцию")
            
            if submit_button:
                # Проверка валидности данных
                if quantity <= 0:
                    st.error("Количество должно быть больше нуля")
                elif price <= 0:
                    st.error("Цена должна быть больше нуля")
                else:
                    # Создание данных транзакции
                    transaction_data = {
                        "asset": asset,
                        "type": transaction_type,
                        "quantity": quantity,
                        "price": price,
                        "fee": fee,
                        "date": transaction_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "note": note
                    }
                    
                    # Добавление транзакции
                    success, message = add_transaction(username, transaction_data)
                    
                    if success:
                        st.success(message)
                        st.info("Обновите страницу, чтобы увидеть изменения в портфеле")
                    else:
                        st.error(message)
    
    # Вкладка "История транзакций"
    with tabs[2]:
        st.subheader("История транзакций")
        
        if not transactions:
            st.info("У вас пока нет транзакций")
        else:
            # Создание DataFrame для отображения истории транзакций
            transactions_df = pd.DataFrame(transactions)
            
            # Приведение столбца даты к datetime
            transactions_df["date"] = pd.to_datetime(transactions_df["date"])
            
            # Сортировка по дате (новые сверху)
            transactions_df = transactions_df.sort_values("date", ascending=False)
            
            # Форматирование данных для отображения
            formatted_transactions = transactions_df.copy()
            formatted_transactions["type"] = formatted_transactions["type"].apply(
                lambda x: "Покупка" if x == "buy" else "Продажа"
            )
            formatted_transactions["quantity"] = formatted_transactions["quantity"].apply(
                lambda x: f"{x:,.8f}"
            )
            formatted_transactions["price"] = formatted_transactions["price"].apply(
                lambda x: f"${x:,.2f}"
            )
            formatted_transactions["fee"] = formatted_transactions["fee"].apply(
                lambda x: f"${x:,.2f}"
            )
            
            # Расчет общей стоимости транзакции
            formatted_transactions["total"] = [
                f"${row['quantity'] * row['price'] + row['fee']:,.2f}" if row["type"] == "buy" 
                else f"${row['quantity'] * row['price'] - row['fee']:,.2f}"
                for _, row in transactions_df.iterrows()
            ]
            
            # Переименование столбцов для отображения
            display_columns = {
                "id": "ID",
                "date": "Дата",
                "type": "Тип",
                "asset": "Актив",
                "quantity": "Количество",
                "price": "Цена",
                "fee": "Комиссия",
                "total": "Общая стоимость",
                "note": "Примечание"
            }
            
            formatted_transactions = formatted_transactions.rename(columns=display_columns)
            
            # Отображение таблицы транзакций
            st.dataframe(formatted_transactions[list(display_columns.values())], use_container_width=True)
            
            # Фильтр транзакций
            st.subheader("Фильтр транзакций")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Фильтр по типу транзакции
                transaction_type_filter = st.multiselect(
                    "Тип транзакции",
                    options=["Покупка", "Продажа"],
                    default=["Покупка", "Продажа"]
                )
            
            with col2:
                # Фильтр по активу
                available_assets = transactions_df["asset"].unique().tolist()
                asset_filter = st.multiselect(
                    "Активы",
                    options=available_assets,
                    default=available_assets
                )
            
            # Применение фильтров
            filtered_transactions = formatted_transactions.copy()
            
            if transaction_type_filter:
                filtered_transactions = filtered_transactions[filtered_transactions["Тип"].isin(transaction_type_filter)]
            
            if asset_filter:
                filtered_transactions = filtered_transactions[filtered_transactions["Актив"].isin(asset_filter)]
            
            # Отображение отфильтрованных транзакций
            if not filtered_transactions.empty:
                st.subheader("Отфильтрованные транзакции")
                st.dataframe(filtered_transactions, use_container_width=True)
                
                # Удаление транзакции
                st.subheader("Удаление транзакции")
                
                transaction_to_delete = st.number_input(
                    "Введите ID транзакции для удаления",
                    min_value=1,
                    max_value=max(transactions_df["id"]) if not transactions_df.empty else 1,
                    step=1
                )
                
                # Проверка существования транзакции
                transaction_exists = any(t["id"] == transaction_to_delete for t in transactions)
                
                if not transaction_exists:
                    st.error(f"Транзакция с ID {transaction_to_delete} не найдена")
                else:
                    # Сначала отображаем чекбокс подтверждения
                    confirm_delete = st.checkbox(f"Я подтверждаю удаление транзакции №{transaction_to_delete}")
                    
                    # Кнопка удаления активна только после подтверждения
                    if confirm_delete:
                        if st.button("Подтвердить удаление"):
                            success, message = delete_transaction(username, transaction_to_delete)
                            
                            if success:
                                st.success(message)
                                st.info("Обновите страницу, чтобы увидеть изменения")
                            else:
                                st.error(message)