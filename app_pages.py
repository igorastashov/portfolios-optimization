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
import requests # Для API вызовов
import traceback # Для отладки
from decimal import Decimal, InvalidOperation # Added InvalidOperation
import random

# --- Helper function to fetch transactions via API ---
def fetch_transactions_api(backend_api_url, auth_headers):
    try:
        response = requests.get(f"{backend_api_url}/portfolios/me/transactions", headers=auth_headers)
        response.raise_for_status()
        transactions_list = response.json()
        for tx in transactions_list:
            if 'transaction_date' in tx and isinstance(tx['transaction_date'], str):
                tx['transaction_date'] = datetime.fromisoformat(tx['transaction_date'].replace('Z', '+00:00'))
            elif 'created_at' in tx and isinstance(tx['created_at'], str): # Fallback if transaction_date is not present
                 tx['transaction_date'] = datetime.fromisoformat(tx['created_at'].replace('Z', '+00:00'))
            tx['quantity'] = float(tx.get('quantity', 0))
            tx['price'] = float(tx.get('price', 0))
            tx['fee'] = float(tx.get('fee', 0))
            if 'asset_ticker' in tx and 'asset' not in tx:
                tx['asset'] = tx['asset_ticker']
            if 'transaction_type' in tx and 'type' not in tx:
                 tx['type'] = tx['transaction_type']
        return transactions_list
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 401:
            st.error("Сессия истекла или недействительна. Пожалуйста, войдите снова.")
        else:
            st.error(f"Не удалось загрузить транзакции (HTTP {http_err.response.status_code}): {http_err.response.text}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения при загрузке транзакций: {e}")
        return []
    except Exception as e:
        st.error(f"Непредвиденная ошибка при загрузке транзакций: {e}")
        traceback.print_exc()
        return []

# --- Helper function to fetch portfolio summary via API ---
def fetch_portfolio_summary_api(backend_api_url, auth_headers):
    try:
        response = requests.get(f"{backend_api_url}/portfolios/me/summary", headers=auth_headers)
        response.raise_for_status()
        summary_data = response.json()
        # Конвертируем числовые строки в Decimal где нужно
        summary_data['total_portfolio_value'] = Decimal(summary_data.get('total_portfolio_value', 0))
        if summary_data.get('total_invested_value') is not None:
            summary_data['total_invested_value'] = Decimal(summary_data.get('total_invested_value'))
        if summary_data.get('overall_pnl_absolute') is not None:
            summary_data['overall_pnl_absolute'] = Decimal(summary_data.get('overall_pnl_absolute'))
        if summary_data.get('total_value_24h_change_abs') is not None:
            summary_data['total_value_24h_change_abs'] = Decimal(summary_data.get('total_value_24h_change_abs'))
        
        for asset_summary in summary_data.get('assets', []):
            asset_summary['quantity'] = Decimal(asset_summary.get('quantity',0))
            if asset_summary.get('average_buy_price') is not None:
                asset_summary['average_buy_price'] = Decimal(asset_summary.get('average_buy_price'))
            if asset_summary.get('current_market_price') is not None:
                asset_summary['current_market_price'] = Decimal(asset_summary.get('current_market_price'))
            if asset_summary.get('current_value') is not None:
                asset_summary['current_value'] = Decimal(asset_summary.get('current_value'))
            if asset_summary.get('pnl_absolute') is not None:
                asset_summary['pnl_absolute'] = Decimal(asset_summary.get('pnl_absolute'))
            if asset_summary.get('value_24h_change_abs') is not None:
                asset_summary['value_24h_change_abs'] = Decimal(asset_summary.get('value_24h_change_abs'))
        return summary_data
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 401:
            st.error("Сессия истекла или недействительна. Пожалуйста, войдите снова.")
        elif http_err.response.status_code == 404: # Портфель может еще не существовать
            st.info("Портфель еще не создан или пуст. Данные для сводки отсутствуют.")
            return None # Возвращаем None, чтобы UI мог это обработать
        else:
            st.error(f"Не удалось загрузить сводку портфеля (HTTP {http_err.response.status_code}): {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения при загрузке сводки портфеля: {e}")
        return None
    except Exception as e:
        st.error(f"Непредвиденная ошибка при обработке сводки портфеля: {e}")
        traceback.print_exc()
        return None

# --- Helper function to fetch portfolio value history via API ---
def fetch_portfolio_value_history_api(backend_api_url, auth_headers, start_date_str=None, end_date_str=None, lookback_days=None):
    params = {}
    if start_date_str: params['startDate'] = start_date_str
    if end_date_str: params['endDate'] = end_date_str
    if lookback_days and not (start_date_str and end_date_str) : params['lookbackDays'] = lookback_days
    
    try:
        response = requests.get(f"{backend_api_url}/portfolios/me/value-history", headers=auth_headers, params=params)
        response.raise_for_status()
        history_response_data = response.json()
        history_points = history_response_data.get('history', [])
        
        df_history = pd.DataFrame(history_points)
        if not df_history.empty:
            df_history['date'] = pd.to_datetime(df_history['date'])
            df_history['value'] = df_history['value'].apply(lambda x: Decimal(str(x)))
            
        return df_history, history_response_data.get('start_date'), history_response_data.get('end_date'), history_response_data.get('currency_code')
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Не удалось загрузить историю стоимости (HTTP {http_err.response.status_code}): {http_err.response.text}")
        return pd.DataFrame(), None, None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения при загрузке истории стоимости: {e}")
        return pd.DataFrame(), None, None, None
    except Exception as e:
        st.error(f"Непредвиденная ошибка при обработке истории стоимости: {e}")
        traceback.print_exc()
        return pd.DataFrame(), None, None, None

def render_dashboard(username: str, backend_api_url: str, auth_headers: dict, available_assets: list): # Added backend_api_url, auth_headers, available_assets
    st.header(f"Аналитический дашборд, {username}")
    st.markdown("---")

    # Initialize session state for this page if not already done
    if 'db_status_message' not in st.session_state: st.session_state.db_status_message = ""
    if 'db_portfolio_summary' not in st.session_state: st.session_state.db_portfolio_summary = None
    if 'db_portfolio_history' not in st.session_state: st.session_state.db_portfolio_history = None
    if 'db_recent_transactions' not in st.session_state: st.session_state.db_recent_transactions = []
    if 'db_kpi_data' not in st.session_state: st.session_state.db_kpi_data = {} # Placeholder for KPIs
    if 'db_model_performance_data' not in st.session_state: st.session_state.db_model_performance_data = pd.DataFrame() # Placeholder

    # --- Data Loading ---
    # Load data on first run or if explicitly requested by a refresh button (not implemented yet)
    # We can use a simple flag to load once per session or manage it more granularly.
    if 'db_data_loaded' not in st.session_state:
        st.session_state.db_data_loaded = False

    if not st.session_state.db_data_loaded:
        with st.spinner("Загрузка данных для дашборда..."):
            # 1. Fetch Portfolio Summary (for allocation chart and some KPIs)
            summary_data, error = fetch_portfolio_summary_api(backend_api_url, auth_headers)
            if error:
                st.session_state.db_status_message += f"Ошибка загрузки сводки портфеля: {error}\n"
            else:
                st.session_state.db_portfolio_summary = summary_data

            # 2. Fetch Portfolio Value History (for dynamics chart)
            # Default to 90 days for the main dashboard chart
            history_data, error = fetch_portfolio_value_history_api(backend_api_url, auth_headers, lookback_days=90)
            if error:
                st.session_state.db_status_message += f"Ошибка загрузки истории стоимости: {error}\n"
            else:
                st.session_state.db_portfolio_history = history_data
            
            # 3. Fetch Recent Transactions
            transactions_data, error = fetch_transactions_api(backend_api_url, auth_headers, params={"limit": 5}) # Get latest 5
            if error:
                st.session_state.db_status_message += f"Ошибка загрузки транзакций: {error}\n"
            else:
                st.session_state.db_recent_transactions = transactions_data
            
            # 4. Placeholder for KPIs - In a real app, this would be an API call
            st.session_state.db_kpi_data = {
                "total_trades_value_month": Decimal(random.uniform(50000, 200000)).quantize(Decimal("0.01")),
                "drl_profitability_pct": Decimal(random.uniform(5, 25)).quantize(Decimal("0.01")),
                "avg_portfolio_return_pa": Decimal(random.uniform(8, 18)).quantize(Decimal("0.01")),
                "active_strategies": random.randint(1, 5)
            }

            # 5. Placeholder for Model Performance - In a real app, this would be an API call
            # For now, create a dummy DataFrame similar to what might come from an API
            model_names = [f"Стратегия DRL {i+1}" for i in range(5)]
            model_perf_values = [random.uniform(0.05, 0.35) for _ in range(5)] # e.g., Sharpe or % return
            st.session_state.db_model_performance_data = pd.DataFrame({
                "model_name": model_names,
                "performance_metric": model_perf_values
            }).sort_values(by="performance_metric", ascending=False)

            if not st.session_state.db_status_message:
                st.session_state.db_status_message = "Данные дашборда успешно загружены."
            st.session_state.db_data_loaded = True # Mark as loaded
        
        if st.session_state.db_status_message:
            if "Ошибка" in st.session_state.db_status_message:
                st.error(st.session_state.db_status_message)
            else:
                # st.success(st.session_state.db_status_message) # Can be too verbose
                pass


    # --- Display KPIs ---
    st.subheader("Ключевые показатели эффективности (KPI)")
    kpis = st.session_state.db_kpi_data
    if kpis:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Объем торгов (мес.)", f"${kpis.get('total_trades_value_month', 0):,.0f}", help="Симулированные данные")
        col2.metric("Прибыльность DRL (сред.)", f"{kpis.get('drl_profitability_pct', 0):.2f}%", help="Симулированные данные")
        col3.metric("Доходность портфеля (сред. годовая)", f"{kpis.get('avg_portfolio_return_pa', 0):.2f}%", help="Симулированные данные")
        col4.metric("Активных стратегий", str(kpis.get('active_strategies', 'N/A')), help="Симулированные данные")
    else:
        st.info("Данные KPI недоступны. Требуется API эндпоинт.")
    st.markdown("---")

    # --- Main Layout: Two Columns ---
    col_main_chart, col_side_info = st.columns([2, 1])

    with col_main_chart:
        st.subheader("Динамика стоимости портфеля (90 дней)")
        history = st.session_state.db_portfolio_history
        if history and history.get('history'):
            history_df = pd.DataFrame([p.model_dump() for p in history['history']])
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df = history_df.sort_values(by='date')
            
            fig_dynamics = px.line(history_df, x='date', y='value', title="Стоимость портфеля", labels={'value': 'Стоимость (USD)', 'date': 'Дата'})
            fig_dynamics.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_dynamics, use_container_width=True)
        elif history is None and not st.session_state.db_status_message: # Data not loaded yet
             st.info("Загрузка графика динамики...")
        else:
            st.warning("Данные для графика динамики стоимости портфеля недоступны.")

    with col_side_info:
        st.subheader("Распределение активов")
        summary = st.session_state.db_portfolio_summary
        if summary and summary.get('assets'):
            assets_data = []
            for asset_summary in summary['assets']:
                try:
                    # Ensure quantity and current_value are Decimal for calculations
                    quantity = Decimal(str(asset_summary.get('quantity', 0)))
                    current_value = Decimal(str(asset_summary.get('current_value', 0))) # This should be 'current_value' from summary
                    if quantity > Decimal('1e-9') and current_value > Decimal('0'): # Filter out zero/negligible value assets
                        assets_data.append({
                            "ticker": asset_summary.get('ticker', 'N/A'),
                            "current_value": current_value
                        })
                except (InvalidOperation, TypeError) as e:
                    st.error(f"Ошибка обработки данных актива {asset_summary.get('ticker','N/A')}: {e}")


            if assets_data:
                alloc_df = pd.DataFrame(assets_data)
                alloc_df = alloc_df[alloc_df['current_value'] > 0] # Ensure only positive values for pie chart

                if not alloc_df.empty:
                    fig_pie = px.pie(alloc_df, values="current_value", names="ticker", hole=0.3)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, height=280)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Нет активов с положительной стоимостью для отображения распределения.")
            else:
                st.info("Нет данных для графика распределения активов.")

        elif summary is None and not st.session_state.db_status_message : # Data not loaded yet
            st.info("Загрузка данных распределения...")
        else:
            st.info("Нет активов в портфеле или данные о них недоступны.")

        st.markdown("---")
        st.subheader("Последние операции")
        recent_txs = st.session_state.db_recent_transactions
        if recent_txs:
            tx_display_list = []
            for tx_raw in recent_txs:
                # tx_raw is already a dict from API. Ensure fields are present.
                tx_date_str = tx_raw.get('transaction_date') or tx_raw.get('created_at')
                tx_date = pd.to_datetime(tx_date_str).strftime('%Y-%m-%d %H:%M') if tx_date_str else "N/A"
                
                type_str = str(tx_raw.get('transaction_type', 'N/A')).upper()
                if type_str == "BUY": type_emoji = "➡️"
                elif type_str == "SELL": type_emoji = "⬅️"
                else: type_emoji = "⚙️"

                tx_display_list.append({
                    "Дата": tx_date,
                    "Тип": f"{type_emoji} {type_str}",
                    "Актив": tx_raw.get('asset_ticker', 'N/A'),
                    "Кол-во": f"{Decimal(str(tx_raw.get('quantity',0))):.4f}", # Ensure Decimal conversion
                    "Цена": f"${Decimal(str(tx_raw.get('price',0))):.2f}"      # Ensure Decimal conversion
                })
            tx_df = pd.DataFrame(tx_display_list)
            st.dataframe(tx_df, use_container_width=True, height=200) # Adjust height as needed
        elif not recent_txs and st.session_state.db_portfolio_summary is not None : # Data loaded, but no transactions
            st.info("Нет недавних операций.")
        else: # Data not loaded yet
            st.info("Загрузка последних операций...")


    st.markdown("---")
    
    # --- Performance Section ---
    st.subheader("Производительность моделей (Top 5)")
    model_perf_df = st.session_state.db_model_performance_data
    if not model_perf_df.empty:
        fig_model_perf = px.bar(
            model_perf_df.head(5),
            x="model_name",
            y="performance_metric",
            title="Топ-5 моделей по производительности (симуляция)",
            labels={"model_name": "Модель/Стратегия", "performance_metric": "Показатель эффективности (например, Sharpe)"},
            color="model_name"
        )
        fig_model_perf.update_layout(xaxis_title=None, yaxis_title="Эффективность", showlegend=False)
        st.plotly_chart(fig_model_perf, use_container_width=True)
    else:
        st.info("Данные о производительности моделей недоступны. Требуется API эндпоинт.")

    st.markdown("---")
    st.subheader("Рекомендации по ребалансировке")
    # Placeholder for recommendations - this could link to the recommendations page or show a brief summary from an API
    st.info("Детальные рекомендации доступны на странице 'Рекомендации'. (Эта секция будет обновлена для отображения краткой сводки).")
    if st.button("Перейти к Рекомендациям", key="db_goto_reco"):
        st.session_state.active_page = "Рекомендации"
        st.rerun()

    # Old code that used price_data, model_returns, model_actions, assets directly
    # This is now replaced by API calls and new logic above.
    # Commenting out or removing the old implementation details.
    """
    # ... (old code for KPIs, charts, transactions using local data) ...
    # Пример старого кода для KPI:
    # total_volume = price_data.apply(lambda x: x * np.random.randint(1, 100)).sum().sum()
    # col1.metric("Общий объем торгов", f"${total_volume:,.0f}")

    # Пример старого кода для графика:
    # portfolio_value_over_time = price_data.mean(axis=1) * 1000 # Примерный расчет
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=portfolio_value_over_time.index, y=portfolio_value_over_time.values, mode='lines', name='Стоимость портфеля'))
    # st.plotly_chart(fig, use_container_width=True)
    
    # Пример старого кода для распределения:
    # current_portfolio = assets[:5] # Пример
    # current_allocations = np.random.rand(len(current_portfolio))
    # current_allocations /= current_allocations.sum()
    # fig_pie = px.pie(values=current_allocations, names=current_portfolio, title="Распределение активов")
    # st.plotly_chart(fig_pie, use_container_width=True)
    """
    return # Explicit return, though not strictly necessary for Streamlit rendering functions

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

def render_account_dashboard(username: str, available_assets: list, backend_api_url: str, auth_headers: dict):
    st.header("Единый торговый аккаунт")

    # --- Инициализация состояния сессии для этой страницы ---
    # Префикс 'ad_' для Account Dashboard
    if 'ad_portfolio_summary_data' not in st.session_state: st.session_state.ad_portfolio_summary_data = None
    if 'ad_force_reload_summary' not in st.session_state: st.session_state.ad_force_reload_summary = True
    if 'ad_value_history_df' not in st.session_state: st.session_state.ad_value_history_df = pd.DataFrame()
    if 'ad_history_start_date' not in st.session_state: st.session_state.ad_history_start_date = None
    if 'ad_history_end_date' not in st.session_state: st.session_state.ad_history_end_date = None
    if 'ad_history_currency' not in st.session_state: st.session_state.ad_history_currency = "USD"
    if 'ad_last_interval_key' not in st.session_state: st.session_state.ad_last_interval_key = None 

    if st.button("🔄 Обновить данные аккаунта"):
        st.session_state.ad_force_reload_summary = True
        st.session_state.ad_value_history_df = pd.DataFrame() 
        st.rerun() 

    if st.session_state.ad_force_reload_summary:
        with st.spinner("Загрузка сводки портфеля..."):
            st.session_state.ad_portfolio_summary_data = fetch_portfolio_summary_api(backend_api_url, auth_headers)
        st.session_state.ad_force_reload_summary = False

    portfolio_summary = st.session_state.ad_portfolio_summary_data

    if not portfolio_summary:
        st.info("Не удалось загрузить данные портфеля или портфель пуст. "
                "Если вы только что зарегистрировались или еще не добавляли транзакции, "
                "перейдите в раздел 'Управление транзакциями'.")
        if st.button("Перейти к управлению транзакциями", key="goto_tm_from_ad_empty"):
            st.warning("Пожалуйста, перейдите на страницу 'Управление активами' через боковое меню.")
        return

    st.caption(f"Данные портфеля '{portfolio_summary.get('name', 'N/A')}' ({portfolio_summary.get('portfolio_id', 'N/A')}). Валюта: {portfolio_summary.get('currency_code', 'N/A')}")
    st.caption(f"Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (локальное время UI)")
    
    col1, col2, col3, col4 = st.columns(4)

    total_balance = portfolio_summary.get('total_portfolio_value', Decimal('0'))
    today_pnl_abs = portfolio_summary.get('total_value_24h_change_abs', Decimal('0'))
    today_pnl_pct = portfolio_summary.get('total_value_24h_change_pct', 0.0)
    overall_pnl_abs = portfolio_summary.get('overall_pnl_absolute')
    overall_pnl_pct = portfolio_summary.get('overall_pnl_percentage')
    currency_symbol = portfolio_summary.get('currency_code', '$') 

    with col1:
        st.metric("Общий баланс", f"{currency_symbol}{total_balance:,.2f}")
    with col2:
        arrow = "▲" if today_pnl_abs >= 0 else "▼"
        st.metric("P&L за 24ч", 
                  f"{arrow} {currency_symbol}{abs(today_pnl_abs):,.2f}", 
                  f"{arrow} {abs(today_pnl_pct):.2f}%" if today_pnl_pct is not None else "N/A",
                  delta_color="normal" if today_pnl_abs >= 0 else "inverse")
    with col3:
        if overall_pnl_abs is not None and overall_pnl_pct is not None:
            arrow_total = "▲" if overall_pnl_abs >= 0 else "▼"
            st.metric("Общий P&L", 
                      f"{arrow_total} {currency_symbol}{abs(overall_pnl_abs):,.2f}", 
                      f"{arrow_total} {abs(overall_pnl_pct):.2f}%",
                      delta_color="normal" if overall_pnl_abs >= 0 else "inverse")
        else:
            st.metric("Общий P&L", "N/A", "Нет данных или покупок")
    with col4:
        st.metric("Доступная маржа", f"{currency_symbol}{(total_balance * Decimal('0.7')):,.2f}", "(Пример)") 

    st.subheader("Ваши активы")
    assets_in_portfolio = portfolio_summary.get('assets', [])
    if assets_in_portfolio:
        assets_df_data = []
        for item in assets_in_portfolio:
            assets_df_data.append({
                "Актив": item.get('asset_ticker'),
                "Кол-во": item.get('quantity', Decimal('0')),
                "Цена (тек.)": item.get('current_market_price'),
                "Стоимость (тек.)": item.get('current_value'),
                "Цена (ср.пок.)": item.get('average_buy_price'),
                "P&L (абс.)": item.get('pnl_absolute'),
                "P&L (%)": item.get('pnl_percentage'),
                "Изм. 24ч (абс.)": item.get('value_24h_change_abs'),
                "Изм. 24ч (%)": item.get('value_24h_change_pct')
            })
        assets_df = pd.DataFrame(assets_df_data)
        
        formatted_df = assets_df.copy()
        formatted_df["Кол-во"] = formatted_df["Кол-во"].apply(lambda x: f"{x:,.8f}" if x is not None else "-")
        for col in ["Цена (тек.)", "Стоимость (тек.)", "Цена (ср.пок.)", "P&L (абс.)", "Изм. 24ч (абс.)"]:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{currency_symbol}{x:,.2f}" if x is not None else "-")
        for col in ["P&L (%)", "Изм. 24ч (%)"]:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%" if x is not None else "-")
        
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    else:
        st.info("В вашем портфеле пока нет активов.")

    st.subheader("Распределение капитала")
    if assets_in_portfolio:
        pie_data = [{'Актив': item.get('asset_ticker'), 'Стоимость': item.get('current_value', Decimal('0'))} 
                    for item in assets_in_portfolio if item.get('current_value', Decimal('0')) > 0]
        if pie_data:
            pie_df = pd.DataFrame(pie_data)
            fig_pie = px.pie(pie_df, values="Стоимость", names="Актив", title="Распределение капитала по текущей стоимости")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Нет данных для графика распределения (все активы с нулевой или отсутствующей стоимостью).")
    else:
        st.info("Нет активов для отображения распределения.")

    st.header("Динамика стоимости портфеля")
    interval_options = {"7д": 7, "30д": 30, "90д": 90, "180д": 180, "1г": 365, "Все": 3650}
    selected_interval_key = st.radio("Интервал:", list(interval_options.keys()), index=2, horizontal=True, key="ad_interval_radio")
    days_to_look_back = interval_options[selected_interval_key]

    if st.session_state.ad_last_interval_key != selected_interval_key or st.session_state.ad_value_history_df.empty:
        with st.spinner("Загрузка истории стоимости портфеля..."):
            df_hist, hist_start, hist_end, hist_curr = fetch_portfolio_value_history_api(
                backend_api_url, auth_headers, lookback_days=days_to_look_back
            )
            st.session_state.ad_value_history_df = df_hist
            st.session_state.ad_history_start_date = hist_start
            st.session_state.ad_history_end_date = hist_end
            st.session_state.ad_history_currency = hist_curr or currency_symbol 
            st.session_state.ad_last_interval_key = selected_interval_key

    portfolio_history_df = st.session_state.ad_value_history_df
    hist_start_str = st.session_state.ad_history_start_date
    hist_end_str = st.session_state.ad_history_end_date
    hist_currency = st.session_state.ad_history_currency

    if not portfolio_history_df.empty:
        display_start = pd.to_datetime(hist_start_str).strftime('%Y-%m-%d') if hist_start_str else "N/A"
        display_end = pd.to_datetime(hist_end_str).strftime('%Y-%m-%d') if hist_end_str else "N/A"
        st.write(f"Отображается динамика стоимости портфеля ({hist_currency}) с {display_start} по {display_end}.")

        if len(portfolio_history_df) > 1:
            start_val = portfolio_history_df['value'].iloc[0]
            end_val = portfolio_history_df['value'].iloc[-1]
            change_val = end_val - start_val
            change_pct = (change_val / start_val * 100) if start_val != 0 else (Decimal('inf') if end_val > 0 else Decimal('0'))
            
            days_num = (pd.to_datetime(hist_end_str) - pd.to_datetime(hist_start_str)).days if hist_start_str and hist_end_str else 0
            apy_pct = Decimal('0')
            if days_num > 0 and start_val != 0: # Проверяем start_val != 0
                try: 
                    apy_pct = ((end_val / start_val) ** (Decimal('365.0') / Decimal(days_num)) - 1) * 100
                except Exception: # Ловим более общие исключения для Decimal
                    pass

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Изм. за период", f"{'▲' if change_val >=0 else '▼'} {hist_currency}{abs(change_val):.2f}")
            m_col2.metric("Изм. (%) за период", f"{'▲' if change_pct >=0 else '▼'} {abs(change_pct):.2f}%" if start_val != 0 else "N/A")
            m_col3.metric("APY", f"{'▲' if apy_pct >=0 else '▼'} {abs(apy_pct):.2f}%" if days_num > 0 and start_val != 0 else "N/A")

        fig_hist = px.line(portfolio_history_df, x='date', y='value', title=f"Динамика стоимости ({hist_currency})", markers=True)
        fig_hist.update_layout(xaxis_title="Дата", yaxis_title=f"Стоимость ({hist_currency})")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Нет данных для отображения истории стоимости портфеля.")

    st.info("Для изменения состава портфеля, перейдите в раздел 'Управление транзакциями'.")
    if st.button("Перейти к управлению транзакциями", key="goto_tm_from_ad_bottom"):
        st.warning("Пожалуйста, перейдите на страницу 'Управление активами' через боковое меню.") 

def render_transactions_manager(username: str, available_assets: list, backend_api_url: str, auth_headers: dict):
    st.header("Управление транзакциями")
    st.info("Здесь вы можете добавлять, просматривать и удалять транзакции.")

    if 'tm_transactions_list' not in st.session_state: st.session_state.tm_transactions_list = []
    if 'tm_force_reload_transactions' not in st.session_state: st.session_state.tm_force_reload_transactions = True
    if 'tm_error_message' not in st.session_state: st.session_state.tm_error_message = None
    if 'tm_success_message' not in st.session_state: st.session_state.tm_success_message = None

    if st.session_state.tm_force_reload_transactions:
        with st.spinner("Загрузка списка транзакций..."):
            st.session_state.tm_transactions_list = fetch_transactions_api(backend_api_url, auth_headers)
            st.session_state.tm_force_reload_transactions = False
            # Сбрасываем сообщения после перезагрузки
            st.session_state.tm_error_message = None
            st.session_state.tm_success_message = None
    
    transactions = st.session_state.tm_transactions_list

    # Отображение сообщений об успехе/ошибке
    if st.session_state.tm_success_message:
        st.success(st.session_state.tm_success_message)
        st.session_state.tm_success_message = None # Очищаем после показа
    if st.session_state.tm_error_message:
        st.error(st.session_state.tm_error_message)
        st.session_state.tm_error_message = None # Очищаем после показа

    tab1, tab2 = st.tabs(["Добавить транзакцию", "Список транзакций и удаление"])

    with tab1:
        st.subheader("Добавить новую транзакцию")
        
        with st.form("add_transaction_form_api", clear_on_submit=True):
            asset_ticker = st.selectbox(
                "Актив", 
                options=available_assets if available_assets else ["Загрузка списка активов..."], # Используем available_assets из auth_app
                help="Выберите актив из списка доступных."
            )
            transaction_type = st.selectbox(
                "Тип транзакции", 
                options=["buy", "sell"],
                format_func=lambda x: "Покупка" if x == "buy" else "Продажа",
                help="Выберите тип операции: покупка или продажа."
            )
            quantity_val = st.number_input( # Изменено имя переменной
                "Количество", 
                min_value=0.00000001, 
                value=1.0, 
                step=0.00000001, 
                format="%.8f",
                help="Количество актива в транзакции."
            )
            price_val = st.number_input( # Изменено имя переменной
                "Цена за единицу", 
                min_value=0.00000001, 
                value=100.0, 
                step=0.00000001, 
                format="%.8f",
                help="Цена актива за одну единицу в валюте портфеля."
            )
            transaction_date_val = st.date_input( # Изменено имя переменной
                "Дата транзакции", 
                value=datetime.now().date(),
                help="Дата совершения транзакции."
            )
            # transaction_time = st.time_input("Время транзакции", value=datetime.now().time())
            fee_val = st.number_input("Комиссия", min_value=0.0, value=0.0, step=0.01, format="%.2f", help="Сумма комиссии за транзакцию.")
            notes = st.text_area("Примечания", help="Дополнительные заметки по транзакции.")

            submitted = st.form_submit_button("Добавить транзакцию")

            if submitted:
                if not asset_ticker or asset_ticker == "Загрузка списка активов...":
                    st.session_state.tm_error_message = "Пожалуйста, выберите актив."
                elif quantity_val <= 0 or price_val <= 0:
                    st.session_state.tm_error_message = "Количество и цена должны быть положительными."
                else:
                    payload = {
                        "asset_ticker": asset_ticker,
                        "transaction_type": transaction_type,
                        "quantity": str(quantity_val), # API ожидает Decimal, передаем как строку
                        "price": str(price_val),       # API ожидает Decimal, передаем как строку
                        "transaction_date": transaction_date_val.isoformat(), # Только дата
                        "fee": str(fee_val),           # API ожидает Decimal, передаем как строку
                        "notes": notes
                    }
                    try:
                        with st.spinner("Отправка транзакции..."):
                            response = requests.post(
                                f"{backend_api_url}/portfolios/me/transactions", 
                                headers=auth_headers, 
                                json=payload
                            )
                            response.raise_for_status()
                            st.session_state.tm_success_message = "Транзакция успешно добавлена!"
                            st.session_state.tm_force_reload_transactions = True # Флаг для перезагрузки списка
                            # Очистка полей формы происходит автоматически из-за clear_on_submit=True
                    except requests.exceptions.HTTPError as http_err:
                        try:
                            error_detail = http_err.response.json().get("detail", http_err.response.text)
                        except: # requests.exceptions.JSONDecodeError or other
                            error_detail = http_err.response.text
                        st.session_state.tm_error_message = f"Ошибка добавления транзакции (HTTP {http_err.response.status_code}): {error_detail}"
                    except requests.exceptions.RequestException as req_err:
                        st.session_state.tm_error_message = f"Ошибка соединения: {req_err}"
                    except Exception as e:
                        st.session_state.tm_error_message = f"Непредвиденная ошибка: {e}"
                        traceback.print_exc()
                st.rerun() # Перезапускаем, чтобы обновить сообщения и список, если нужно

    with tab2:
        st.subheader("Список транзакций")
        if st.button("🔄 Обновить список транзакций", key="reload_transactions_tab2"):
            st.session_state.tm_force_reload_transactions = True
            st.rerun()

        if transactions:
            # Конвертация для отображения, включая корректную обработку Decimal из API
            df_transactions = pd.DataFrame(transactions)
            
            # Преобразование полей для отображения
            df_display = df_transactions.copy()
            if 'transaction_date' in df_display.columns:
                 df_display['Дата'] = pd.to_datetime(df_display['transaction_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                 df_display['Дата'] = "N/A"

            df_display['Актив'] = df_display['asset_ticker']
            df_display['Тип'] = df_display['transaction_type'].apply(lambda x: 'Покупка' if x == 'buy' else ('Продажа' if x == 'sell' else x))
            
            # Используем .get() с преобразованием в Decimal, затем в float для форматирования
            df_display['Количество'] = df_display['quantity'].apply(lambda x: f"{float(x):.8f}" if x is not None else "N/A")
            df_display['Цена'] = df_display['price'].apply(lambda x: f"${float(x):.2f}" if x is not None else "N/A")
            df_display['Комиссия'] = df_display['fee'].apply(lambda x: f"${float(x):.2f}" if x is not None else "N/A")
            df_display['Примечания'] = df_display['notes']
            df_display['ID'] = df_display['id']

            st.dataframe(
                df_display[['ID', 'Дата', 'Актив', 'Тип', 'Количество', 'Цена', 'Комиссия', 'Примечания']],
                use_container_width=True,
                hide_index=True
            )

            # --- Функционал удаления транзакций ---
            st.subheader("Удалить транзакцию")
            if not df_transactions.empty: # Убедимся, что есть транзакции для удаления
                transaction_ids = df_transactions["id"].tolist()
                
                # Формируем читаемые опции для selectbox
                # transaction_options = {
                #     f"ID: {tx['id']} - {tx.get('transaction_date','N/A')} - {tx['asset_ticker']} - {tx['transaction_type']} - Q:{tx['quantity']} P:{tx['price']}": tx['id'] 
                #     for tx in transactions # Используем оригинальный список transactions
                # }

                # Более простой вариант для отображения ID
                transaction_options_display = [f"ID: {tid}" for tid in transaction_ids]


                selected_display_option = st.selectbox(
                    "Выберите транзакцию для удаления:",
                    options=[""] + transaction_options_display, # Добавляем пустой вариант для "не выбрано"
                    index=0
                )

                if selected_display_option:
                    selected_tx_id = int(selected_display_option.split(": ")[1]) # Извлекаем ID
                    
                    st.warning(f"Вы уверены, что хотите удалить транзакцию с ID: {selected_tx_id}?")
                    if st.button("Удалить выбранную транзакцию", key=f"delete_tx_{selected_tx_id}"):
                        try:
                            with st.spinner(f"Удаление транзакции {selected_tx_id}..."):
                                delete_response = requests.delete(
                                    f"{backend_api_url}/portfolios/me/transactions/{selected_tx_id}",
                                    headers=auth_headers
                                )
                                delete_response.raise_for_status()
                                st.session_state.tm_success_message = f"Транзакция ID {selected_tx_id} успешно удалена."
                                st.session_state.tm_force_reload_transactions = True
                        except requests.exceptions.HTTPError as http_err:
                            try:
                                error_detail = http_err.response.json().get("detail", http_err.response.text)
                            except:
                                error_detail = http_err.response.text
                            st.session_state.tm_error_message = f"Ошибка удаления (HTTP {http_err.response.status_code}): {error_detail}"
                        except requests.exceptions.RequestException as req_err:
                            st.session_state.tm_error_message = f"Ошибка соединения: {req_err}"
                        except Exception as e:
                            st.session_state.tm_error_message = f"Непредвиденная ошибка: {e}"
                            traceback.print_exc()
                        st.rerun() # Перезапуск для обновления списка и сообщений
            else:
                st.info("Нет транзакций для удаления.")
        else:
            st.info("Список транзакций пуст.")

def render_about():
    st.title("О проекте")
    st.markdown("""
    ### Система Оптимизации и Мониторинга Финансовых Портфелей "ProInvest"
    
    **Версия:** 0.7.0 (Интеграция API, Docker, RabbitMQ)

    **Назначение:**
    Данная система разработана в рамках выпускной квалификационной работы (ВКР) и предназначена для помощи частным инвесторам и финансовым аналитикам в:
    - Управлении и отслеживании своих инвестиционных портфелей.
    - Проведении анализа эффективности портфелей на исторических данных (бэктестинг).
    - Получении рекомендаций по ребалансировке портфелей на основе современных моделей (Markowitz, DRL).
    - Анализе новостного фона по отдельным активам с использованием NLP-моделей.
    - Моделировании гипотетических портфельных стратегий.

    **Ключевые технологии:**
    - **Frontend:** Streamlit
    - **Backend:** FastAPI (Python)
    - **База данных:** PostgreSQL
    - **Асинхронные задачи:** Celery
    - **Брокер сообщений:** RabbitMQ
    - **Кэш/Результаты задач:** Redis
    - **Управление зависимостями:** Poetry
    - **Контейнеризация:** Docker, Docker Compose
    - **Миграции БД:** Alembic
    - **Аутентификация:** JWT (Access & Refresh Tokens - *refresh token пока не реализован полностью*)
    - **ML/DRL Модели (планируется):**
        - Классическая оптимизация Марковица.
        - Агенты глубокого обучения с подкреплением (DRL) для торговли.
        - NLP-модели (например, FinBERT) для анализа тональности новостей.

    **Текущий статус и известные проблемы:**
    - Произведен рефакторинг большинства страниц Streamlit для работы через FastAPI бэкенд.
    - Реализованы базовые CRUD операции для портфелей, активов (неявное создание через транзакции), транзакций.
    - Интегрирован Celery с RabbitMQ (брокер) и Redis (результаты) для выполнения фоновых задач (анализ портфеля, новости, рекомендации, гипотетическое моделирование).
    - Логика в Celery задачах пока что симулированная, требует наполнения реальными алгоритмами.
    - **Проблема с Alembic миграциями:** Остается нерешенной проблема с `BACKEND_CORS_ORIGINS` и `sqlalchemy.exc.OperationalError` при `alembic upgrade head` в Docker. Требует дополнительной отладки конфигурации `.env` и/или `env.py` Alembic. *Пользователь временно пропустил этот шаг.*
    - Streamlit приложение контейнеризировано.
    - Некоторые страницы Streamlit (`render_dashboard`, `render_portfolio_optimization`, `render_model_training`, `render_model_comparison`, `render_backtest`) еще требуют полного рефакторинга для работы с API.
    - Необходимо реализовать полноценную бизнес-логику в Celery-задачах.
    - Уточнить требования к списку тикеров для `available_assets`.

    **Дальнейшие шаги:**
    1.  Реализация настоящей бизнес-логики в Celery задачах.
    2.  Завершение рефакторинга оставшихся Streamlit страниц.
    3.  Устранение проблем с Alembic миграциями.
    4.  Развитие функционала MLOps с использованием ClearML.
    5.  Улучшение UI/UX.

    ---
    *Разработчик: [Ваше Имя/Псевдоним]*
    *Научный руководитель: [Имя Научного Руководителя]*
    *Университет/Организация, Год*
    """)


# ... (остальной код файла app_pages.py, если есть) ...