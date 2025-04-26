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
import traceback # Ensure traceback is imported

# Импорт модулей приложения
from portfolios_optimization.data_loader import load_price_data, load_return_data, load_model_actions
from portfolios_optimization.portfolio_optimizer import optimize_markowitz_portfolio
from portfolios_optimization.portfolio_analysis import calculate_metrics, plot_efficient_frontier
from portfolios_optimization.visualization import plot_portfolio_performance, plot_asset_allocation
from portfolios_optimization.model_trainer import train_model, load_trained_model
from portfolios_optimization.authentication import (
    initialize_users_file, register_user, authenticate_user, get_user_info,
    update_user_portfolio, get_user_portfolios, get_user_portfolio, get_portfolio_with_quantities,
    get_user_transactions
)

# <<< NEW: Import the analysis function >>>
from portfolio_analyzer import run_portfolio_analysis

# Импорт страниц приложения
from app_pages import render_account_dashboard, render_transactions_manager

# Конфигурация страницы
st.set_page_config(
    page_title="Portfolio Optimization System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация файла пользователей
initialize_users_file()

# Инициализация состояния сессии
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'login_message' not in st.session_state:
    st.session_state.login_message = None
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Login"

# Функция для выхода из аккаунта
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.active_page = "Login"
    st.session_state.login_message = "Вы вышли из системы"
    # Clear analysis results on logout
    if 'analysis_results' in st.session_state: del st.session_state['analysis_results']
    if 'analysis_figure' in st.session_state: del st.session_state['analysis_figure']

# Загрузка данных для всего приложения
@st.cache_data(ttl=3600)
def load_data():
    # Данные цен активов
    price_data = load_price_data()
    
    # Данные доходности моделей
    model_returns = load_return_data()
    
    # Данные распределения портфелей моделей
    model_actions = load_model_actions()
    
    return price_data, model_returns, model_actions

# Загрузка данных
price_data, model_returns, model_actions = load_data()

# Получение списка доступных активов
assets = price_data.columns.tolist() if not price_data.empty else []

# Основной заголовок приложения
st.title("Investment Portfolio Monitoring & Optimization System")

# Страница аутентификации
if not st.session_state.logged_in:
    # Вкладки для входа и регистрации
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])
    
    # Вкладка входа
    with tab1:
        st.header("Вход в систему")
        
        # Форма входа
        with st.form("login_form"):
            username = st.text_input("Имя пользователя")
            password = st.text_input("Пароль", type="password")
            submit_button = st.form_submit_button("Войти")
            
            if submit_button:
                success, message = authenticate_user(username, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.active_page = "Мой кабинет"
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # Сообщение о выходе или ошибке
        if st.session_state.login_message:
            st.info(st.session_state.login_message)
            st.session_state.login_message = None
    
    # Вкладка регистрации
    with tab2:
        st.header("Регистрация нового пользователя")
        
        # Форма регистрации
        with st.form("register_form"):
            new_username = st.text_input("Имя пользователя")
            new_password = st.text_input("Пароль", type="password")
            confirm_password = st.text_input("Подтверждение пароля", type="password")
            email = st.text_input("Email")
            register_button = st.form_submit_button("Зарегистрироваться")
            
            if register_button:
                if not new_username or not new_password:
                    st.error("Имя пользователя и пароль обязательны")
                elif new_password != confirm_password:
                    st.error("Пароли не совпадают")
                else:
                    success, message = register_user(new_username, new_password, email)
                    
                    if success:
                        st.success(message)
                        st.info("Теперь вы можете войти в систему")
                    else:
                        st.error(message)

# Интерфейс для авторизованных пользователей
else:
    # Боковая панель для навигации
    st.sidebar.header(f"Привет, {st.session_state.username}!")
    
    # Кнопка выхода в боковой панели
    if st.sidebar.button("Выйти"):
        logout()
        st.rerun()
    
    # Меню навигации
    st.sidebar.header("Навигация")
    page_options = ["Мой кабинет", "Управление активами", "Единый торговый аккаунт", "Анализ портфеля"]
    
    # Устанавливаем индекс для radio на основе текущей активной страницы в состоянии сессии
    try:
        # Handle case where previously active page might be removed
        if st.session_state.active_page not in page_options:
            st.session_state.active_page = page_options[0] # Default to first available
        current_page_index = page_options.index(st.session_state.active_page)
    except ValueError:
        current_page_index = 0 # По умолчанию первая страница
        st.session_state.active_page = page_options[0]

    selected_page = st.sidebar.radio(
        "Выберите раздел",
        page_options,
        index=current_page_index,
        key="main_nav_radio"
    )
    
    # Обновляем состояние сессии, ТОЛЬКО если пользователь выбрал ДРУГУЮ страницу в radio
    if selected_page != st.session_state.active_page:
        st.session_state.active_page = selected_page
        st.rerun()

    # Страница личного кабинета пользователя
    if st.session_state.active_page == "Мой кабинет":
        st.header("Личный кабинет")
        
        # Получение информации о пользователе
        user_info = get_user_info(st.session_state.username)
        
        if user_info:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Ваш профиль")
                st.write(f"**Email:** {user_info.get('email', 'Не указан')}")
                st.write(f"**Дата регистрации:** {user_info.get('created_at', 'Неизвестно')}")
                st.write(f"**Последний вход:** {user_info.get('last_login', 'Неизвестно')}")
            
            st.subheader("Ваши портфели")
            
            # Получение данных о транзакционном портфеле пользователя
            portfolio_data = get_portfolio_with_quantities(st.session_state.username)
            
            # Проверка наличия активов в портфеле
            has_assets = portfolio_data and any(portfolio_data["quantities"].values())
            
            if has_assets:
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
                        
                        portfolio_items.append({
                            "Актив": asset,
                            "Количество": quantity,
                            "Средняя цена покупки": avg_buy_price,
                            "Текущая цена": current_price,
                            "Текущая стоимость": current_value,
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
                    formatted_df["Количество"] = formatted_df["Количество"].apply(lambda x: f"{x:,.8f}")
                    formatted_df["Средняя цена покупки"] = formatted_df["Средняя цена покупки"].apply(lambda x: f"${x:,.2f}")
                    formatted_df["Текущая цена"] = formatted_df["Текущая цена"].apply(lambda x: f"${x:,.2f}")
                    formatted_df["Текущая стоимость"] = formatted_df["Текущая стоимость"].apply(lambda x: f"${x:,.2f}")
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
                    
                    st.info("Чтобы изменить состав портфеля, добавьте или удалите активы через раздел 'Управление активами'.")
            else:
                st.info("""
                У вас пока нет активов в портфеле.
                
                Чтобы сформировать портфель:
                1. Перейдите в раздел 'Управление активами'
                2. На вкладке 'Добавить транзакцию' добавьте свои первые активы
                3. После добавления транзакций, портфель сформируется автоматически
                
                Ваш портфель будет отображаться здесь и в разделе 'Единый торговый аккаунт'.
                """)
                
                # Кнопка перехода к разделу "Управление активами"
                if st.button("Перейти к управлению активами", key="goto_manage_assets_from_cabinet"):
                    st.session_state.active_page = "Управление активами"
                    st.rerun()
        else:
            st.error("Не удалось загрузить информацию о пользователе")
    
    # Страница единого торгового аккаунта в стиле Bybit
    elif st.session_state.active_page == "Единый торговый аккаунт":
        st.header("Единый торговый аккаунт")
        st.markdown("--- ")

        # --- Imports for this page ---
        from collections import OrderedDict
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
        import plotly.express as px

        # --- Load User Transactions --- 
        username = st.session_state.username
        transactions_list = get_user_transactions(username)

        # Convert list of transactions to DataFrame
        if transactions_list:
             transactions_df_raw = pd.DataFrame(transactions_list)
             # Ensure correct dtypes after loading from JSON
             transactions_df_raw['date'] = pd.to_datetime(transactions_df_raw['date'])
             transactions_df_raw['quantity'] = pd.to_numeric(transactions_df_raw['quantity'])
             transactions_df_raw['price'] = pd.to_numeric(transactions_df_raw['price'])
             transactions_df_raw['fee'] = pd.to_numeric(transactions_df_raw.get('fee', 0))
             if 'total_cost' not in transactions_df_raw.columns:
                  transactions_df_raw['total_cost'] = transactions_df_raw['quantity'] * transactions_df_raw['price'] + transactions_df_raw['fee']
             else:
                   transactions_df_raw['total_cost'] = pd.to_numeric(transactions_df_raw['total_cost'])
             # Sort transactions chronologically - IMPORTANT
             transactions_df_raw = transactions_df_raw.sort_values(by='date').reset_index(drop=True)
        else:
            transactions_df_raw = pd.DataFrame()

        if transactions_df_raw.empty:
            st.info("У вас пока нет транзакций. Добавьте их на странице 'Управление активами'.")
            if st.button("Перейти к Управлению активами", key="uta_goto_manage"):
                 st.session_state.active_page = "Управление активами"
                 st.rerun()
            st.stop()

        # Filter only buy transactions for the core logic of the notebook
        # Note: Sell transactions are ignored in the notebook's P&L logic, only used for markers
        buy_transactions_df = transactions_df_raw[transactions_df_raw['type'] == 'buy'].copy()
        # We need an ID for each buy transaction for the logic
        buy_transactions_df['Purchase_ID'] = buy_transactions_df.index # Simple ID based on order

        if buy_transactions_df.empty:
            st.info("В истории нет транзакций покупки. График динамики P&L не может быть построен.")
            # We can still show current holdings based on all transactions
            # (Add logic here later if needed to show holdings even without buys)
            st.stop()

        # --- Calculate Current Holdings (based on ALL transactions) --- 
        holdings = {}
        for _, row in transactions_df_raw.iterrows():
            asset = row['asset']
            quantity = row['quantity']
            type = row['type']
            if asset not in holdings: holdings[asset] = 0
            if type == 'buy': holdings[asset] += quantity
            elif type == 'sell': holdings[asset] -= quantity
        current_holdings = {asset: q for asset, q in holdings.items() if q > 1e-9} # Tolerance
        required_assets = list(current_holdings.keys())

        if not required_assets:
            st.info("После обработки транзакций у вас нет активов в портфеле.")
            st.stop()

        # --- Load Historical Data (Function definition kept separate for clarity) --- 
        @st.cache_data(ttl=1800)
        def load_and_preprocess_historical_data_uta(assets_list):
            csv_base_path = 'D:\\__projects__\\diploma\\portfolios-optimization\\data'
            all_prices = {}
            data_found = False
            min_start_date = pd.Timestamp.max.tz_localize(None)
            for asset in assets_list:
                file_path = os.path.join(csv_base_path, f'{asset}_hourly_data.csv')
                try:
                    df = pd.read_csv(file_path)
                    if 'Open time' not in df.columns: time_col = next((c for c in df.columns if 'time' in c.lower()), None); df.rename(columns={time_col:'Open time'}, inplace=True)
                    if 'Close' not in df.columns: price_col = next((c for c in df.columns if 'close' in c.lower()), None); df.rename(columns={price_col:'Close'}, inplace=True)
                    df['Open time'] = pd.to_datetime(df['Open time'])
                    df = df.set_index('Open time')
                    df = df[['Close']].rename(columns={'Close': f'{asset}_Price'})
                    df[f'{asset}_Price'] = df[f'{asset}_Price'].astype(float)
                    all_prices[asset] = df
                    data_found = True
                    if not df.empty: min_start_date = min(min_start_date, df.index.min())
                except Exception as e:
                    st.warning(f"Ошибка загрузки/обработки {asset}: {e}")
            if not data_found: return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)
            if min_start_date == pd.Timestamp.max.tz_localize(None): min_start_date = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=1)
            return pd.concat(all_prices.values(), axis=1), min_start_date

        historical_prices, earliest_data_date = load_and_preprocess_historical_data_uta(required_assets)

        if historical_prices.empty:
            st.error("Не удалось загрузить исторические данные для активов в портфеле.")
            st.stop()

        # --- Calculate Metrics (Current State) --- 
        latest_prices = historical_prices.ffill().iloc[-1]
        prices_24h_ago = historical_prices.ffill().iloc[-25] if len(historical_prices) >= 25 else latest_prices
        
        total_balance = 0
        total_balance_24h_ago = 0
        current_holdings_list = []
        total_cost_basis_from_all_tx = 0 # Recalculate cost basis from raw transactions for accuracy
        temp_holdings_for_cost = {}
        for _, row in transactions_df_raw.iterrows():
            asset = row['asset']
            q = row['quantity']
            cost = row['total_cost']
            type = row['type']
            if asset not in temp_holdings_for_cost: temp_holdings_for_cost[asset] = {'q':0, 'cost':0}
            if type == 'buy': 
                temp_holdings_for_cost[asset]['q'] += q
                temp_holdings_for_cost[asset]['cost'] += cost
            elif type == 'sell':
                if temp_holdings_for_cost[asset]['q'] > 1e-9: # Check if holding exists
                    ratio = min(q / temp_holdings_for_cost[asset]['q'], 1.0)
                    temp_holdings_for_cost[asset]['cost'] *= (1 - ratio)
                    temp_holdings_for_cost[asset]['q'] -= q
                temp_holdings_for_cost[asset]['q'] = max(0, temp_holdings_for_cost[asset]['q'])
        total_cost_basis_from_all_tx = sum(d['cost'] for d in temp_holdings_for_cost.values() if d['q'] > 1e-9)

        for asset, quantity in current_holdings.items():
            current_price = latest_prices.get(f'{asset}_Price', 0)
            price_24h = prices_24h_ago.get(f'{asset}_Price', current_price)
            current_value = quantity * current_price
            value_24h = quantity * price_24h
            total_balance += current_value
            total_balance_24h_ago += value_24h
            current_holdings_list.append({"Актив": asset, "Кол-во": quantity, "Стоимость (USD)": current_value})

        today_pnl_usd = total_balance - total_balance_24h_ago
        today_pnl_pct = (today_pnl_usd / total_balance_24h_ago * 100) if total_balance_24h_ago > 0 else 0
        total_pnl_usd = total_balance - total_cost_basis_from_all_tx
        total_pnl_pct = (total_pnl_usd / total_cost_basis_from_all_tx * 100) if total_cost_basis_from_all_tx > 0 else 0

        # --- Display Metrics --- 
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Баланс аккаунта (USD)", f"{total_balance:,.2f}")
        with col2: st.metric("P&L за сегодня (USD)", f"{today_pnl_usd:,.2f}", delta=f"{today_pnl_pct:.2f}%", delta_color="normal" if today_pnl_usd >= 0 else "inverse")
        with col3: st.metric("Общий P&L портфеля (USD)", f"{total_pnl_usd:,.2f}", delta=f"{total_pnl_pct:.2f}%", delta_color="normal" if total_pnl_usd >= 0 else "inverse")
        st.markdown("--- ")
        
        # --- Assets Overview --- 
        col_assets, col_chart_placeholder = st.columns([1, 2]) # Placeholder for chart area
        with col_assets:
            st.subheader("Активы")
            if current_holdings_list:
                holdings_df = pd.DataFrame(current_holdings_list)
                holdings_df["Доля (%)"] = (holdings_df["Стоимость (USD)"] / total_balance * 100).round(2) if total_balance > 0 else 0
                holdings_df = holdings_df.sort_values("Стоимость (USD)", ascending=False)
                st.dataframe(holdings_df.style.format({"Кол-во": "{:.6f}", "Стоимость (USD)": "${:,.2f}", "Доля (%)": "{:.2f}%"}), use_container_width=True)
            else: st.write("Нет активов для отображения.")

        # --- Detailed Historical Calculation (from Notebook logic) ---
        st.markdown("--- ")
        st.subheader(f"Анализ динамики портфеля")
        
        days_history_options = {"7 дней": 7, "30 дней": 30, "90 дней": 90, "180 дней": 180, "Все время": None}
        selected_days_label = st.radio("Период анализа:", days_history_options.keys(), index=3, horizontal=True, key="analysis_interval") # Default 180d
        selected_days = days_history_options[selected_days_label]
        
        report_date_viz = historical_prices.index.max()
        if selected_days:
            start_date_viz = report_date_viz - pd.Timedelta(days=selected_days)
        else:
            start_date_viz = transactions_df_raw['date'].min()
        start_date_viz = max(start_date_viz, earliest_data_date) # Ensure we don't go before data exists

        historical_prices_filtered = historical_prices[
            (historical_prices.index >= start_date_viz) &
            (historical_prices.index <= report_date_viz)
        ].copy()
        historical_prices_filtered = historical_prices_filtered.ffill().bfill().dropna(how='all')

        if historical_prices_filtered.empty:
            st.warning(f"Нет исторических данных для анализа в выбранном периоде ({selected_days_label}).")
        else:
            # Find actual purchase prices within the *full* historical data for accuracy
            buy_transactions_df['Purchase_Price_Actual'] = np.nan
            buy_transactions_df['Actual_Purchase_Time_Index'] = pd.NaT
            for index, row in buy_transactions_df.iterrows():
                asset = row['asset']
                purchase_date = row['date']
                price_col = f'{asset}_Price'
                if price_col not in historical_prices.columns: continue
                relevant_prices_index = historical_prices.index[historical_prices.index >= purchase_date]
                if not relevant_prices_index.empty:
                    actual_purchase_time_index = relevant_prices_index[0]
                    try:
                        purchase_price = historical_prices.loc[actual_purchase_time_index, price_col]
                        if pd.notna(purchase_price) and purchase_price > 0:
                            buy_transactions_df.loc[index, 'Purchase_Price_Actual'] = purchase_price
                            buy_transactions_df.loc[index, 'Actual_Purchase_Time_Index'] = actual_purchase_time_index
                    except KeyError: pass # Ignore if time index not found exactly
            
            # Drop buys where we couldn't find a valid price/time
            buy_transactions_df.dropna(subset=['Actual_Purchase_Time_Index', 'Purchase_Price_Actual'], inplace=True)

            if buy_transactions_df.empty:
                st.warning("Не найдено действительных транзакций покупки с ценами для построения графика P&L.")
            else:
                # Calculate Cumulative Cost based *only* on the valid buys found
                historical_prices_filtered['Cumulative_Cost'] = 0.0
                for _, row in buy_transactions_df.iterrows():
                    cost = row['total_cost']
                    purchase_time = row['Actual_Purchase_Time_Index']
                    historical_prices_filtered.loc[historical_prices_filtered.index >= purchase_time, 'Cumulative_Cost'] += cost

                # Calculate individual value, P&L, contribution
                purchase_value_cols = []
                purchase_pnl_cols = []
                purchase_perc_contrib_cols = []
                purchase_labels = []
                
                with st.spinner("Расчет динамики портфеля..."):
                    for index, purchase_row in buy_transactions_df.iterrows():
                        purchase_id = purchase_row['Purchase_ID']
                        asset = purchase_row['asset']
                        initial_investment = purchase_row['total_cost']
                        purchase_price = purchase_row['Purchase_Price_Actual']
                        purchase_time = purchase_row['Actual_Purchase_Time_Index']
                        price_col = f'{asset}_Price'
                        
                        if price_col not in historical_prices_filtered.columns: continue

                        value_col_name = f"Value_ID{purchase_id}_{asset}"
                        pnl_col_name = f"PnL_ID{purchase_id}_{asset}"
                        perc_contrib_col_name = f"PercContrib_ID{purchase_id}_{asset}"
                        label = f"{asset} (ID:{purchase_id}, ${initial_investment:,.2f})"

                        purchase_value_cols.append(value_col_name)
                        purchase_pnl_cols.append(pnl_col_name)
                        purchase_perc_contrib_cols.append(perc_contrib_col_name)
                        purchase_labels.append(label)

                        historical_prices_filtered[value_col_name] = 0.0
                        historical_prices_filtered[pnl_col_name] = 0.0
                        historical_prices_filtered[perc_contrib_col_name] = 0.0

                        mask = historical_prices_filtered.index >= purchase_time
                        if mask.any():
                            current_prices = historical_prices_filtered.loc[mask, price_col]
                            if pd.isna(purchase_price) or purchase_price <= 0:
                                price_ratio = pd.Series(0.0, index=current_prices.index)
                            else:
                                price_ratio = current_prices / purchase_price
                                price_ratio = price_ratio.fillna(0).replace([np.inf, -np.inf], 0)
                            current_purchase_value = initial_investment * price_ratio
                            historical_prices_filtered.loc[mask, value_col_name] = current_purchase_value
                            historical_prices_filtered.loc[mask, pnl_col_name] = current_purchase_value - initial_investment

                    # Sum up totals
                    historical_prices_filtered['Total_Value_Relative'] = historical_prices_filtered[purchase_value_cols].sum(axis=1)
                    historical_prices_filtered['Total_PnL'] = historical_prices_filtered['Total_Value_Relative'] - historical_prices_filtered['Cumulative_Cost']
                    
                    # Calculate percentage contributions
                    denom = historical_prices_filtered['Total_Value_Relative']
                    valid_denom_mask = np.abs(denom) > 1e-9
                    for pnl_col, perc_contrib_col in zip(purchase_pnl_cols, purchase_perc_contrib_cols):
                        percentage_contribution = np.zeros_like(denom)
                        percentage_contribution[valid_denom_mask] = (historical_prices_filtered.loc[valid_denom_mask, pnl_col] / denom[valid_denom_mask]) * 100
                        historical_prices_filtered[perc_contrib_col] = pd.Series(percentage_contribution, index=historical_prices_filtered.index).fillna(0)
                    
                    total_pnl_percentage = np.zeros_like(denom)
                    total_pnl_percentage[valid_denom_mask] = (historical_prices_filtered.loc[valid_denom_mask, 'Total_PnL'] / denom[valid_denom_mask]) * 100
                    historical_prices_filtered['Total_PnL_Percentage'] = pd.Series(total_pnl_percentage, index=historical_prices_filtered.index).fillna(0)

                # --- Plotting --- 
                pio.templates.default = "plotly_dark"
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    subplot_titles=(
                        f'Стоимость портфеля vs Вложенные средства',
                        'Вклад каждой инвестиции в Абсолютный P&L',
                        'Вклад P&L каждой инвестиции в % от Общей Стоимости'
                    ))

                # Chart 1
                fig.add_trace(go.Scatter(x=historical_prices_filtered.index, y=historical_prices_filtered['Total_Value_Relative'], mode='lines', name='Общая стоимость', line=dict(color='#388BFF', width=2), hovertemplate='Дата: %{x}<br>Стоимость: %{y:,.2f} USDT<extra></extra>'), row=1, col=1)
                fig.add_trace(go.Scatter(x=historical_prices_filtered.index, y=historical_prices_filtered['Cumulative_Cost'], mode='lines', name='Вложено средств', line=dict(color='#AAAAAA', dash='dash', width=1.5), hovertemplate='Дата: %{x}<br>Вложено: %{y:,.2f} USDT<extra></extra>'), row=1, col=1)
                
                # Add markers for ALL transactions in range (Buys and Sells)
                transactions_in_plot_range = transactions_df_raw[transactions_df_raw['date'] >= historical_prices_filtered.index.min()]
                buy_markers = transactions_in_plot_range[transactions_in_plot_range['type'] == 'buy']
                sell_markers = transactions_in_plot_range[transactions_in_plot_range['type'] == 'sell']
                
                buy_marker_times = []
                buy_marker_values = []
                buy_marker_texts = []
                for _, row in buy_markers.iterrows():
                    # Find closest index <= transaction date
                    marker_time_idx = historical_prices_filtered.index[historical_prices_filtered.index <= row['date']]
                    if not marker_time_idx.empty:
                        marker_time = marker_time_idx[-1]
                        buy_marker_times.append(marker_time)
                        buy_marker_values.append(historical_prices_filtered.loc[marker_time, 'Total_Value_Relative'])
                        buy_marker_texts.append(f"<b>Покупка {row['asset']}</b><br>Дата: {row['date'].strftime('%Y-%m-%d %H:%M')}<br>Кол-во: {row['quantity']:.6f}<br>Цена: ${row['price']:.2f}<br>Сумма: ${row['total_cost']:,.2f}<extra></extra>")
                if buy_marker_times:
                     fig.add_trace(go.Scatter(x=buy_marker_times, y=buy_marker_values, mode='markers', name='Покупки', marker=dict(color='#00BFFF', size=7, symbol='triangle-up', line=dict(color='white', width=1)), hoverinfo='text', text=buy_marker_texts), row=1, col=1)

                sell_marker_times = []
                sell_marker_values = []
                sell_marker_texts = []
                for _, row in sell_markers.iterrows():
                     marker_time_idx = historical_prices_filtered.index[historical_prices_filtered.index <= row['date']]
                     if not marker_time_idx.empty:
                         marker_time = marker_time_idx[-1]
                         sell_marker_times.append(marker_time)
                         sell_marker_values.append(historical_prices_filtered.loc[marker_time, 'Total_Value_Relative'])
                         sell_marker_texts.append(f"<b>Продажа {row['asset']}</b><br>Дата: {row['date'].strftime('%Y-%m-%d %H:%M')}<br>Кол-во: {row['quantity']:.6f}<br>Цена: ${row['price']:.2f}<br>Сумма: ${row['total_cost']:,.2f}<extra></extra>")
                if sell_marker_times:
                    fig.add_trace(go.Scatter(x=sell_marker_times, y=sell_marker_values, mode='markers', name='Продажи', marker=dict(color='#FF6347', size=7, symbol='triangle-down', line=dict(color='white', width=1)), hoverinfo='text', text=sell_marker_texts), row=1, col=1)

                # Chart 2 - Absolute P&L Stack
                num_colors = len(purchase_labels)
                colors = px.colors.qualitative.T10
                if num_colors > len(colors): colors = colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)]
                color_map = {label: colors[i] for i, label in enumerate(purchase_labels)}

                for i, (pnl_col, label) in enumerate(zip(purchase_pnl_cols, purchase_labels)):
                    color = color_map[label]
                    fig.add_trace(go.Scatter(x=historical_prices_filtered.index, y=historical_prices_filtered[pnl_col].fillna(0), mode='lines', name=label, stackgroup='pnl_absolute', line=dict(width=0), fillcolor=color, hovertemplate=f'<b>{label}</b><br>Дата: %{{x}}<br>Абс. P&L: %{{y:,.2f}} USDT<extra></extra>', legendgroup=label, showlegend=False), row=2, col=1)
                fig.add_trace(go.Scatter(x=historical_prices_filtered.index, y=historical_prices_filtered['Total_PnL'], mode='lines', name='Общий P&L', line=dict(color='white', dash='dot', width=2), hovertemplate='<b>Общий P&L</b><br>Дата: %{x}<br>P&L: %{y:,.2f}} USDT<extra></extra>', legendgroup="total_pnl"), row=2, col=1)
                fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="grey", row=2, col=1)

                # Chart 3 - Percentage P&L Stack
                for i, (perc_contrib_col, label) in enumerate(zip(purchase_perc_contrib_cols, purchase_labels)):
                    color = color_map[label]
                    fig.add_trace(go.Scatter(x=historical_prices_filtered.index, y=historical_prices_filtered[perc_contrib_col].fillna(0), mode='lines', name=label, stackgroup='pnl_percentage', line=dict(width=0), fillcolor=color, hovertemplate=f'<b>{label}</b><br>Дата: %{{x}}<br>% Вклад P&L: %{{y:.2f}}%<extra></extra>', legendgroup=label, showlegend=False), row=3, col=1)
                fig.add_trace(go.Scatter(x=historical_prices_filtered.index, y=historical_prices_filtered['Total_PnL_Percentage'], mode='lines', name='Общий P&L %', line=dict(color='white', dash='dot', width=2), hovertemplate='<b>Общий P&L %</b><br>Дата: %{x}<br>P&L: %{y:.2f}%<extra></extra>', legendgroup="total_pnl_perc"), row=3, col=1)
                fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="grey", row=3, col=1)

                # Layout updates
                fig.update_layout(
                    height=800, hovermode='x unified',
                    legend=dict(traceorder='normal', orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
                    margin=dict(l=50, r=20, t=60, b=50)
                )
                fig.update_xaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, gridcolor='rgba(128, 128, 128, 0.2)')
                fig.update_yaxes(showline=True, linewidth=1, linecolor='grey', mirror=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=False)
                fig.update_yaxes(title_text="Стоимость (USDT)", tickprefix="$", row=1, col=1)
                fig.update_yaxes(title_text="Абс. P&L (USDT)", tickprefix="$", row=2, col=1)
                fig.update_yaxes(title_text="% Вклад P&L", ticksuffix="%", row=3, col=1)
                fig.update_xaxes(title_text="Дата", row=3, col=1)

                st.plotly_chart(fig, use_container_width=True)

    # Страница управления активами и транзакциями
    elif st.session_state.active_page == "Управление активами":
        render_transactions_manager(st.session_state.username, price_data, assets)
    
    # Подключение страниц из app_pages.py
    elif st.session_state.active_page == "Анализ портфеля":
        st.header("Анализ эффективности портфеля")
        st.markdown("Здесь вы можете проанализировать, как бы изменилась стоимость вашего портфеля, \\n        если бы вы следовали различным инвестиционным стратегиям, основанным на вашей истории транзакций.")

        # --- Настройки Анализа --- 
        st.subheader("Параметры анализа")
        # Определяем даты по умолчанию
        today_date = datetime.now().date()
        default_start_date = today_date - timedelta(days=180)

        col1, col2 = st.columns(2)
        with col1:
             start_date = st.date_input("Начальная дата анализа", value=default_start_date, max_value=today_date - timedelta(days=1))
             commission_input = st.number_input("Комиссия за ребалансировку (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.01, format="%.3f")
             rebalance_interval = st.number_input("Интервал ребалансировки (дни)", min_value=1, value=20, step=1)
        with col2:
             end_date = st.date_input("Конечная дата анализа", value=today_date, min_value=start_date + timedelta(days=1))
             bank_apr_input = st.number_input("Годовая ставка банка (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5, format="%.1f")
             drl_rebalance_interval = st.number_input("Интервал ребалансировки DRL (дни)", min_value=1, value=20, step=1)
        
        # Конвертация процентов в доли
        commission_rate_analysis = commission_input / 100.0
        bank_apr_analysis = bank_apr_input / 100.0

        # --- Пути к данным и моделям (ВАЖНО: Настроить под ваше окружение) --- 
        # Эти пути должны быть доступны из места запуска Streamlit
        # Лучше использовать относительные пути от корня проекта или абсолютные пути
        # Пример: использовать os.path.dirname(__file__) для относительных путей 
        # ИЛИ передавать через переменные окружения / конфигурационный файл
        # *** Настройте эти пути! ***
        data_path_analysis = "data" # Пример: папка data в корне проекта
        drl_models_dir_analysis = "notebooks/trained_models" # Пример: папка моделей
        st.caption(f"Путь к данным: {os.path.abspath(data_path_analysis)}, Путь к моделям: {os.path.abspath(drl_models_dir_analysis)}")

        # --- Получение данных пользователя --- 
        user_transactions_raw = get_user_transactions(st.session_state.username)

        if not user_transactions_raw:
             st.warning("История транзакций не найдена. Пожалуйста, добавьте транзакции в разделе 'Управление активами'.")
        else:
             # --- Подготовка данных транзакций --- 
             try:
                  transactions_df_analysis = pd.DataFrame(user_transactions_raw)

                  # --- START ADDED/MODIFIED CODE ---
                  # Ensure necessary columns are numeric BEFORE calculating total_cost
                  transactions_df_analysis['quantity'] = pd.to_numeric(transactions_df_analysis['quantity'], errors='coerce')
                  transactions_df_analysis['price'] = pd.to_numeric(transactions_df_analysis['price'], errors='coerce')
                  if 'fee' in transactions_df_analysis.columns:
                       transactions_df_analysis['fee'] = pd.to_numeric(transactions_df_analysis.get('fee', 0), errors='coerce')
                  else:
                       transactions_df_analysis['fee'] = 0 # Add fee column with 0 if it doesn't exist
                  # Use assignment instead of inplace on slice
                  transactions_df_analysis['fee'] = transactions_df_analysis['fee'].fillna(0)
                  # Calculate total_cost if missing
                  if 'total_cost' not in transactions_df_analysis.columns:
                      transactions_df_analysis['total_cost'] = (transactions_df_analysis['quantity'] * transactions_df_analysis['price']) + transactions_df_analysis['fee']
                      # Handle potential NaNs from calculation (if quantity or price were NaN)
                      transactions_df_analysis['total_cost'] = transactions_df_analysis['total_cost'].fillna(0)
                  else:
                       # Ensure existing total_cost is numeric and handle NaNs
                       transactions_df_analysis['total_cost'] = pd.to_numeric(transactions_df_analysis['total_cost'], errors='coerce')
                       # Use assignment instead of inplace on slice
                       transactions_df_analysis['total_cost'] = transactions_df_analysis['total_cost'].fillna(0)
                  # Ensure fee is handled (copied from calculation block for consistency)
                  if 'fee' not in transactions_df_analysis.columns:
                       transactions_df_analysis['fee'] = 0
                  transactions_df_analysis['fee'] = pd.to_numeric(transactions_df_analysis.get('fee', 0), errors='coerce')
                  # Use assignment instead of inplace on slice
                  transactions_df_analysis['fee'] = transactions_df_analysis['fee'].fillna(0)
                  # --- END ADDED/MODIFIED CODE ---

                  # Переименование колонок под формат run_portfolio_analysis
                  rename_map = {
                       "date": "Дата_Транзакции",
                       "asset": "Актив",
                       "type": "Тип",
                       "quantity": "Количество",
                       "total_cost": "Общая стоимость"
                  }
                  # Проверяем наличие необходимых колонок (ключей из rename_map) в DataFrame
                  required_cols = list(rename_map.keys())
                  missing_cols = [col for col in required_cols if col not in transactions_df_analysis.columns]
                  if missing_cols:
                      st.error(f"Ошибка: Не все необходимые колонки найдены в истории транзакций пользователя. Отсутствуют: {missing_cols}")
                  else:
                     # Выбираем только нужные колонки ПЕРЕД переименованием
                     transactions_df_analysis = transactions_df_analysis[required_cols].rename(columns=rename_map)

                     # --- START ADDED CODE ---
                     # Преобразуем значения в колонке 'Тип'
                     type_value_map = {'buy': 'Покупка', 'sell': 'Продажа'}
                     # Используем .get для безопасного маппинга, оставляя другие значения без изменений
                     transactions_df_analysis['Тип'] = transactions_df_analysis['Тип'].map(lambda x: type_value_map.get(x, x))
                     # --- END ADDED CODE ---

                     # Преобразование типов на всякий случай
                     transactions_df_analysis['Дата_Транзакции'] = pd.to_datetime(transactions_df_analysis['Дата_Транзакции'], errors='coerce')
                     transactions_df_analysis['Количество'] = pd.to_numeric(transactions_df_analysis['Количество'], errors='coerce')
                     # Общая стоимость может быть NaN для продаж, это нормально
                     transactions_df_analysis['Общая стоимость'] = pd.to_numeric(transactions_df_analysis['Общая стоимость'].astype(str).str.replace(',', '', regex=False).str.replace('$', '', regex=False), errors='coerce')
                     
                     transactions_df_analysis.dropna(subset=['Дата_Транзакции', 'Актив', 'Тип'], inplace=True)
                     # Проверка на наличие валидных транзакций после обработки
                     if transactions_df_analysis.empty:
                          st.warning("Не найдено валидных транзакций после обработки. Проверьте формат данных в истории.")
                     else:
                          st.dataframe(transactions_df_analysis.head(), use_container_width=True)

                          # --- Кнопка Запуска --- 
                          if st.button("🚀 Запустить анализ портфеля", use_container_width=True):
                              with st.spinner("Выполняется анализ... Это может занять некоторое время."):
                                  try:
                                     results_df, fig = run_portfolio_analysis(
                                         transactions_df=transactions_df_analysis,
                                         start_date_str=start_date.strftime('%Y-%m-%d'),
                                         end_date_str=end_date.strftime('%Y-%m-%d'),
                                         data_path=data_path_analysis,
                                         drl_models_dir=drl_models_dir_analysis,
                                         bank_apr=bank_apr_analysis,
                                         commission_rate=commission_rate_analysis,
                                         rebalance_interval_days=rebalance_interval,
                                         drl_rebalance_interval_days=drl_rebalance_interval
                                     )
                                     # Сохраняем результаты в session_state для отображения
                                     if results_df is not None and fig is not None:
                                          st.session_state.analysis_results = results_df
                                          st.session_state.analysis_figure = fig
                                          st.success("Анализ успешно завершен!")
                                     else:
                                          st.error("Ошибка во время анализа. Проверьте консоль для деталей.")
                                          if 'analysis_results' in st.session_state: del st.session_state['analysis_results']
                                          if 'analysis_figure' in st.session_state: del st.session_state['analysis_figure']
                                  except Exception as e:
                                      st.error(f"Произошла непредвиденная ошибка: {e}")
                                      traceback.print_exc()
                                      if 'analysis_results' in st.session_state: del st.session_state['analysis_results']
                                      if 'analysis_figure' in st.session_state: del st.session_state['analysis_figure']
             except Exception as e:
                 st.error(f"Ошибка при подготовке данных транзакций: {e}")
                 traceback.print_exc()

        # --- Отображение Результатов --- 
        if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
             st.subheader("Результаты анализа")
             st.dataframe(st.session_state.analysis_results, use_container_width=True)
        
        if 'analysis_figure' in st.session_state and st.session_state.analysis_figure is not None:
             st.subheader("График сравнения стратегий")
             st.pyplot(st.session_state.analysis_figure)

    # Страница управления активами и транзакциями
    elif st.session_state.active_page == "Управление активами":
        render_transactions_manager(st.session_state.username, price_data, assets)
    
    # Подключение страниц из app_pages.py
    elif st.session_state.active_page == "Dashboard":
        render_dashboard(st.session_state.username, price_data, model_returns, model_actions, assets)
    
    elif st.session_state.active_page == "Portfolio Optimization":
        render_portfolio_optimization(st.session_state.username, price_data, assets)
    
    elif st.session_state.active_page == "Model Training":
        render_model_training(st.session_state.username, price_data, assets)
    
    elif st.session_state.active_page == "Model Comparison":
        render_model_comparison(st.session_state.username, model_returns, model_actions, price_data)
    
    elif st.session_state.active_page == "Backtest Results":
        render_backtest(st.session_state.username, model_returns, price_data)
    
    elif st.session_state.active_page == "About":
        render_about() 


'''
poetry run streamlit run auth_app.py
'''