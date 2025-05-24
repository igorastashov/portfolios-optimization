import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta, date, time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
import json
import requests
import time

# --- ADDED: Import pipeline ---
import torch

# --- Backend API Configuration ---
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

# Импорт модулей приложения (These will be gradually replaced by API calls)
from app_pages import (
    render_dashboard, 
    render_portfolio_optimization, 
    render_model_training, 
    render_model_comparison, 
    render_backtest,
    render_account_dashboard,
    render_transactions_manager,
    render_about
)

# --- Helper Functions (To be refactored or removed) ---

# --- NEW: Authentication functions using FastAPI backend ---
def initialize_users_file():
    # This function is no longer needed as user management is handled by the backend.
    pass

def register_user(username, password, email):
    """Registers a new user by calling the backend API."""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/auth/register",
            json={"username": username, "email": email, "password": password}
        )
        if response.status_code == 201: # FastAPI returns 201 for created
            return True, "Регистрация прошла успешно! Теперь вы можете войти."
        else:
            error_detail = response.json().get("detail", "Неизвестная ошибка регистрации.")
            return False, f"Ошибка регистрации: {error_detail}"
    except requests.exceptions.RequestException as e:
        return False, f"Ошибка соединения с сервером: {e}"

def authenticate_user(username, password):
    """Authenticates a user by calling the backend API."""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/auth/login/access-token",
            data={"username": username, "password": password} # FastAPI expects form data for token endpoint
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data.get("access_token")
            st.session_state.refresh_token = data.get("refresh_token")
            st.session_state.token_type = data.get("token_type", "bearer")
            return True, "Вход выполнен успешно!"
        elif response.status_code == 401:
             error_detail = response.json().get("detail", "Неверные учетные данные.")
             return False, f"Ошибка входа: {error_detail}"
        else:
            error_detail = response.json().get("detail", "Неизвестная ошибка входа.")
            return False, f"Ошибка входа (код {response.status_code}): {error_detail}"
    except requests.exceptions.RequestException as e:
        return False, f"Ошибка соединения с сервером: {e}"

def get_user_info(username): # username argument might become redundant if using token
    """Fetches user information from the backend API using the stored token."""
    if "access_token" not in st.session_state:
        return None # Or raise an error, or redirect to login

    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.get(f"{BACKEND_API_URL}/users/me", headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            # Adapt backend user data to the structure previously expected by Streamlit if necessary
            # For example, if Streamlit expects 'created_at', 'last_login'
            return {
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "is_active": user_data.get("is_active"),
                "is_superuser": user_data.get("is_superuser"),
                "id": user_data.get("id"),
                # Placeholder for fields that might have been in the old local user file
                "created_at": user_data.get("created_at", datetime.now().isoformat()), # Or fetch from backend if available
                "last_login": user_data.get("last_login", datetime.now().isoformat())  # Or fetch from backend if available
            }
        else:
            st.error(f"Не удалось получить информацию о пользователе (код {response.status_code}): {response.text}")
            # Potentially handle token refresh here if 401 Unauthorized
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с сервером при получении информации о пользователе: {e}")
        return None
# --- End NEW Authentication functions ---


# --- Functions to be refactored to use API calls ---
def get_user_portfolios(username): # To be refactored
    # This function might be deprecated if we only manage one portfolio per user via /portfolios/me
    st.warning("get_user_portfolios needs to be refactored or deprecated.")
    return []

def get_user_portfolio(username, portfolio_name): # To be refactored
    # This function might be deprecated if we only manage one portfolio per user via /portfolios/me
    st.warning("get_user_portfolio needs to be refactored or deprecated.")
    return None

def get_portfolio_with_quantities(username): # Refactored
    """Fetches the current user's portfolio summary from the backend API."""
    if "access_token" not in st.session_state:
        st.error("User not authenticated. Cannot fetch portfolio.")
        return {"quantities": {}, "avg_prices": {}, "current_prices": {}}

    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        # Assuming backend provides a summary endpoint for the authenticated user's portfolio
        # This endpoint should return quantities, average buy prices, and ideally current prices
        response = requests.get(f"{BACKEND_API_URL}/portfolios/me/summary", headers=headers)
        if response.status_code == 200:
            data = response.json()
            # Expected backend response structure:
            # {
            #   "assets": [
            #     { "ticker": "BTCUSDT", "quantity": 0.5, "average_buy_price": 30000, "current_price": 60000 },
            #     { "ticker": "ETHUSDT", "quantity": 10, "average_buy_price": 2000, "current_price": 3000 }
            #   ],
            #   "total_value": 33000,
            #   "total_pnl": ...
            # }
            # We need to transform this into the old structure expected by the Streamlit page for now.
            quantities = {}
            avg_prices = {}
            current_prices = {} # Add a dictionary for current prices from backend
            for asset_data in data.get("assets", []):
                ticker = asset_data.get("ticker")
                if ticker:
                    quantities[ticker] = asset_data.get("quantity", 0)
                    avg_prices[ticker] = asset_data.get("average_buy_price", 0)
                    current_prices[ticker] = asset_data.get("current_price", 0) # Store current price
            
            # The Streamlit UI calculates P&L and total value, so we primarily need these raw components.
            # If the backend provides total_value, total_pnl, daily_change, it can be used directly too.
            st.session_state.portfolio_summary_from_api = data # Store full summary for potential later use
            return {"quantities": quantities, "avg_prices": avg_prices, "current_prices": current_prices}
        elif response.status_code == 401:
            st.error("Сессия истекла или недействительна. Пожалуйста, войдите снова.")
            logout() # Force logout
            st.rerun()
            return {"quantities": {}, "avg_prices": {}, "current_prices": {}}
        else:
            st.error(f"Не удалось загрузить данные портфеля (код {response.status_code}): {response.text}")
            return {"quantities": {}, "avg_prices": {}, "current_prices": {}}
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с сервером при загрузке портфеля: {e}")
        return {"quantities": {}, "avg_prices": {}, "current_prices": {}}

def get_user_transactions(username): # Refactored
    """Fetches user's transactions from the backend API."""
    if "access_token" not in st.session_state:
        st.error("User not authenticated. Cannot fetch transactions.")
        return []

    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        # Assuming backend provides transactions for the authenticated user
        response = requests.get(f"{BACKEND_API_URL}/portfolios/me/transactions", headers=headers)
        if response.status_code == 200:
            transactions_list = response.json()
            # Convert date strings from API (e.g., ISO format) to datetime objects
            for tx in transactions_list:
                if 'created_at' in tx and isinstance(tx['created_at'], str):
                    tx['date'] = datetime.fromisoformat(tx['created_at'].replace('Z', '+00:00'))
                # Ensure other fields like quantity, price, fee are correct type (e.g. float)
                tx['quantity'] = float(tx.get('quantity', 0))
                tx['price'] = float(tx.get('price', 0))
                tx['fee'] = float(tx.get('fee', 0))
                # The backend model uses 'asset_ticker', Streamlit old code used 'asset'
                if 'asset_ticker' in tx and 'asset' not in tx:
                    tx['asset'] = tx['asset_ticker']
                if 'transaction_type' in tx and 'type' not in tx:
                     tx['type'] = tx['transaction_type']
            return transactions_list
        elif response.status_code == 401:
            st.error("Сессия истекла или недействительна. Пожалуйста, войдите снова.")
            logout()
            st.rerun()
            return []
        else:
            st.error(f"Не удалось загрузить транзакции (код {response.status_code}): {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с сервером при загрузке транзакций: {e}")
        return []

def update_user_portfolio(username, portfolio_data): # To be refactored
    # This function is likely superseded by adding individual transactions via API
    st.warning("update_user_portfolio is likely deprecated. Add transactions instead.")
    pass
# --- End Functions to be refactored ---

# --- END HELPER FUNCTIONS ---

# <<< NEW: Dummy function for fetching news - REPLACE WITH REAL IMPLEMENTATION >>>
def fetch_dummy_news(asset_ticker):
    """Возвращает пример текста новости. Замените реальной логикой получения новостей."""
    st.warning(f"Примечание: Используются **демонстрационные** новостные данные для {asset_ticker}.")
    # Пример текста
    if asset_ticker == "BTCUSDT":
        return f"Bitcoin (BTCUSDT) price surged above $70,000 amid growing institutional interest. Several large investment firms announced new Bitcoin ETF filings. However, some analysts warn of potential volatility ahead of the upcoming halving event. The overall market sentiment remains cautiously optimistic. Key players like MicroStrategy continue to add to their BTC holdings."
    elif asset_ticker == "ETHUSDT":
        return f"Ethereum (ETHUSDT) saw moderate gains, following the general market trend. Discussions around the potential approval of an Ethereum Spot ETF continue, but regulatory uncertainty persists. Network activity remains high, driven by DeFi and NFT sectors. Vitalik Buterin recently commented on the importance of Layer 2 scaling solutions."
    else:
        return f"General market news for {asset_ticker}: Crypto markets experienced mixed trading today. Regulatory developments in the US and Asia are being closely watched by investors. Stablecoin regulations are also a hot topic. Overall trading volume was moderate."
# <<< END DUMMY NEWS FUNCTION >>>

# Конфигурация страницы
st.set_page_config(
    page_title="Портфельный Оптимизатор Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# <<< Add the missing function definition here >>>
# Function to load combined data with caching
# Use a parameter that changes upon update to invalidate cache
@st.cache_data
def load_combined_data_cached(update_trigger):
    """Loads combined data from CSV, using update_trigger for cache invalidation."""
    print(f"Cache Trigger: {update_trigger}. Loading combined data...") # Debug print
    combined_data_path = os.path.join("data", "data_compare_eda.csv")
    if os.path.exists(combined_data_path):
        try:
            combined_df = pd.read_csv(combined_data_path, index_col=0, parse_dates=True)
            # Ensure index is DatetimeIndex
            if not isinstance(combined_df.index, pd.DatetimeIndex):
                 combined_df.index = pd.to_datetime(combined_df.index, errors='coerce')
                 combined_df = combined_df.dropna(axis=0, subset=[combined_df.index.name]) # Drop rows if date parse failed
            combined_df.sort_index(inplace=True)
            print(f"Loaded {combined_data_path}, shape: {combined_df.shape}")
            return combined_df
        except Exception as e:
            st.error(f"Ошибка при загрузке {combined_data_path}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Файл {combined_data_path} не найден.")
        return pd.DataFrame()

# Инициализация файла пользователей (No longer needed)
# initialize_users_file()

# Инициализация состояния сессии
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'access_token' not in st.session_state: st.session_state.access_token = ""
if 'active_page' not in st.session_state: st.session_state.active_page = "Главная"
if 'user_info' not in st.session_state: st.session_state.user_info = None
if 'portfolio_summary' not in st.session_state: st.session_state.portfolio_summary = None
if 'transactions' not in st.session_state: st.session_state.transactions = []
if 'assets' not in st.session_state: st.session_state.assets = [] # Список доступных тикеров

# Функция для выхода из аккаунта
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.active_page = "Главная"
    st.session_state.login_message = "Вы вышли из системы"
    # Clear analysis results on logout
    if 'analysis_results' in st.session_state: del st.session_state['analysis_results']
    if 'analysis_figure' in st.session_state: del st.session_state['analysis_figure']
    # Clear API tokens
    if 'access_token' in st.session_state: del st.session_state['access_token']
    if 'refresh_token' in st.session_state: del st.session_state['refresh_token']
    if 'token_type' in st.session_state: del st.session_state['token_type']

# Загрузка данных для всего приложения
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_app_data(): # Renamed to avoid conflict if there was an old load_data
    """Loads data required globally by the application, e.g., list of all available assets."""
    headers = {}
    if "access_token" in st.session_state: # Send token if available, some endpoints might be protected
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    
    all_assets_list = []
    try:
        response = requests.get(f"{BACKEND_API_URL}/assets/", headers=headers)
        response.raise_for_status()
        assets_raw = response.json()
        all_assets_list = [asset.get("ticker_symbol") for asset in assets_raw if asset.get("ticker_symbol")]
    except requests.exceptions.RequestException as e:
        st.error(f"Не удалось загрузить список всех активов: {e}")
        # Fallback or empty list
    
    # For price_data: This was used to get current prices. 
    # The new /portfolios/me/summary provides current prices for portfolio assets.
    # For other assets (e.g., for selection in optimization), we might need a different approach.
    # For now, let's assume `all_assets_list` is what we need for asset dropdowns.
    # A comprehensive `price_data` DataFrame with historical data for ALL assets is too large to load globally.
    # It should be fetched on-demand by specific pages.
    # So, the old `price_data` DataFrame is largely replaced by targeted API calls.
    
    # What was price_data used for? Primarily:
    # 1. Populating asset selectors (now use all_assets_list)
    # 2. Getting current price for P&L in Мой кабинет (now from /me/summary)
    # 3. Getting current price for P&L in ЕТА (now from /me/summary or historical API)
    # 4. Potentially by optimization/analysis pages (these should fetch their own required data)
    
    # Therefore, the global price_data, model_returns, model_actions can be deprecated or changed.
    # We will return the list of asset tickers for now.
    st.session_state.all_available_tickers = all_assets_list

    # model_returns and model_actions were for model comparison.
    # This data should be fetched from specific backend endpoints when on those pages.
    # For now, returning empty placeholders for them.
    model_returns_placeholder = pd.DataFrame()
    model_actions_placeholder = pd.DataFrame()

    # The global price_data pandas DataFrame is problematic with API-first approach.
    # We'll return the list of tickers, and pages can fetch prices as needed.
    # For compatibility, some old sections might try to use `assets` which was price_data.columns
    # We'll set st.session_state.assets directly from all_available_tickers
    st.session_state.assets = all_assets_list

    # Return empty dataframes for the old tuple to avoid breaking too much code immediately
    # These will be gradually removed.
    return pd.DataFrame(columns=all_assets_list), model_returns_placeholder, model_actions_placeholder

# Загрузка данных (Call the new function)
# price_data, model_returns, model_actions = load_data() # Old call
price_data_global, model_returns_global, model_actions_global = load_app_data() 

# Получение списка доступных активов
# assets = price_data.columns.tolist() if not price_data.empty else [] # Old way
# Now use the session state set by load_app_data or the first element of the returned tuple
assets = st.session_state.get("assets", [])
if not assets and not price_data_global.empty:
    assets = price_data_global.columns.tolist()

# Основной заголовок приложения
st.title("Investment Portfolio Monitoring & Optimization System")

# Страница аутентификации
if not st.session_state.authenticated:
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
                    st.session_state.authenticated = True
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
    page_options = [
        "Мой кабинет", 
        "Данные и Анализ", 
        "Исследование", # Added "Исследование"
        "Управление активами", 
        "Единый торговый аккаунт", 
        "Анализ портфеля", 
        "Рекомендации"
    ]
    
    # Устанавливаем индекс для radio на основе текущей активной страницы в состоянии сессии
    try:
        if st.session_state.active_page not in page_options:
            st.session_state.active_page = page_options[0]
        current_page_index = page_options.index(st.session_state.active_page)
    except ValueError:
        current_page_index = 0
        st.session_state.active_page = page_options[0]

    selected_page = st.sidebar.radio(
        "Выберите раздел",
        page_options,
        index=current_page_index,
        key="main_nav_radio"
    )
    
    if selected_page != st.session_state.active_page:
        st.session_state.active_page = selected_page
        st.rerun()

    # Страница личного кабинета пользователя
    if st.session_state.active_page == "Мой кабинет":
        st.header("Личный кабинет")

        # Получение информации о пользователе (already uses API)
        user_info = get_user_info(st.session_state.username)

        if user_info:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Ваш профиль")
                st.write(f"**Email:** {user_info.get('email', 'Не указан')}")
                # Assuming 'created_at' and 'last_login' are provided by backend's /users/me or are placeholders
                created_at_str = user_info.get('created_at', '')
                last_login_str = user_info.get('last_login', '')
                try:
                    created_at_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00')) if created_at_str else None
                    last_login_dt = datetime.fromisoformat(last_login_str.replace('Z', '+00:00')) if last_login_str else None
                    st.write(f"**Дата регистрации:** {created_at_dt.strftime('%Y-%m-%d %H:%M:%S') if created_at_dt else 'Неизвестно'}")
                    st.write(f"**Последний вход:** {last_login_dt.strftime('%Y-%m-%d %H:%M:%S') if last_login_dt else 'Неизвестно'}")
                except ValueError:
                    st.write(f"**Дата регистрации:** {created_at_str}")
                    st.write(f"**Последний вход:** {last_login_str}")

            st.subheader("Ваши портфели")

            # Получение данных о портфеле пользователя через API
            portfolio_summary = get_portfolio_with_quantities(st.session_state.username)
            portfolio_quantities = portfolio_summary.get("quantities", {})
            portfolio_avg_prices = portfolio_summary.get("avg_prices", {})
            portfolio_current_prices = portfolio_summary.get("current_prices", {}) # Use current prices from API

            has_assets = portfolio_quantities and any(q > 0 for q in portfolio_quantities.values())

            if has_assets:
                portfolio_items = []
                total_portfolio_value = 0
                total_invested_value = 0 # For calculating total P&L

                # --- Temporarily load all price data for assets in portfolio for 24h change --- 
                # This should be replaced by a more efficient backend call for 24h price data
                assets_in_portfolio = [asset for asset, quantity in portfolio_quantities.items() if quantity > 0]
                # For now, we assume portfolio_current_prices has the latest prices
                # The 24h change calculation will need adjustment or a dedicated API field.
                # Placeholder for price_data which used to be global
                # For now, we will rely on current_price from the API summary for most calculations.
                # The 24h change metric might be inaccurate or disabled until price_data fetching is fully refactored.
                
                # --- Fallback: if portfolio_current_prices is empty, try to use global price_data (old way, to be removed) ---
                # This is a temporary bridge. The API should ideally provide all necessary price points.
                temp_price_data_holder = {}
                if not portfolio_current_prices and not price_data_global.empty: # price_data is from global load_data()
                    st.warning("Using stale global price_data for current prices as API did not provide them. This will be removed.")
                    for asset in assets_in_portfolio:
                        if asset in price_data_global.columns:
                            temp_price_data_holder[asset] = price_data_global[asset].iloc[-1]
                            # For 24h change, we'd need price_data[asset].iloc[-2] too
                elif not portfolio_current_prices:
                    st.error("Current prices not available from API and no fallback global price data. Values will be zero.")
                # --- End Fallback ---

                for asset, quantity in portfolio_quantities.items():
                    if quantity > 0:
                        current_price = portfolio_current_prices.get(asset, temp_price_data_holder.get(asset, 0))
                        avg_buy_price = portfolio_avg_prices.get(asset, 0)

                        current_value = quantity * current_price
                        invested_value = quantity * avg_buy_price
                        profit_loss = current_value - invested_value
                        profit_loss_percent = (profit_loss / invested_value * 100) if invested_value > 0 else 0

                        total_portfolio_value += current_value
                        total_invested_value += invested_value

                        portfolio_items.append({
                            "Актив": asset,
                            "Количество": quantity,
                            "Средняя цена покупки": avg_buy_price,
                            "Текущая цена": current_price,
                            "Текущая стоимость": current_value,
                            "P&L": profit_loss,
                            "P&L (%)": profit_loss_percent
                        })
                
                total_profit_loss = total_portfolio_value - total_invested_value

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Общая стоимость портфеля", f"${total_portfolio_value:,.2f}")
                with col2:
                    delta_value = (total_profit_loss / total_invested_value * 100) if total_invested_value > 0 else 0
                    st.metric("Общий P&L", 
                              f"${total_profit_loss:,.2f}", 
                              delta=f"{delta_value:.2f}%" if total_invested_value else None,
                              delta_color="normal" if total_profit_loss >=0 else "inverse")
                
                with col3:
                    # 24h Change: This needs to be provided by the backend or fetched efficiently.
                    # Using a placeholder or a note that it's unavailable for now.
                    # If st.session_state.portfolio_summary_from_api has this info, use it:
                    change_24h_value_abs = st.session_state.get('portfolio_summary_from_api', {}).get('total_value_24h_change_abs')
                    change_24h_value_pct = st.session_state.get('portfolio_summary_from_api', {}).get('total_value_24h_change_pct')
                    if change_24h_value_abs is not None and change_24h_value_pct is not None:
                         st.metric("Изменение за 24ч", 
                                   f"${change_24h_value_abs:,.2f}", 
                                   delta=f"{change_24h_value_pct:.2f}%",
                                   delta_color="normal" if change_24h_value_abs >= 0 else "inverse")
                    else:
                        st.metric("Изменение за 24ч", "N/A", delta="Требуются данные API", delta_color="off")

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

        # --- Imports for this page (some might be already at top) ---
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

        # --- Load Historical Data (To be replaced by API call) ---
        @st.cache_data(ttl=1800)
        def load_and_preprocess_historical_data_uta_api(assets_list, start_date_str=None, end_date_str=None, interval="1h"):
            st.info(f"UTA: Fetching historical data for {assets_list} from {start_date_str} to {end_date_str} ({interval}) from backend API...")
            if "access_token" not in st.session_state:
                st.error("Authentication token not found. Cannot fetch historical data.")
                return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)
            
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            params = {
                "tickers": assets_list,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "interval": interval
            }
            
            try:
                response = requests.get(f"{BACKEND_API_URL}/assets/market-data/historical", headers=headers, params=params)
                response.raise_for_status() # Raises an exception for 4XX/5XX errors
                
                api_data = response.json()
                if not api_data:
                    st.warning("No historical data returned from API.")
                    return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)

                # Convert API data (list of dicts) to DataFrame expected by the chart
                # Expected format: DateTimeIndex, columns: {ASSET_TICKER}_Price
                # API returns: list of {"ticker": "X", "timestamp": "T", "close": Y, ...}
                
                df_list = []
                for item in api_data:
                    df_list.append({
                        'date_index': pd.to_datetime(item['timestamp']),
                        'ticker': item['ticker'],
                        'price': item['close'] # Using 'close' price for the '_Price' column
                    })
                if not df_list:
                    st.warning("Processed API data is empty.")
                    return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)

                temp_df = pd.DataFrame(df_list)
                # Pivot to get tickers as columns: {ASSET_TICKER}_Price
                pivot_df = temp_df.pivot_table(index='date_index', columns='ticker', values='price')
                # Rename columns to f'{ticker}_Price'
                pivot_df.columns = [f'{col}_Price' for col in pivot_df.columns]
                
                # Ensure index is sorted
                pivot_df.sort_index(inplace=True)
                
                min_date_from_data = pivot_df.index.min() if not pivot_df.empty else pd.Timestamp.now().tz_localize(None)
                st.success(f"Successfully fetched and processed {len(pivot_df)} data points for {len(assets_list)} assets.")
                return pivot_df, min_date_from_data

            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error fetching historical data: {http_err} - {response.text}")
                return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)
            except requests.exceptions.RequestException as req_err:
                st.error(f"Request error fetching historical data: {req_err}")
                return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)
            except Exception as e:
                st.error(f"Error processing historical data from API: {e}")
                traceback.print_exc()
                return pd.DataFrame(), pd.Timestamp.now().tz_localize(None)
        
        # Determine date range for historical data based on transactions or a default period
        oldest_tx_date = transactions_df_raw['date'].min() if not transactions_df_raw.empty else datetime.now() - timedelta(days=180)
        # Fetch data from a bit before the oldest transaction to today
        # Ensure dates are in ISO format strings for the API call
        api_start_date = (oldest_tx_date - timedelta(days=2)).isoformat()
        api_end_date = datetime.now().isoformat()

        historical_prices, earliest_data_date = load_and_preprocess_historical_data_uta_api(
            required_assets,
            start_date_str=api_start_date,
            end_date_str=api_end_date,
            interval="1h" # Default interval for now
        )

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
        render_transactions_manager(st.session_state.username, price_data_global, assets)
    
    # Подключение страниц из app_pages.py
    elif st.session_state.active_page == "Анализ портфеля":
        st.header("Анализ эффективности портфеля")
        st.markdown("Здесь вы можете проанализировать, как бы изменилась стоимость вашего портфеля, \\n        если бы вы следовали различным инвестиционным стратегиям.")

        # --- Session state initialization for this page ---
        if 'analysis_portfolio_id' not in st.session_state: st.session_state.analysis_portfolio_id = None
        if 'analysis_task_id' not in st.session_state: st.session_state.analysis_task_id = None
        if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
        if 'analysis_status_message' not in st.session_state: st.session_state.analysis_status_message = ""
        if 'analysis_error' not in st.session_state: st.session_state.analysis_error = None
        if 'last_polled_task_id' not in st.session_state: st.session_state.last_polled_task_id = None

        # --- Get current user's portfolio ID --- 
        if st.session_state.analysis_portfolio_id is None and "access_token" in st.session_state:
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            try:
                response = requests.get(f"{BACKEND_API_URL}/portfolios/", headers=headers, params={"limit": 1})
                response.raise_for_status()
                portfolios = response.json()
                if portfolios and len(portfolios) > 0:
                    st.session_state.analysis_portfolio_id = portfolios[0].get("id")
                else:
                    st.session_state.analysis_status_message = "Не найден портфель для анализа. Создайте портфель."
            except requests.exceptions.RequestException as e:
                st.session_state.analysis_error = f"Не удалось получить ID портфеля: {e}"
        
        if st.session_state.analysis_portfolio_id:
            st.info(f"Анализ будет выполнен для вашего портфеля ID: {st.session_state.analysis_portfolio_id}")
        elif not "access_token" in st.session_state:
            st.warning("Пользователь не аутентифицирован. Невозможно выбрать портфель для анализа.")
        else:
            st.warning(st.session_state.analysis_status_message or "Определение портфеля для анализа...")

        # --- Настройки Анализа --- 
        st.subheader("Параметры анализа")
        today_date = datetime.now().date()
        default_start_date = today_date - timedelta(days=180)

        col1, col2 = st.columns(2)
        with col1:
             start_date_analysis = st.date_input("Начальная дата анализа", value=default_start_date, max_value=today_date - timedelta(days=1), key="analysis_start_date")
             commission_input = st.number_input("Комиссия за ребалансировку (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.01, format="%.3f", key="analysis_commission")
        with col2:
             end_date_analysis = st.date_input("Конечная дата анализа", value=today_date, min_value=start_date_analysis + timedelta(days=1) if start_date_analysis else None, key="analysis_end_date")
             initial_capital_analysis = st.number_input("Начальный капитал (USD)", min_value=1.0, value=10000.0, step=100.0, format="%.2f", key="analysis_initial_capital")
        # Removed bank_apr, rebalance_interval, drl_rebalance_interval for simplicity, can be added to analysis_parameters if needed

        # --- Кнопка Запуска --- 
        if st.button("🚀 Запустить анализ портфеля (через API)", use_container_width=True, disabled=(st.session_state.analysis_portfolio_id is None)):
            st.session_state.analysis_task_id = None 
            st.session_state.analysis_results = None
            st.session_state.analysis_error = None
            st.session_state.last_polled_task_id = None
            st.session_state.analysis_status_message = "Отправка запроса на анализ..."

            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            analysis_params_for_backend = {
                "start_date": start_date_analysis.isoformat() if start_date_analysis else None,
                "end_date": end_date_analysis.isoformat() if end_date_analysis else None,
                "initial_capital": initial_capital_analysis,
                "commission_rate": commission_input / 100.0,
                # Add other params like rebalance_interval if your backend task uses them
            }
            payload = {
                "portfolio_id": st.session_state.analysis_portfolio_id,
                "analysis_parameters": analysis_params_for_backend
            }
            try:
                response = requests.post(f"{BACKEND_API_URL}/portfolios/analyze", json=payload, headers=headers)
                response.raise_for_status()
                task_info = response.json()
                st.session_state.analysis_task_id = task_info.get("task_id")
                st.session_state.last_polled_task_id = st.session_state.analysis_task_id # Initialize for polling
                st.session_state.analysis_status_message = f"Анализ запущен. ID Задачи: {st.session_state.analysis_task_id}. Обновление статуса..."
                st.success(st.session_state.analysis_status_message)
                st.experimental_rerun() # Start polling immediately
            except requests.exceptions.HTTPError as http_err:
                st.session_state.analysis_error = f"HTTP ошибка: {http_err} - {response.text}"
            except requests.exceptions.RequestException as req_err:
                st.session_state.analysis_error = f"Ошибка соединения: {req_err}"
            except Exception as e:
                st.session_state.analysis_error = f"Неожиданная ошибка: {e}"
        
        # Polling for task status
        if st.session_state.analysis_task_id and st.session_state.analysis_task_id == st.session_state.last_polled_task_id and st.session_state.analysis_results is None and st.session_state.analysis_error is None:
            time.sleep(3) # Wait before polling
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
            try:
                status_response = requests.get(f"{BACKEND_API_URL}/utils/tasks/{st.session_state.analysis_task_id}", headers=headers)
                status_response.raise_for_status()
                task_status_data = status_response.json()
                status = task_status_data.get("status")
                result = task_status_data.get("result") 
                meta_info = task_status_data.get("meta", {})

                if status == "SUCCESS":
                    st.session_state.analysis_status_message = "Анализ успешно завершен!"
                    st.session_state.analysis_results = result 
                    st.session_state.last_polled_task_id = None # Stop polling
                elif status == "FAILURE" or status == "REVOKED":
                    st.session_state.analysis_error = f"Ошибка выполнения анализа (Статус: {status}). Результат: {result or meta_info.get('exc_message', 'Нет деталей')}"
                    st.session_state.last_polled_task_id = None # Stop polling
                elif status == "PROGRESS":
                    current_step = meta_info.get('current', '')
                    total_steps = meta_info.get('total', '')
                    step_status = meta_info.get('status', 'Выполняется...')
                    progress_val = (current_step / total_steps) if isinstance(current_step, int) and isinstance(total_steps, int) and total_steps > 0 else 0
                    st.session_state.analysis_status_message = f"Прогресс: {step_status} ({current_step}/{total_steps})"
                    st.experimental_rerun() 
                else: # PENDING or other states
                    st.session_state.analysis_status_message = f"Статус задачи: {status}. {meta_info.get('status', 'Ожидание...')}"
                    st.experimental_rerun() 
            except requests.exceptions.RequestException as req_err:
                st.warning(f"Ошибка соединения при проверке статуса: {req_err}. Повторная попытка через несколько секунд...")
                st.experimental_rerun()
            except Exception as e:
                st.warning(f"Ошибка при проверке статуса задачи: {e}. Повторная попытка...")
                st.experimental_rerun()
        
        # Display status or results
        if st.session_state.analysis_error:
            st.error(st.session_state.analysis_error)
        elif st.session_state.analysis_results:
            st.subheader("Результаты Анализа Портфеля (из API)")
            results_package = st.session_state.analysis_results
            metrics_data = results_package.get("metrics")
            if metrics_data:
                st.markdown("**Ключевые метрики:**")
                # Assuming metrics_data is a dict like: {"period": "...", "final_value_buy_hold": ...}
                for key, value in metrics_data.items():
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
            else:
                st.info("Метрики не найдены в результатах.")
            
            st.markdown("--- ")
            st.markdown("**Полные данные результата:**")
            st.json(results_package) 
        elif st.session_state.analysis_task_id:
             if st.session_state.analysis_status_message:
                st.info(st.session_state.analysis_status_message)
             # Add a manual refresh button if stuck in pending for too long
             if st.button("🔄 Обновить статус задачи вручную"):
                 st.session_state.last_polled_task_id = st.session_state.analysis_task_id # Re-enable polling
                 st.experimental_rerun()
        else:
            st.info("Заполните параметры и запустите анализ, чтобы увидеть результаты.")

        # ----- Chatbot Section (Temporarily Simplified/Disabled) -----
        st.divider()
        st.subheader("Задать вопрос по результатам анализа (AI Агент)")
        st.info("Чат-бот для анализа результатов будет доработан после стабилизации получения результатов от API.")
        # ... (rest of the simplified chatbot UI) ...

    # --- End Section: Portfolio Analysis ---

    # Страница управления активами и транзакциями
    elif st.session_state.active_page == "Управление активами":
        render_transactions_manager(st.session_state.username, price_data_global, assets)
    
    # Подключение страниц из app_pages.py
    elif st.session_state.active_page == "Dashboard":
        # Ensure all necessary data is passed to render_dashboard
        # price_data_global, model_returns_global, model_actions_global are from load_app_data()
        # assets is derived from price_data_global.columns or st.session_state.assets
        auth_headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
        available_assets_list = st.session_state.get("all_available_tickers", []) # from load_app_data
        render_dashboard(st.session_state.username, BACKEND_API_URL, auth_headers, available_assets_list)
    
    elif st.session_state.active_page == "Portfolio Optimization":
        render_portfolio_optimization(st.session_state.username, price_data_global, assets)
    
    elif st.session_state.active_page == "Model Training":
        render_model_training(st.session_state.username, price_data_global, assets)
    
    elif st.session_state.active_page == "Model Comparison":
        render_model_comparison(st.session_state.username, model_returns_global, model_actions_global, price_data_global)
    
    elif st.session_state.active_page == "Backtest Results":
        render_backtest(st.session_state.username, model_returns_global, price_data_global)
    
    elif st.session_state.active_page == "About":
        render_about()

    # <<< Add block for the new Recommendations page >>>
    elif st.session_state.active_page == "Рекомендации":
        st.header("Рекомендации по ребалансировке портфеля")
        st.markdown("Получите рекомендации по оптимальному распределению активов для вашего портфеля.")

        # --- Session state initialization for this page ---
        if 'reco_portfolio_id' not in st.session_state: st.session_state.reco_portfolio_id = None
        if 'reco_task_id' not in st.session_state: st.session_state.reco_task_id = None
        if 'reco_results' not in st.session_state: st.session_state.reco_results = None
        if 'reco_status_message' not in st.session_state: st.session_state.reco_status_message = ""
        if 'reco_error' not in st.session_state: st.session_state.reco_error = None
        if 'last_polled_reco_task_id' not in st.session_state: st.session_state.last_polled_reco_task_id = None

        # --- Get current user's portfolio ID (similar to Analysis page) --- 
        if st.session_state.reco_portfolio_id is None and "access_token" in st.session_state:
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            try:
                response = requests.get(f"{BACKEND_API_URL}/portfolios/", headers=headers, params={"limit": 1})
                response.raise_for_status()
                portfolios = response.json()
                if portfolios and len(portfolios) > 0:
                    st.session_state.reco_portfolio_id = portfolios[0].get("id")
                    st.session_state.reco_status_message = f"Рекомендации будут сгенерированы для вашего портфеля ID: {st.session_state.reco_portfolio_id}"
                else:
                    st.session_state.reco_status_message = "Не найден портфель для генерации рекомендаций. Сначала создайте портфель."
                    st.session_state.reco_portfolio_id = "ERROR_NO_PORTFOLIO" # Special value
            except requests.exceptions.RequestException as e:
                st.session_state.reco_error = f"Не удалось получить ID портфеля: {e}"
        
        if st.session_state.reco_portfolio_id and st.session_state.reco_portfolio_id != "ERROR_NO_PORTFOLIO":
            st.info(st.session_state.reco_status_message)
        elif st.session_state.reco_portfolio_id == "ERROR_NO_PORTFOLIO":
            st.warning(st.session_state.reco_status_message)
        elif not "access_token" in st.session_state:
            st.warning("Пользователь не аутентифицирован. Невозможно запросить рекомендации.")
        else:
            st.warning(st.session_state.reco_error or "Определение портфеля для рекомендаций...")

        # --- Параметры Рекомендации (упрощенные) ---
        st.subheader("Параметры для генерации рекомендаций")
        
        custom_params_json = st.text_area(
            "Дополнительные параметры для бэкенда (JSON, опционально)",
            value='''{
  "risk_profile": "moderate",
  "target_return_annual_pct": 15.0,
  "drl_model_name": "PPO" 
}''',
            height=120,
            help="Эти параметры будут переданы в Celery задачу как `recommendation_parameters`."
        )

        # --- Кнопка Запуска ---
        run_button_disabled = not (st.session_state.reco_portfolio_id and st.session_state.reco_portfolio_id != "ERROR_NO_PORTFOLIO")
        
        if st.button("💡 Получить рекомендации по ребалансировке", use_container_width=True, disabled=run_button_disabled):
            st.session_state.reco_task_id = None 
            st.session_state.reco_results = None
            st.session_state.reco_error = None
            st.session_state.last_polled_reco_task_id = None
            st.session_state.reco_status_message = "Отправка запроса на генерацию рекомендаций..."

            parsed_custom_params = {}
            if custom_params_json.strip(): # Check if not empty or just whitespace
                try:
                    parsed_custom_params = json.loads(custom_params_json)
                except json.JSONDecodeError as e:
                    st.session_state.reco_error = f"Ошибка в формате JSON для дополнительных параметров: {e}"
                    # st.experimental_rerun() # Avoid immediate rerun on parse error, let user fix
            
            if not st.session_state.reco_error: # Proceed if JSON is valid or empty
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                payload = {
                    # portfolio_id will be derived by backend from current_user for /recommendations/rebalance
                    # "portfolio_id": st.session_state.reco_portfolio_id, 
                    "recommendation_parameters": parsed_custom_params 
                }
                try:
                    response = requests.post(f"{BACKEND_API_URL}/recommendations/rebalance", json=payload, headers=headers)
                    response.raise_for_status()
                    task_info = response.json()
                    st.session_state.reco_task_id = task_info.get("task_id")
                    st.session_state.last_polled_reco_task_id = st.session_state.reco_task_id
                    st.session_state.reco_status_message = f"Запрос на рекомендации отправлен. ID Задачи: {st.session_state.reco_task_id}. Обновление статуса..."
                    st.success(st.session_state.reco_status_message)
                    st.experimental_rerun() 
                except requests.exceptions.HTTPError as http_err:
                    error_detail = http_err.response.json().get("detail") if http_err.response else str(http_err)
                    st.session_state.reco_error = f"HTTP ошибка: {http_err.response.status_code} - {error_detail}"
                except requests.exceptions.RequestException as req_err:
                    st.session_state.reco_error = f"Ошибка соединения: {req_err}"
                except Exception as e:
                    st.session_state.reco_error = f"Неожиданная ошибка: {e}"
        
        # Polling for task status (similar to analysis page)
        if st.session_state.reco_task_id and st.session_state.reco_task_id == st.session_state.last_polled_reco_task_id and st.session_state.reco_results is None and st.session_state.reco_error is None:
            # Added a small visual cue for polling
            with st.spinner(f"Ожидание результата задачи {st.session_state.reco_task_id}..."):
                time.sleep(3) 
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
                try:
                    status_response = requests.get(f"{BACKEND_API_URL}/utils/tasks/{st.session_state.reco_task_id}", headers=headers)
                    status_response.raise_for_status()
                    task_status_data = status_response.json()
                    status = task_status_data.get("status")
                    result = task_status_data.get("result") 
                    meta_info = task_status_data.get("meta", {})

                    if status == "SUCCESS":
                        st.session_state.reco_status_message = "Рекомендации успешно сгенерированы!"
                        st.session_state.reco_results = result 
                        st.session_state.last_polled_reco_task_id = None 
                        st.experimental_rerun() # Rerun to display results and clear spinner
                    elif status == "FAILURE" or status == "REVOKED":
                        err_msg = meta_info.get('exc_message', 'Нет деталей')
                        if isinstance(result, dict) and 'error' in result: err_msg = result['error']
                        elif isinstance(result, str) : err_msg = result
                        st.session_state.reco_error = f"Ошибка генерации рекомендаций (Статус: {status}). Детали: {err_msg}"
                        st.session_state.last_polled_reco_task_id = None
                        st.experimental_rerun()
                    elif status == "PROGRESS":
                        current_step = meta_info.get('current', '')
                        total_steps = meta_info.get('total', '')
                        step_status = meta_info.get('status', 'Выполняется...')
                        st.session_state.reco_status_message = f"Прогресс: {step_status} ({current_step}/{total_steps})"
                        st.experimental_rerun() 
                    else: 
                        st.session_state.reco_status_message = f"Статус задачи: {status}. {meta_info.get('status', 'Ожидание...')}"
                        st.experimental_rerun() 
                except requests.exceptions.RequestException as req_err:
                    st.warning(f"Ошибка соединения при проверке статуса рекомендаций: {req_err}. Повторная попытка...")
                    st.experimental_rerun()
                except Exception as e:
                    st.warning(f"Ошибка при проверке статуса задачи рекомендаций: {e}. Повторная попытка...")
                    st.experimental_rerun()
        
        # Display status or results
        if st.session_state.reco_error:
            st.error(st.session_state.reco_error)
        elif st.session_state.reco_results:
            st.subheader("Полученные рекомендации")
            st.json(st.session_state.reco_results) 
            
            # Example of more structured display (can be customized based on actual backend response)
            if isinstance(st.session_state.reco_results, dict):
                if "target_allocation_pct" in st.session_state.reco_results:
                    st.markdown("#### Целевое распределение активов:")
                    alloc_data = st.session_state.reco_results["target_allocation_pct"]
                    if isinstance(alloc_data, dict):
                        alloc_df = pd.DataFrame(list(alloc_data.items()), columns=['Актив', 'Доля (%)'])
                        # alloc_df["Доля (%)"] = alloc_df["Доля (%)"] * 100 # Assuming backend sends as fraction, convert to %
                        st.dataframe(alloc_df.style.format({"Доля (%)": "{:.2f}%"}))
                        
                        fig_pie = px.pie(alloc_df, values="Доля (%)", names="Актив", title="Рекомендуемое распределение", hole=0.3)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.markdown("Данные по распределению активов имеют неверный формат.")
                
                if "summary" in st.session_state.reco_results:
                    st.markdown("#### Комментарий от системы:")
                    st.info(st.session_state.reco_results["summary"])
                
                if "next_rebalance_date" in st.session_state.reco_results:
                    st.markdown(f"**Рекомендуемая дата следующей ребалансировки:** {st.session_state.reco_results['next_rebalance_date']}")

        elif st.session_state.reco_task_id:
             if st.session_state.reco_status_message:
                st.info(st.session_state.reco_status_message)
             if st.button("🔄 Обновить статус задачи рекомендаций вручную"):
                 st.session_state.last_polled_reco_task_id = st.session_state.reco_task_id
                 st.experimental_rerun()
        elif not run_button_disabled:
            st.info("Настройте параметры (если необходимо) и нажмите кнопку, чтобы получить рекомендации.")
        

    # <<< Add block for the new Data & Analysis page >>>
    elif st.session_state.active_page == "Данные и Анализ":
        st.header("Управление данными и Анализ рынка")
        # Keep the existing "Обновление данных активов" and "Анализ рынка" (plots) sections as is for now.
        # We will focus on "Анализ новостей по активу" and "Чат по новостям".

        # --- Existing code for data update and market plots ---
        # (Assuming this part remains unchanged for this refactoring task)
        st.subheader("Обновление данных активов")
        if 'last_update_status' not in st.session_state: st.session_state.last_update_status = None
        if 'last_update_time' not in st.session_state: st.session_state.last_update_time = None
        if 'update_counter' not in st.session_state: st.session_state.update_counter = 0
        # ... (rest of the data update UI and logic - assumed to be kept) ...
        # Example: if st.button("🔄 Обновить рыночные данные" ...): ...
        # st.markdown("--- ")
        # st.subheader("Анализ рынка")
        # with st.spinner("Загрузка агрегированных данных..."):
        #      combined_df = load_combined_data_cached(st.session_state.update_counter)
        # ... (rest of the market analysis plots: normalized, correlation, single asset - assumed to be kept) ...
        # --- End of existing code to keep ---

        st.markdown("---") # Separator before News Analysis section
        st.subheader(f"Анализ новостей по активу (через API)")

        # --- Initialize session state for API-based news analysis --- 
        if 'news_api_asset_ticker' not in st.session_state: st.session_state.news_api_asset_ticker = None
        if 'news_api_task_id' not in st.session_state: st.session_state.news_api_task_id = None
        if 'news_api_results' not in st.session_state: st.session_state.news_api_results = None # This will store results from /news/asset/{ticker} or task result
        if 'news_api_status_message' not in st.session_state: st.session_state.news_api_status_message = ""
        if 'news_api_error' not in st.session_state: st.session_state.news_api_error = None
        if 'last_polled_news_api_task_id' not in st.session_state: st.session_state.last_polled_news_api_task_id = None
        if 'news_chat_history_api' not in st.session_state: st.session_state.news_chat_history_api = []
        if 'news_chat_task_id' not in st.session_state: st.session_state.news_chat_task_id = None
        if 'news_chat_error' not in st.session_state: st.session_state.news_chat_error = None

        # --- Asset Selection --- 
        # Use all_available_tickers from session state, loaded by load_app_data()
        available_tickers_for_news = st.session_state.get("all_available_tickers", [])
        if not available_tickers_for_news:
            st.warning("Список доступных активов пуст. Загрузите данные или проверьте API.")
            # You might want to add a button to trigger load_app_data() again or guide the user.
        
        col1_news_opts, col2_news_opts = st.columns([2, 3])
        with col1_news_opts:
            selected_asset_ticker_news_api = st.selectbox(
                "Выберите актив для анализа новостей:", 
                options=available_tickers_for_news,
                key="news_api_asset_select",
                index=available_tickers_for_news.index(st.session_state.news_api_asset_ticker) if st.session_state.news_api_asset_ticker and st.session_state.news_api_asset_ticker in available_tickers_for_news else 0,
                on_change=lambda: st.session_state.update({
                    'news_api_task_id': None, 
                    'news_api_results': None, 
                    'news_api_error': None,
                    'last_polled_news_api_task_id': None,
                    'news_chat_history_api': [],
                    'news_chat_task_id': None,
                    'news_chat_error': None
                })
            )
            if selected_asset_ticker_news_api:
                 st.session_state.news_api_asset_ticker = selected_asset_ticker_news_api
        
        with col2_news_opts:
            today_date_news = datetime.now().date()
            news_start_date_api = st.date_input("Начальная дата новостей", value=today_date_news - timedelta(days=7), max_value=today_date_news, key="news_api_start_date")
            news_end_date_api = st.date_input("Конечная дата новостей", value=today_date_news, min_value=news_start_date_api, max_value=today_date_news, key="news_api_end_date")
            num_articles_api = st.number_input("Макс. кол-во новостей для анализа", min_value=1, max_value=100, value=20, step=5, key="news_api_num_articles")

        # --- Buttons for Fetching Last Analysis & Triggering New Analysis ---
        col1_buttons, col2_buttons = st.columns(2)
        with col1_buttons:
            if st.button("🔍 Загрузить последний анализ (из БД)", key="fetch_last_news_analysis_button", disabled=not selected_asset_ticker_news_api):
                st.session_state.news_api_task_id = None # Clear any pending task
                st.session_state.news_api_results = None
                st.session_state.news_api_error = None
                st.session_state.news_chat_history_api = []
                st.session_state.news_api_status_message = f"Запрос последнего анализа для {selected_asset_ticker_news_api}..."
                with st.spinner(st.session_state.news_api_status_message):
                    headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
                    try:
                        response = requests.get(f"{BACKEND_API_URL}/news/asset/{selected_asset_ticker_news_api}", headers=headers)
                        response.raise_for_status()
                        st.session_state.news_api_results = response.json()
                        st.session_state.news_api_status_message = "Последний анализ успешно загружен."
                        st.success(st.session_state.news_api_status_message)
                        # Populate chat with a summary from these results if desired
                    except requests.exceptions.HTTPError as http_err:
                        if http_err.response.status_code == 404:
                            st.session_state.news_api_status_message = f"Сохраненный анализ для {selected_asset_ticker_news_api} не найден. Запустите новый анализ."
                            st.info(st.session_state.news_api_status_message)
                        else:
                            error_detail = http_err.response.json().get("detail") if http_err.response else str(http_err)
                            st.session_state.news_api_error = f"HTTP ошибка при загрузке анализа: {http_err.response.status_code} - {error_detail}"
                    except requests.exceptions.RequestException as req_err:
                        st.session_state.news_api_error = f"Ошибка соединения при загрузке анализа: {req_err}"
                    except Exception as e:
                        st.session_state.news_api_error = f"Неожиданная ошибка: {e}"
        
        with col2_buttons:
            if st.button("🚀 Запустить новый анализ новостей (через API)", key="run_new_news_analysis_button", disabled=not selected_asset_ticker_news_api):
                st.session_state.news_api_task_id = None 
                st.session_state.news_api_results = None
                st.session_state.news_api_error = None
                st.session_state.last_polled_news_api_task_id = None
                st.session_state.news_chat_history_api = []
                st.session_state.news_api_status_message = f"Отправка запроса на анализ новостей для {selected_asset_ticker_news_api}..."

                analysis_params_for_backend = {
                    "start_date": news_start_date_api.isoformat(),
                    "end_date": news_end_date_api.isoformat(),
                    "max_articles": num_articles_api,
                    # Add other relevant parameters for the backend task, e.g., news_sources, language
                }
                payload = {
                    "asset_ticker": selected_asset_ticker_news_api,
                    "analysis_parameters": analysis_params_for_backend
                }
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
                try:
                    response = requests.post(f"{BACKEND_API_URL}/news/analyze", json=payload, headers=headers)
                    response.raise_for_status()
                    task_info = response.json()
                    st.session_state.news_api_task_id = task_info.get("task_id")
                    st.session_state.last_polled_news_api_task_id = st.session_state.news_api_task_id
                    st.session_state.news_api_status_message = f"Анализ новостей запущен. ID Задачи: {st.session_state.news_api_task_id}. Обновление статуса..."
                    st.success(st.session_state.news_api_status_message)
                    st.experimental_rerun()
                except requests.exceptions.HTTPError as http_err:
                    error_detail = http_err.response.json().get("detail") if http_err.response else str(http_err)
                    st.session_state.news_api_error = f"HTTP ошибка: {http_err.response.status_code} - {error_detail}"
                except requests.exceptions.RequestException as req_err:
                    st.session_state.news_api_error = f"Ошибка соединения: {req_err}"
                except Exception as e:
                    st.session_state.news_api_error = f"Неожиданная ошибка: {e}"

        # Polling for task status (for news analysis task)
        if st.session_state.news_api_task_id and st.session_state.news_api_task_id == st.session_state.last_polled_news_api_task_id and st.session_state.news_api_results is None and st.session_state.news_api_error is None:
            with st.spinner(f"Ожидание результата анализа новостей (Задача: {st.session_state.news_api_task_id})..."):
                time.sleep(3) 
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
                try:
                    status_response = requests.get(f"{BACKEND_API_URL}/utils/tasks/{st.session_state.news_api_task_id}", headers=headers)
                    status_response.raise_for_status()
                    task_status_data = status_response.json()
                    status = task_status_data.get("status")
                    result = task_status_data.get("result") 
                    meta_info = task_status_data.get("meta", {})

                    if status == "SUCCESS":
                        st.session_state.news_api_status_message = "Анализ новостей успешно завершен!"
                        st.session_state.news_api_results = result 
                        st.session_state.last_polled_news_api_task_id = None 
                        st.experimental_rerun()
                    elif status == "FAILURE" or status == "REVOKED":
                        err_msg = meta_info.get('exc_message', 'Нет деталей')
                        if isinstance(result, dict) and 'error' in result: err_msg = result['error']
                        elif isinstance(result, str) : err_msg = result
                        st.session_state.news_api_error = f"Ошибка выполнения анализа новостей (Статус: {status}). Детали: {err_msg}"
                        st.session_state.last_polled_news_api_task_id = None
                        st.experimental_rerun()
                    elif status == "PROGRESS":
                        current_step = meta_info.get('current', '')
                        total_steps = meta_info.get('total', '')
                        step_status = meta_info.get('status', 'Выполняется...')
                        st.session_state.news_api_status_message = f"Прогресс: {step_status} ({current_step}/{total_steps})"
                        st.experimental_rerun() 
                    else: 
                        st.session_state.news_api_status_message = f"Статус задачи: {status}. {meta_info.get('status', 'Ожидание...')}"
                        st.experimental_rerun() 
                except requests.exceptions.RequestException as req_err:
                    st.warning(f"Ошибка соединения при проверке статуса анализа новостей: {req_err}. Повторная попытка...")
                    st.experimental_rerun()
                except Exception as e:
                    st.warning(f"Ошибка при проверке статуса задачи анализа новостей: {e}. Повторная попытка...")
                    st.experimental_rerun()
        
        # --- Display News Analysis Results --- 
        if st.session_state.news_api_error:
            st.error(st.session_state.news_api_error)
        elif st.session_state.news_api_results:
            st.markdown("---")
            st.subheader(f"Результаты анализа новостей для {st.session_state.news_api_results.get('asset_ticker', selected_asset_ticker_news_api)}")
            results_data = st.session_state.news_api_results
            # Expected structure from backend (either from /news/asset/{ticker} or Celery task result for NewsAnalysisResultPublic)
            # { "asset_ticker": "...", "analysis_timestamp": "...", "news_count": ..., 
            #   "overall_sentiment_label": "...", "overall_sentiment_score": ..., 
            #   "key_themes": ["theme1", "theme2"], "full_summary": "...", "analysis_parameters": {...} }
            
            st.markdown(f"**Обновлено:** {results_data.get('analysis_timestamp', 'N/A')}")
            st.markdown(f"**Количество проанализированных новостей:** {results_data.get('news_count', 'N/A')}")
            
            col_sent_label, col_sent_score = st.columns(2)
            with col_sent_label:
                st.metric("Общая тональность", results_data.get('overall_sentiment_label', 'N/A'))
            with col_sent_score:
                score = results_data.get('overall_sentiment_score')
                st.metric("Оценка тональности", f"{score:.2f}" if isinstance(score, float) else 'N/A',
                          help="-1 (Негативная) до +1 (Позитивная)")
            
            if results_data.get('key_themes'):
                st.markdown("**Ключевые темы:**")
                # st.write(", ".join(results_data['key_themes']))
                for theme in results_data['key_themes']:
                    st.markdown(f"- {theme}")
            
            if results_data.get('full_summary'):
                st.markdown("**AI Сводка:**")
                st.info(results_data['full_summary'])
            
            # Optionally, display raw parameters or individual news items if backend provides them
            # st.expander("Детали анализа (сырые данные от API)").json(results_data)

            # --- Auto-populate chat with a summary from these results ---
            if not st.session_state.news_chat_history_api: # Only if chat is empty
                summary_for_chat = results_data.get('full_summary', "Результаты анализа новостей загружены.")
                if results_data.get('overall_sentiment_label'):
                    summary_for_chat = f"Анализ для {results_data.get('asset_ticker')}: Тональность - {results_data.get('overall_sentiment_label')} (Оценка: {results_data.get('overall_sentiment_score', 0):.2f}). " + summary_for_chat
                st.session_state.news_chat_history_api.append({"role": "assistant", "content": summary_for_chat})

        elif st.session_state.news_api_task_id:
             if st.session_state.news_api_status_message:
                st.info(st.session_state.news_api_status_message)
             if st.button("🔄 Обновить статус задачи анализа новостей вручную"):
                 st.session_state.last_polled_news_api_task_id = st.session_state.news_api_task_id
                 st.experimental_rerun()
        elif selected_asset_ticker_news_api: # No task, no results, but asset selected
            st.info(f"Выберите действие для актива {selected_asset_ticker_news_api}: загрузить последний анализ или запустить новый.")
        
        # --- News Chat Interface (API based) ---
        st.markdown("---")
        st.subheader(f"AI Чат по новостям для {selected_asset_ticker_news_api if selected_asset_ticker_news_api else 'актива'}")

        # Display chat messages
        chat_container = st.container(height=300)
        with chat_container:
            for message in st.session_state.news_chat_history_api:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Задайте вопрос по новостному анализу...", key="news_chat_api_input", disabled=not selected_asset_ticker_news_api):
            st.session_state.news_chat_history_api.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            st.session_state.news_chat_task_id = None # Reset previous task ID for chat
            st.session_state.news_chat_error = None

            with st.spinner("AI обрабатывает ваш запрос... (может занять время)"): 
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
                # Construct payload for news chat API
                # Context could be the news_api_results or just the ticker
                chat_payload = {
                    "asset_ticker": selected_asset_ticker_news_api,
                    "user_query": prompt,
                    "chat_history": st.session_state.news_chat_history_api[:-1], # Send history excluding current user prompt
                    "analysis_context": st.session_state.news_api_results # Send current analysis results as context
                }
                try:
                    response = requests.post(f"{BACKEND_API_URL}/news/chat", json=chat_payload, headers=headers)
                    response.raise_for_status()
                    chat_task_info = response.json()
                    st.session_state.news_chat_task_id = chat_task_info.get("task_id")

                    # Start polling for chat task result
                    ai_response_content = None
                    polling_attempts = 0
                    max_polling_attempts = 20 # Approx 1 minute (20 * 3s)
                    
                    while polling_attempts < max_polling_attempts and ai_response_content is None:
                        time.sleep(3)
                        status_response = requests.get(f"{BACKEND_API_URL}/utils/tasks/{st.session_state.news_chat_task_id}", headers=headers)
                        status_response.raise_for_status()
                        task_status_data = status_response.json()
                        status = task_status_data.get("status")
                        result = task_status_data.get("result")

                        if status == "SUCCESS":
                            ai_response_content = result.get("ai_response", "Не удалось получить ответ от AI.") if isinstance(result, dict) else "Ответ AI в неизвестном формате."
                            break
                        elif status == "FAILURE" or status == "REVOKED":
                            st.session_state.news_chat_error = f"Ошибка задачи AI чата (Статус: {status}). {result}"
                            break 
                        # Add slight delay for PENDING/PROGRESS before next attempt
                        polling_attempts += 1
                    
                    if ai_response_content:
                        st.session_state.news_chat_history_api.append({"role": "assistant", "content": ai_response_content})
                    elif not st.session_state.news_chat_error:
                        st.session_state.news_chat_error = "AI не ответил вовремя. Попробуйте еще раз."
                    
                    if st.session_state.news_chat_error: # Display error if any
                         st.session_state.news_chat_history_api.append({"role": "assistant", "content": f"(Ошибка: {st.session_state.news_chat_error})"})
                    
                    st.experimental_rerun() # Rerun to display AI response or error

                except requests.exceptions.HTTPError as http_err:
                    error_detail = http_err.response.json().get("detail") if http_err.response else str(http_err)
                    st.session_state.news_chat_history_api.append({"role": "assistant", "content": f"(HTTP ошибка чата: {http_err.response.status_code} - {error_detail})"})
                    st.experimental_rerun()
                except requests.exceptions.RequestException as req_err:
                    st.session_state.news_chat_history_api.append({"role": "assistant", "content": f"(Ошибка соединения с чатом: {req_err})"}) 
                    st.experimental_rerun()
                except Exception as e:
                    st.session_state.news_chat_history_api.append({"role": "assistant", "content": f"(Неожиданная ошибка чата: {e})"}) 
                    st.experimental_rerun()
        
        # Old news analysis and chat code is now fully replaced by the API-driven version above.
        # Remove or comment out the old section:
        # # <<< START NEW: FinNLP Analysis Section >>>
        # ... (old code was here) ...
        # # <<< END NEW: FinNLP Analysis Section >>>

    # --- NEW PAGE: Исследование (Hypothetical Portfolio Simulation) ---
    elif st.session_state.active_page == "Исследование":
        st.header("Исследование: Моделирование гипотетического портфеля")
        st.markdown("Проверьте потенциальную доходность и риски различных портфельных стратегий.")

        # --- Session state initialization for this page ---
        if 'hypothetical_sim_task_id' not in st.session_state: st.session_state.hypothetical_sim_task_id = None
        if 'hypothetical_sim_results' not in st.session_state: st.session_state.hypothetical_sim_results = None
        if 'hypothetical_sim_status_message' not in st.session_state: st.session_state.hypothetical_sim_status_message = ""
        if 'hypothetical_sim_error' not in st.session_state: st.session_state.hypothetical_sim_error = None
        if 'last_polled_hypothetical_sim_task_id' not in st.session_state: st.session_state.last_polled_hypothetical_sim_task_id = None

        # --- Форма для ввода параметров ---
        with st.form("hypothetical_simulation_form"):
            st.subheader("Параметры моделирования")
            
            col1_hs, col2_hs = st.columns(2)
            with col1_hs:
                hs_initial_capital = st.number_input("Начальный капитал (USD)", min_value=1.0, value=10000.0, step=100.0, format="%.2f", key="hs_initial_capital")
                hs_start_date = st.date_input("Дата начала", value=datetime.now().date() - timedelta(days=365), key="hs_start_date")
                hs_rebalancing_frequency = st.selectbox(
                    "Частота ребалансировки",
                    options=["none", "monthly", "quarterly", "annually"],
                    index=0,
                    key="hs_rebalancing_frequency"
                )

            with col2_hs:
                # Получаем список доступных тикеров для примера в JSON
                available_tickers_for_sim = st.session_state.get("all_available_tickers", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
                example_assets_weights_json = json.dumps({ticker: round(1/len(available_tickers_for_sim), 2) for ticker in available_tickers_for_sim[:3]}, indent=2) if available_tickers_for_sim else '{ "BTCUSDT": 0.5, "ETHUSDT": 0.5 }'
                
                hs_assets_weights_json = st.text_area(
                    "Веса активов (JSON формат, сумма весов должна быть ~1.0)",
                    value=example_assets_weights_json,
                    height=150,
                    key="hs_assets_weights_json",
                    help='Пример: {"BTCUSDT": 0.6, "ETHUSDT": 0.4}. Убедитесь, что тикеры существуют.'
                )
                hs_end_date = st.date_input("Дата окончания", value=datetime.now().date(), key="hs_end_date")
                hs_commission_rate = st.number_input("Комиссия за транзакцию (доля, например, 0.001 для 0.1%)", min_value=0.0, max_value=0.1, value=0.001, step=0.0001, format="%.4f", key="hs_commission_rate")

            hs_submit_button = st.form_submit_button("🚀 Запустить моделирование")

        if hs_submit_button:
            st.session_state.hypothetical_sim_task_id = None
            st.session_state.hypothetical_sim_results = None
            st.session_state.hypothetical_sim_error = None
            st.session_state.last_polled_hypothetical_sim_task_id = None
            st.session_state.hypothetical_sim_status_message = "Отправка запроса на моделирование..."

            try:
                assets_weights = json.loads(hs_assets_weights_json)
                if not isinstance(assets_weights, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in assets_weights.items()):
                    raise json.JSONDecodeError("Веса активов должны быть словарем {тикер: вес}", hs_assets_weights_json, 0)
                
                # Конвертация Decimal для Pydantic, если необходимо, но FastAPI должен справиться с float
                # assets_weights_decimal = {k: Decimal(str(v)) for k, v in assets_weights.items()}

                payload = {
                    "initial_capital": float(hs_initial_capital),
                    "assets_weights": assets_weights, # FastAPI/Pydantic сконвертирует float в Decimal, если в схеме Decimal
                    "start_date": hs_start_date.isoformat(),
                    "end_date": hs_end_date.isoformat(),
                    "rebalancing_frequency": hs_rebalancing_frequency,
                    "commission_rate": float(hs_commission_rate)
                }

                if "access_token" not in st.session_state:
                    st.session_state.hypothetical_sim_error = "Ошибка: Пользователь не аутентифицирован."
                else:
                    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                    response = requests.post(f"{BACKEND_API_URL}/portfolios/simulate_hypothetical", json=payload, headers=headers)
                    response.raise_for_status()
                    task_info = response.json()
                    st.session_state.hypothetical_sim_task_id = task_info.get("task_id")
                    st.session_state.last_polled_hypothetical_sim_task_id = st.session_state.hypothetical_sim_task_id
                    st.session_state.hypothetical_sim_status_message = f"Моделирование запущено. ID Задачи: {st.session_state.hypothetical_sim_task_id}. Обновление статуса..."
                    st.success(st.session_state.hypothetical_sim_status_message)
                    st.experimental_rerun()

            except json.JSONDecodeError as e:
                st.session_state.hypothetical_sim_error = f"Ошибка в формате JSON для весов активов: {e}"
            except requests.exceptions.HTTPError as http_err:
                error_detail = "Не удалось получить детали ошибки."
                try:
                    error_detail = http_err.response.json().get("detail", str(http_err))
                except json.JSONDecodeError:
                    error_detail = http_err.response.text if http_err.response.text else str(http_err)
                st.session_state.hypothetical_sim_error = f"HTTP ошибка: {http_err.response.status_code} - {error_detail}"
            except requests.exceptions.RequestException as req_err:
                st.session_state.hypothetical_sim_error = f"Ошибка соединения: {req_err}"
            except Exception as e:
                st.session_state.hypothetical_sim_error = f"Неожиданная ошибка: {e}"
                traceback.print_exc()

        # Polling for task status
        if st.session_state.hypothetical_sim_task_id and \
           st.session_state.hypothetical_sim_task_id == st.session_state.last_polled_hypothetical_sim_task_id and \
           st.session_state.hypothetical_sim_results is None and \
           st.session_state.hypothetical_sim_error is None:
            
            with st.spinner(f"Ожидание результата моделирования (Задача: {st.session_state.hypothetical_sim_task_id})..."):
                time.sleep(3) # Wait before polling
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"} if "access_token" in st.session_state else {}
                try:
                    status_response = requests.get(f"{BACKEND_API_URL}/utils/tasks/{st.session_state.hypothetical_sim_task_id}", headers=headers)
                    status_response.raise_for_status()
                    task_status_data = status_response.json()
                    status = task_status_data.get("status")
                    result = task_status_data.get("result") 
                    meta_info = task_status_data.get("meta", {})

                    if status == "SUCCESS":
                        st.session_state.hypothetical_sim_status_message = "Моделирование успешно завершено!"
                        st.session_state.hypothetical_sim_results = result 
                        st.session_state.last_polled_hypothetical_sim_task_id = None # Stop polling
                        st.experimental_rerun()
                    elif status == "FAILURE" or status == "REVOKED":
                        err_msg = meta_info.get('exc_message', 'Детали ошибки отсутствуют.')
                        if isinstance(result, dict) and 'error' in result: err_msg = result['error']
                        elif isinstance(result, str) : err_msg = result
                        st.session_state.hypothetical_sim_error = f"Ошибка выполнения моделирования (Статус: {status}). Детали: {err_msg}"
                        st.session_state.last_polled_hypothetical_sim_task_id = None # Stop polling
                        st.experimental_rerun()
                    elif status == "PROGRESS":
                        current_step = meta_info.get('current', '')
                        total_steps = meta_info.get('total', '')
                        step_status = meta_info.get('status', 'Выполняется...')
                        st.session_state.hypothetical_sim_status_message = f"Прогресс: {step_status} ({current_step}/{total_steps})"
                        st.experimental_rerun() 
                    else: # PENDING or other states
                        st.session_state.hypothetical_sim_status_message = f"Статус задачи: {status}. {meta_info.get('status', 'Ожидание...')}"
                        st.experimental_rerun() 
                except requests.exceptions.RequestException as req_err:
                    st.warning(f"Ошибка соединения при проверке статуса моделирования: {req_err}. Повторная попытка...")
                    st.experimental_rerun()
                except Exception as e:
                    st.warning(f"Ошибка при проверке статуса задачи моделирования: {e}. Повторная попытка...")
                    traceback.print_exc()
                    st.experimental_rerun()
        
        # Display status or results
        if st.session_state.hypothetical_sim_error:
            st.error(st.session_state.hypothetical_sim_error)
        elif st.session_state.hypothetical_sim_results:
            st.subheader("Результаты моделирования гипотетического портфеля")
            results_package = st.session_state.hypothetical_sim_results
            
            metrics = results_package.get("metrics")
            if metrics:
                st.markdown("#### Ключевые метрики:")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Период", metrics.get("period", "N/A"))
                    st.metric("Начальный капитал", f"${metrics.get('initial_capital', 0):,.2f}")
                    st.metric("Конечная стоимость портфеля", f"${metrics.get('final_value_hypothetical', 0):,.2f}")
                    st.metric("CAGR (среднегодовая доходность)", f"{metrics.get('cagr_hypothetical', 0)*100:.2f}%")
                with col_m2:
                    st.metric("Коэффициент Шарпа", f"{metrics.get('sharpe_hypothetical', 0):.2f}")
                    st.metric("Волатильность", f"{metrics.get('volatility_hypothetical', 0)*100:.2f}%")
                    st.metric("Максимальная просадка", f"{metrics.get('max_drawdown_hypothetical', 0)*100:.2f}%")
                    st.metric("Частота ребалансировки", str(metrics.get("rebalancing_frequency", "N/A")).title())
                    st.metric("Комиссия", f"{metrics.get('commission_rate', 0)*100:.3f}%")

                # Можно добавить отображение simulation_parameters, если нужно
                with st.expander("Параметры моделирования (для справки)"):
                    st.json(results_package.get("simulation_parameters", {}))
                
                # Здесь можно будет добавить график, если бэкенд будет возвращать данные для него
                # if "plot_data" in results_package:
                #     st.subheader("График стоимости портфеля")
                #     # ... код для построения графика ...
            else:
                st.info("Метрики не найдены в результатах.")
            
            st.markdown("---")
            st.markdown("Полные данные результата (JSON):")
            st.json(results_package)

        elif st.session_state.hypothetical_sim_task_id: # Task is running or pending
            if st.session_state.hypothetical_sim_status_message:
                st.info(st.session_state.hypothetical_sim_status_message)
            if st.button("🔄 Обновить статус задачи моделирования вручную"):
                st.session_state.last_polled_hypothetical_sim_task_id = st.session_state.hypothetical_sim_task_id # Re-enable polling for this task
                st.experimental_rerun()
        elif not hs_submit_button and not st.session_state.hypothetical_sim_task_id : # Initial state, no button pressed yet, no task running for this page
             st.info("Заполните параметры выше и нажмите 'Запустить моделирование', чтобы увидеть результаты.")



    '''
    poetry run streamlit run auth_app.py
    '''