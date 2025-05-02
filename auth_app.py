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
import json # For holdings display
# --- ADDED: Import pipeline --- 
from transformers import pipeline
import torch # Ensure torch is imported if used for device placement

# Импорт модулей приложения
from portfolios_optimization.data_loader import (
    load_price_data,
    load_return_data,
    load_model_actions,
    update_all_asset_data, # New
    create_combined_data,  # New
    generate_normalized_plot, # New
    generate_correlation_heatmap, # New
    generate_single_asset_plot # New
)
from portfolios_optimization.portfolio_optimizer import optimize_markowitz_portfolio
from portfolios_optimization.portfolio_analysis import calculate_metrics, plot_efficient_frontier
from portfolios_optimization.visualization import plot_portfolio_performance, plot_asset_allocation
from portfolios_optimization.model_trainer import train_model, load_trained_model
from portfolios_optimization.authentication import (
    initialize_users_file, register_user, authenticate_user, get_user_info,
    update_user_portfolio, get_user_portfolios, get_user_portfolio, get_portfolio_with_quantities,
    get_user_transactions
)
# <<< Import the new hypothetical simulator function >>>
from portfolios_optimization.hypothetical_simulator import run_hypothetical_analysis

# --- Imports needed for Recommendations --- #
from portfolio_analyzer import ( # Assuming these are defined in portfolio_analyzer.py
    preprocess_asset_data, # Need this helper
    FeatureEngineer,       # Need the class
    # StockPortfolioEnv,     # Might not be needed directly if DRLAgent handles env creation/reset
    DRLAgent,              # Need the agent class for prediction method
    INDICATORS,            # Need the list of indicators used for training
    STABLECOIN_ASSET,      # Need the stablecoin identifier
    softmax_normalization, # Need the normalization function
    # <<< REMOVE A2C, PPO, etc. from here >>>
)
# <<< Add direct import for SB3 models >>>
from stable_baselines3 import A2C, PPO, SAC, DDPG
# --- End Imports for Recommendations --- #

# <<< NEW: Import the analysis function >>>
from portfolio_analyzer import run_portfolio_analysis

# --- NEW: Imports for FinRobot/Autogen Chatbot ---
import autogen
# from finrobot.agents.workflow import SingleAssistant # Old import
from portfolios_optimization.finrobot_core.workflow import SingleAssistant # New import
# --- End Imports for FinRobot/Autogen Chatbot ---

# Импорт страниц приложения
from app_pages import render_account_dashboard, render_transactions_manager

# --- HELPER FUNCTIONS (Moved to top after imports) ---


 # --- NEW: Function to fetch real news from processed CSV files >>>
def fetch_news_from_csv(asset_name, start_date=None, end_date=None, base_news_dir="notebooks/news_data", num_articles=15):
    """
    Читает обработанный CSV с новостями для актива, фильтрует по дате 
    и возвращает сводку последних `num_articles` новостей в указанном диапазоне.
    
    Args:
        asset_name (str): Имя актива/папки (e.g., 'btc', 'eth').
        start_date (datetime.date, optional): Начальная дата для фильтрации новостей.
        end_date (datetime.date, optional): Конечная дата для фильтрации новостей.
        base_news_dir (str): Базовая директория с папками новостей.
        num_articles (int): Максимальное количество последних новостей для возврата.
        
    Returns:
        dict: Словарь с ключами 'text', 'summaries', 'start_date', 'end_date', 'count' или None при ошибке.
    """
    # --- No change needed for mapping ticker to asset_name here, asset_name is now passed directly --- 
    
    combined_file = os.path.join(base_news_dir, asset_name, f"combined_{asset_name}_news.csv")

    if not os.path.exists(combined_file):
        st.error(f"Файл с новостями не найден: {combined_file}. Запустите скрипт process_news_data.py.")
        return None

    try:
        df = pd.read_csv(combined_file)
        
        # Ensure 'date_published' exists and convert to datetime
        if 'date_published' not in df.columns:
            st.error(f"Колонка 'date_published' отсутствует в файле: {combined_file}")
            return None
        # Convert to datetime objects, coercing errors
        df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
        df.dropna(subset=['date_published'], inplace=True)
        # Keep only date part for comparison with date input
        df['date_published_date'] = df['date_published'].dt.date 

        # Ensure 'summary' exists
        if 'summary' not in df.columns:
                st.error(f"Колонка 'summary' отсутствует в файле: {combined_file}")
                return None
        
        # --- ADDED: Filter by date range --- 
        min_date_in_file = df['date_published_date'].min()
        max_date_in_file = df['date_published_date'].max()
        
        # Use provided dates, default to file min/max if None
        filter_start_date = start_date if start_date else min_date_in_file
        filter_end_date = end_date if end_date else max_date_in_file

        # Apply the filter
        mask = (df['date_published_date'] >= filter_start_date) & (df['date_published_date'] <= filter_end_date)
        df_filtered = df[mask]
        # --- End Date Filtering --- 
        
        if df_filtered.empty:
            st.info(f"Не найдено новостей для {asset_name.upper()} в диапазоне {filter_start_date} - {filter_end_date}.")
            return None
            
        # Sort by actual datetime descending and get latest summaries from filtered data
        df_sorted = df_filtered.sort_values(by='date_published', ascending=False)
        # --- RE-ADDED .head(num_articles) --- 
        latest_summaries = df_sorted['summary'].head(num_articles).tolist()
        df_selected = df_sorted.head(num_articles) # Keep the selected dataframe part
        
        # Get the actual date range of the selected articles
        # Use df_selected to get dates only from the articles we are returning
        if not df_selected.empty:
             actual_min_date = df_selected['date_published_date'].min()
             actual_max_date = df_selected['date_published_date'].max()
        else: # Should not happen if latest_summaries is not empty, but safety check
             actual_min_date = filter_start_date
             actual_max_date = filter_end_date
             
        articles_count = len(latest_summaries)

        if not latest_summaries:
            # This case should be covered by df_filtered.empty check above, but kept for safety
            st.info(f"Не найдено сводок новостей для {asset_name.upper()} в диапазоне {filter_start_date} - {filter_end_date}.")
            return None
            
        # Combine summaries into a single text block
        full_text = "\n\n---\n\n".join([str(s) for s in latest_summaries if pd.notna(s)])
        
        return {
            "text": full_text,
            "summaries": latest_summaries, # Return the selected summaries
            "start_date": actual_min_date,
            "end_date": actual_max_date,
            "count": articles_count # Count reflects selected articles
        }

    except Exception as e:
        st.error(f"Ошибка при чтении или обработке файла новостей {combined_file}: {e}")
        traceback.print_exc()
        return None
    # <<< END NEWS FETCHING FUNCTION >>>


# --- Function to initialize FinRobot Agent ---
@st.cache_resource # Cache the agent resource itself
def initialize_finrobot_agent():
    """Initializes and returns a FinRobot agent configured with a local LLM server."""
    try:
        # --- MODIFIED: Configuration for local LLM ---
        local_model_name = "llama3" # Change if your model name in Ollama/LM Studio is different
        local_api_base = "http://localhost:11434/v1" # Change if your local server runs on a different address/port

        llm_config = {
            "config_list": [
                {
                    "model": local_model_name,
                    "base_url": local_api_base,
                    "api_key": "ollama", # Often required, but value doesn't matter for local non-auth servers
                }
            ],
            "timeout": 300, # Increase timeout for potentially slower local models
            "temperature": 0.2,
        }
        # --- END MODIFICATION ---

        assistant_agent = SingleAssistant(
            name="Portfolio_Analyst_Assistant",
            # --- MODIFIED System Message ---
            system_message="""Ты — ИИ-ассистент, эксперт по анализу инвестиционных портфелей. \
Твоя задача — отвечать на вопросы пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленных ниже данных анализа портфеля. \
Отвечай всегда на РУССКОМ языке. Будь точным, кратким и профессиональным.""",
            # --- END MODIFICATION ---
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        st.success(f"FinRobot agent initialized with local model: {local_model_name} at {local_api_base}") # Add success message
        return assistant_agent
    except FileNotFoundError: # Keep this for OAI_CONFIG_LIST just in case, though not used now
        st.error("Error: OAI_CONFIG_LIST file not found. Please ensure it exists in the project root.")
        return None
    except Exception as e:
        st.error(f"Error initializing FinRobot agent with local LLM: {e}")
        traceback.print_exc()
        st.error(f"Ensure your local LLM server (e.g., Ollama, LM Studio) is running and accessible at {local_api_base} and the model '{local_model_name}' is available.") # Add helpful error message
        return None

# --- Function to format analysis results for LLM ---
def format_portfolio_data_for_llm(analysis_results):
    """Formats the portfolio analysis results into a string for the LLM agent."""
    if not analysis_results or not isinstance(analysis_results, dict):
        return "Нет данных для анализа или неверный формат."

    summary_parts = []

    # Extract metrics (raw numeric data)
    metrics_df = analysis_results.get('metrics') # Use the raw metrics
    if metrics_df is not None and isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        summary_parts.append("**Основные метрики производительности по стратегиям:**")
        for strategy in metrics_df.index: # Iterate through strategies (index)
            summary_parts.append(f"\n*Стратегия: {strategy}*" )
            for metric, value in metrics_df.loc[strategy].items(): # Iterate through metrics for the strategy
                if isinstance(value, (int, float)):
                    # Use original metric names from calculation for formatting clues
                    if any(p in metric.lower() for p in ['cagr', 'return']):
                        formatted_value = f"{value:.2%}"
                    elif any(p in metric.lower() for p in ['volatility', 'drawdown']):
                         formatted_value = f"{value:.2%}"
                    elif any(r in metric.lower() for r in ['ratio']):
                         formatted_value = f"{value:.2f}"
                    else: # Default for Final Value, Net Profit etc.
                         formatted_value = f"{value:,.2f}"
                         if 'value' in metric.lower() or 'profit' in metric.lower():
                              formatted_value = f"${formatted_value}" # Add dollar sign
                else:
                     formatted_value = str(value)
                summary_parts.append(f"  - {metric}: {formatted_value}")
        summary_parts.append("\n") 
    else:
        summary_parts.append("Метрики производительности недоступны.")

    # Extract date range and final values from daily values DataFrame
    daily_values_df = analysis_results.get('portfolio_daily_values')
    if daily_values_df is not None and isinstance(daily_values_df, pd.DataFrame) and not daily_values_df.empty:
        start_date = daily_values_df.index.min().strftime('%Y-%m-%d')
        end_date = daily_values_df.index.max().strftime('%Y-%m-%d')
        summary_parts.append(f"**Период анализа:** {start_date} - {end_date}\n")

        summary_parts.append("**Финальная стоимость портфеля по стратегиям:**")
        final_values = daily_values_df.iloc[-1] # Get last row
        # Filter only strategy value columns (usually start with 'Value_')
        strategy_value_cols = [col for col in daily_values_df.columns if col.startswith('Value_')]
        for strategy_col in strategy_value_cols:
            # Try to map column name back to display name if possible (e.g., from metrics index)
            strategy_name = strategy_col.replace('Value_', '').replace('_', ' ') # Basic name cleanup
            if metrics_df is not None and not metrics_df.empty:
                 matching_names = [idx for idx in metrics_df.index if strategy_col.endswith(idx.replace(' ','_').replace('DRL ',''))]
                 if matching_names: strategy_name = matching_names[0]
            
            value = final_values.get(strategy_col, np.nan)
            if pd.notna(value):
                 summary_parts.append(f"  - {strategy_name}: ${value:,.2f}")
        summary_parts.append("\n")
    else:
        summary_parts.append("Данные о ежедневной стоимости портфеля недоступны.")

    # <<< REMOVED holdings formatting logic >>>

    return "\n".join(summary_parts)

# --- END HELPER FUNCTIONS ---

# Конфигурация страницы
st.set_page_config(
    page_title="Portfolio Optimization System",
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
            csv_base_path = 'D:\\__projects__\\diploma\\portfolios-optimization\\data' # Consider making this relative or configurable
            all_prices = {}
            data_found = False
            min_start_date = pd.Timestamp.max.tz_localize(None)
            for asset in assets_list:
                file_path = os.path.join(csv_base_path, f'{asset}_hourly_data.csv')
                try:
                    # --- Robust Date Parsing --- 
                    # Read Open time as string first
                    df = pd.read_csv(file_path, dtype={'Open time': str}) 
                    
                    # Rename columns if necessary (case-insensitive search)
                    time_col_found = 'Open time' # Assume default first
                    if 'Open time' not in df.columns:
                        time_col_found = next((c for c in df.columns if 'time' in c.lower()), None)
                        if time_col_found: df.rename(columns={time_col_found:'Open time'}, inplace=True)
                        else: raise ValueError("Timestamp column ('Open time' or similar) not found.")

                    price_col_found = 'Close' # Assume default first
                    if 'Close' not in df.columns:
                        price_col_found = next((c for c in df.columns if 'close' in c.lower()), None)
                        if price_col_found: df.rename(columns={price_col_found:'Close'}, inplace=True)
                        else: raise ValueError("Price column ('Close' or similar) not found.")
                    
                    # Parse dates robustly
                    parsed_ms = pd.to_datetime(df['Open time'], unit='ms', errors='coerce')
                    parsed_str = pd.to_datetime(df.loc[parsed_ms.isna(), 'Open time'], errors='coerce')
                    df['date_index'] = parsed_ms.fillna(parsed_str)
                    df.dropna(subset=['date_index'], inplace=True)
                    df = df.set_index('date_index')
                    # --- End Robust Date Parsing ---

                    df = df[['Close']].rename(columns={'Close': f'{asset}_Price'})
                    df[f'{asset}_Price'] = pd.to_numeric(df[f'{asset}_Price'], errors='coerce') # Ensure numeric
                    df.dropna(subset=[f'{asset}_Price'], inplace=True) # Drop if Close price is invalid
                    
                    if not df.empty:
                        all_prices[asset] = df
                        data_found = True
                        min_start_date = min(min_start_date, df.index.min())
                        
                except FileNotFoundError:
                     st.warning(f"Файл данных не найден для {asset}: {file_path}")
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
                              # Clear previous results from session state
                              if 'analysis_results' in st.session_state: del st.session_state['analysis_results']
                              
                              with st.spinner("Выполняется анализ... Это может занять некоторое время."):
                                  try:
                                     # Call the analysis function, it now returns a dictionary
                                     analysis_output = run_portfolio_analysis(
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
                                     
                                     # Store the entire results dictionary in session state
                                     st.session_state.analysis_results = analysis_output
                                     
                                     # Check for errors returned by the function
                                     if analysis_output.get("error"):
                                          st.error(f"Ошибка во время анализа: {analysis_output['error']}")
                                     elif analysis_output.get("figure") is None or analysis_output.get("metrics_display").empty:
                                          st.error("Анализ завершился, но не вернул корректных результатов (график или метрики пусты).")
                                     else:
                                          st.success("Анализ успешно завершен!")
                                          # Rerun to display results correctly
                                          st.rerun()
                                          
                                  except Exception as e:
                                      st.error(f"Произошла непредвиденная ошибка при вызове анализа: {e}")
                                      traceback.print_exc()
                                      # Clear results on critical error
                                      if 'analysis_results' in st.session_state: del st.session_state['analysis_results']
             except Exception as e:
                 st.error(f"Ошибка при подготовке данных транзакций: {e}")
                 traceback.print_exc()

        if st.session_state.get('analysis_in_progress', False):
            st.warning("Анализ портфеля выполняется...")
            st.progress(st.session_state.get('analysis_progress', 0.0))
            if 'analysis_status' in st.session_state:
                st.text(st.session_state.analysis_status)
        else:
            # Display results if available and no error
            if 'analysis_results' in st.session_state and not st.session_state.analysis_results.get("error"):
                results_dict = st.session_state.analysis_results
                st.subheader("Результаты Анализа Портфеля")

                # Display Plotly figure if it exists
                fig_analysis = results_dict.get("figure")
                if fig_analysis:
                    st.plotly_chart(fig_analysis, use_container_width=True, key="portfolio_analysis_figure") 
                else:
                    st.warning("График анализа не найден.") 

                # Display Metrics (using the formatted display DataFrame)
                st.subheader("Метрики производительности")
                metrics_display_df = results_dict.get('metrics_display', pd.DataFrame())
                if not metrics_display_df.empty:
                    st.dataframe(metrics_display_df, use_container_width=True) # Display the pre-formatted df
                else:
                     st.warning("Метрики производительности не рассчитаны.") 

                # ----- NEW: Chatbot Section -----
                st.divider() # Add a visual separator
                st.subheader("Задать вопрос о портфеле (AI Агент)")

                # Initialize chat history in session state if it doesn't exist
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat messages from history on app rerun
                chat_container = st.container(height=300)
                with chat_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                # React to user input
                if prompt := st.chat_input("Спросите что-нибудь о результатах анализа..."):
                    # Display user message in chat message container
                    with chat_container:
                         with st.chat_message("user"):
                             st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # ---- Get AI response ----
                    response = "" # Initialize response variable
                    try:
                        # 1. Check if analysis results exist
                        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
                            response = "Пожалуйста, сначала запустите анализ портфеля, чтобы я мог ответить на ваши вопросы."
                        else:
                            # 2. Initialize the agent (cached)
                            agent = initialize_finrobot_agent()
                            if agent is None:
                                response = "Ошибка: Не удалось инициализировать AI-агента. Проверьте конфигурацию и API ключи."
                            else:
                                # 3. Format the analysis data for the LLM
                                analysis_summary = format_portfolio_data_for_llm(st.session_state.analysis_results)

                                # <<< START NEW: Fetch and format user holdings >>>
                                user_holdings_str = "Данные о текущих активах пользователя недоступны."
                                try:
                                    portfolio_data = get_portfolio_with_quantities(st.session_state.username)
                                    if portfolio_data and portfolio_data.get("quantities"):
                                        holdings = portfolio_data["quantities"]
                                        # Use latest prices to calculate value (optional but nice)
                                        total_value = 0
                                        holdings_details = []
                                        latest_prices = price_data.iloc[-1] if not price_data.empty else None
                                        
                                        for asset, quantity in holdings.items():
                                            if quantity > 0:
                                                detail = f"{asset}: {quantity:.4f}" # Format quantity
                                                if latest_prices is not None and asset in latest_prices.index:
                                                     value = quantity * latest_prices[asset]
                                                     detail += f" (Стоимость: ${value:,.2f})"
                                                     total_value += value
                                                holdings_details.append(detail)

                                        if holdings_details:
                                             user_holdings_str = "**Текущие активы пользователя:**\\n"
                                             user_holdings_str += "\\n".join([f"- {d}" for d in holdings_details])
                                             if latest_prices is not None:
                                                 user_holdings_str += f"\\n\\n**Общая расчетная стоимость портфеля:** ${total_value:,.2f}"
                                        else:
                                             user_holdings_str = "У пользователя нет активов в портфеле."
                                    else:
                                        user_holdings_str = "У пользователя нет активов в портфеле."
                                except Exception as e:
                                     st.warning(f"Не удалось загрузить текущие активы пользователя: {e}")
                                     # Keep the default "unavailable" message
                                # <<< END NEW: Fetch and format user holdings >>>

                                # <<< START NEW: Fetch and format transaction history >>>
                                user_transactions_summary_str = "История транзакций пользователя недоступна."
                                try:
                                    transactions_list = get_user_transactions(st.session_state.username)
                                    if transactions_list:
                                        # Convert to DataFrame for easier manipulation and sorting
                                        transactions_df = pd.DataFrame(transactions_list)
                                        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
                                        transactions_df = transactions_df.sort_values(by='date')
                                        
                                        num_transactions = len(transactions_df)
                                        first_date = transactions_df['date'].min().strftime('%Y-%m-%d %H:%M')
                                        last_date = transactions_df['date'].max().strftime('%Y-%m-%d %H:%M')
                                        
                                        summary_lines = [
                                            f"**История транзакций пользователя:**",
                                            f"- Всего транзакций: {num_transactions}",
                                            f"- Период: с {first_date} по {last_date}"
                                        ]
                                        
                                        # Function to format a transaction row
                                        def format_tx(row):
                                            action = "Покупка" if row.get('type') == 'buy' else "Продажа"
                                            asset = row.get('asset', 'N/A')

                                            # Format quantity safely
                                            try:
                                                quantity_num = float(row['quantity'])
                                                quantity_str = f"{quantity_num:.4f}"
                                            except (ValueError, TypeError, KeyError):
                                                quantity_str = str(row.get('quantity', 'N/A'))

                                            # Format price safely
                                            try:
                                                price_num = float(row['price'])
                                                price_str = f"${price_num:,.2f}"
                                            except (ValueError, TypeError, KeyError):
                                                price_str = str(row.get('price', 'N/A'))

                                            # Format cost safely (only for 'buy')
                                            cost_str = ""
                                            if action == 'Покупка':
                                                total_cost_val = row.get('total_cost') # Get value or None
                                                try:
                                                    if total_cost_val is not None:
                                                        cost_num = float(total_cost_val)
                                                        cost_str = f" на сумму ${cost_num:,.2f}"
                                                    else:
                                                        # If total_cost is None or missing, try calculating from qty*price + fee
                                                        try:
                                                             fee = float(row.get('fee', 0))
                                                             calculated_cost = float(row['quantity']) * float(row['price']) + fee
                                                             cost_str = f" на сумму ${calculated_cost:,.2f}"
                                                        except (ValueError, TypeError, KeyError):
                                                             cost_str = " на сумму N/A" # Fallback if calculation fails
                                                except (ValueError, TypeError):
                                                    cost_str = f" на сумму {str(total_cost_val)}" # Show original if not float

                                            # Format date safely
                                            try:
                                                # Assuming 'date' is already a datetime object from earlier conversion
                                                date_str = row['date'].strftime('%Y-%m-%d') 
                                            except (AttributeError, KeyError, TypeError):
                                                date_str = str(row.get('date', 'N/A'))
                                                
                                            return f"{date_str}: {action} {quantity_str} {asset} @ {price_str}{cost_str}"
                                            
                                        if num_transactions > 0:
                                            summary_lines.append("\n*Недавние транзакции:*" )
                                            # Show last 3 (or fewer if less than 3 total)
                                            for i, row in transactions_df.tail(min(num_transactions, 3)).iterrows():
                                                summary_lines.append(f"  - {format_tx(row)}")
                                        if num_transactions > 6: # Show first 3 only if there are more than 6 total to avoid redundancy
                                            summary_lines.append("\n*Первые транзакции:*" )
                                            for i, row in transactions_df.head(3).iterrows():
                                                 summary_lines.append(f"  - {format_tx(row)}")
                                                 
                                        user_transactions_summary_str = "\n".join(summary_lines)
                                        
                                    else:
                                        user_transactions_summary_str = "У пользователя нет истории транзакций."
                                except Exception as e:
                                    st.warning(f"Не удалось загрузить или обработать историю транзакций: {e}")
                                    # Keep the default "unavailable" message
                                # <<< END NEW: Fetch and format transaction history >>>

                                if not analysis_summary or analysis_summary == "Нет данных для анализа.":
                                     response = "Не удалось получить данные анализа для формирования контекста."
                                else:
                                     # 4. Construct the full prompt
                                     # <<< MODIFIED: Include holdings and transaction summary in the prompt >>>
                                     full_prompt = f"""
                                     Ты — эксперт по анализу инвестиционных портфелей и предоставлению рекомендаций.\n
                                     Основываясь **строго и только** на предоставленных ниже данных (анализ производительности, текущие активы и история транзакций), ответь на вопрос пользователя **на русском языке**.

                                     **1. Результаты последнего анализа производительности портфеля:**
                                     ```markdown
                                     {analysis_summary}
                                     ```

                                     **2. Текущие активы пользователя в портфеле:**
                                     ```markdown
                                     {user_holdings_str}
                                     ```
                                     
                                     **3. Краткая история транзакций пользователя:**
                                     ```markdown
                                     {user_transactions_summary_str}
                                     ```

                                     **Вопрос пользователя:**
                                     '{prompt}'
                                     """
                                     # <<< END MODIFICATION >>>

                                     # 5. Call the agent
                                     with st.spinner("🤖 AI-агент думает..."):
                                          # FinRobot's SingleAssistant might return the response directly
                                          # or store it in the conversation history. Adjust based on its behavior.
                                          # Assuming agent.chat returns the final message content:
                                          agent_reply = agent.chat(full_prompt)
                                          # Extract the actual response content if needed (depends on agent's return format)
                                          # This might need adjustment based on how SingleAssistant returns replies
                                          if isinstance(agent_reply, dict) and 'content' in agent_reply:
                                              response = agent_reply['content']
                                          elif isinstance(agent_reply, str):
                                               response = agent_reply
                                          else:
                                               # Fallback: Try to get the last message from the agent's history
                                               if hasattr(agent, 'agent') and hasattr(agent.agent, 'chat_messages') and agent.agent.chat_messages:
                                                    last_msg = agent.agent.chat_messages.get(agent.agent.opponent, [])[-1]
                                                    response = last_msg.get('content', "Не удалось извлечь ответ агента.")
                                               else:
                                                   response = "Получен неожиданный формат ответа от агента."

                    except Exception as e:
                        response = f"Произошла ошибка при обращении к AI: {e}"
                        traceback.print_exc()

                    # ---- Display AI response ----
                    with chat_container:
                         with st.chat_message("assistant"):
                            st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Rerun to ensure the message list is updated cleanly in the container
                    st.rerun()

            # Display error if analysis failed
            elif 'analysis_results' in st.session_state and st.session_state.analysis_results.get("error"):
                 st.error(f"Ошибка во время анализа: {st.session_state.analysis_results['error']}")
            else:
                 st.info("Запустите анализ, чтобы увидеть результаты.") 

    # --- End Section: Portfolio Analysis ---

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

    # <<< Add block for the new Recommendations page >>>
    elif st.session_state.active_page == "Рекомендации":
        st.header("Рекомендации по ребалансировке портфеля (DRL)")
        st.markdown("Выберите DRL-стратегию для получения рекомендуемого распределения активов из **набора, на котором обучалась модель.**")

        # --- Настройки Рекомендации ---
        st.subheader("Параметры")
        drl_model_names = ["A2C", "PPO", "SAC", "DDPG"]
        
        # Remove columns and number input, keep only model selection
        selected_model_name = st.selectbox("Выберите DRL модель:", drl_model_names, key="drl_model_select")
        # Remove the number input for rebalancing interval
        # reco_rebalance_interval = st.number_input(...)

        # --- Define DRL training assets --- #
        # !! Важно: Этот список ДОЛЖЕН точно совпадать с активами, использованными при обучении моделей !!
        DRL_TRAINING_ASSETS_RECO = ['APTUSDT', 'CAKEUSDT', 'HBARUSDT', 'JUPUSDT', 'PEPEUSDT', 'STRKUSDT', 'USDCUSDT']
        st.info(f"Модель DRL была обучена на определенном наборе активов. Рекомендация ниже показывает целевое распределение **только для тех активов из этого набора, которые есть у вас сейчас**, плюс **{STABLECOIN_ASSET}** как тихую гавань.")

        data_path_reco = "data"
        drl_models_dir_reco = "notebooks/trained_models"
        st.caption(f"Путь к данным: {os.path.abspath(data_path_reco)}, Путь к моделям: {os.path.abspath(drl_models_dir_reco)}")

        # Кнопка для получения рекомендации
        if st.button(f"Получить рекомендацию от {selected_model_name}", use_container_width=True):
            with st.spinner(f"Загрузка данных и запуск модели {selected_model_name}..."):
                try:
                    # --- 1. Get Current Portfolio Value --- #
                    portfolio_data_reco = get_portfolio_with_quantities(st.session_state.username)
                    total_portfolio_value = 0
                    if portfolio_data_reco and any(portfolio_data_reco["quantities"].values()):
                        # Use price_data loaded globally for the app
                        latest_prices_reco = price_data.iloc[-1] if not price_data.empty else None
                        if latest_prices_reco is not None:
                            for asset, quantity in portfolio_data_reco["quantities"].items():
                                if quantity > 0 and asset in latest_prices_reco.index:
                                    total_portfolio_value += quantity * latest_prices_reco[asset]
                        else:
                            st.error("Не удалось получить последние цены активов для расчета общей стоимости портфеля.")
                            st.stop()
                    if total_portfolio_value < 1e-6:
                        st.warning("Текущая стоимость портфеля равна нулю или не удалось ее рассчитать. Рекомендация невозможна.")
                        st.stop()

                    st.write(f"Текущая общая стоимость вашего портфеля: ${total_portfolio_value:,.2f}")

                    # --- 2. Load Latest Data for DRL Assets --- #
                    # Determine lookback period (max indicator window + cov lookback)
                    # Example: Max window = 60 (SMA), Cov lookback = 24 -> need ~84 hours + buffer
                    lookback_days_data = 5 # Load last 5 days of hourly data (adjust as needed)
                    end_date_data = datetime.now()
                    start_date_data = end_date_data - timedelta(days=lookback_days_data)

                    all_drl_data_frames = []
                    print(f"Loading latest data for {len(DRL_TRAINING_ASSETS_RECO)} DRL assets...")
                    print(f"Filtering data between: {start_date_data} and {end_date_data}")
                    for asset in DRL_TRAINING_ASSETS_RECO:
                        filepath = os.path.join(data_path_reco, f"{asset}_hourly_data.csv")
                        df_asset = preprocess_asset_data(filepath, asset, STABLECOIN_ASSET)
                        if not df_asset.empty:
                             print(f"  {asset}: Loaded {len(df_asset)} rows. Date range: {df_asset['date'].min()} to {df_asset['date'].max()}")
                             df_asset_filtered = df_asset[(df_asset['date'] >= start_date_data) & (df_asset['date'] <= end_date_data)]
                             print(f"  {asset}: Filtered to {len(df_asset_filtered)} rows.")
                             if not df_asset_filtered.empty:
                                all_drl_data_frames.append(df_asset_filtered)
                        else:
                              st.error(f"Не удалось загрузить или обработать критически важные данные для DRL актива: {asset}. Рекомендация невозможна.")
                              st.stop()

                    if not all_drl_data_frames:
                         st.error(f"Не найдено актуальных данных для DRL активов в диапазоне {start_date_data.strftime('%Y-%m-%d')} - {end_date_data.strftime('%Y-%m-%d')}. Проверьте актуальность CSV файлов.")
                         st.stop()

                    latest_drl_data = pd.concat(all_drl_data_frames, ignore_index=True)

                    # --- 3. Preprocess Data (Features + Covariance) --- #
                    print("Preprocessing latest data (FeatureEngineer)...")
                    fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS)
                    drl_processed = fe.preprocess_data(latest_drl_data.copy()) # Pass copy
                    # Note: fe.preprocess_data now includes dropna() based on previous steps
                    if drl_processed.empty:
                        st.error("Ошибка FeatureEngineer: DataFrame пуст после добавления индикаторов и dropna(). Возможно, недостаточно свежих данных.")
                        st.stop()

                    print("Calculating latest covariance matrix...")
                    lookback_cov = 24 # Standard lookback for covariance
                    drl_processed = drl_processed.sort_values(['date', 'tic'])
                    last_date_available = drl_processed['date'].max()
                    # Get data for the last lookback_cov hours ending at the last available date
                    cov_data_input = drl_processed[drl_processed['date'] > (last_date_available - timedelta(hours=lookback_cov))].copy()
                    if len(cov_data_input['date'].unique()) < lookback_cov / 2: # Basic check for enough data
                         st.error(f"Недостаточно данных ({len(cov_data_input['date'].unique())} часов) для расчета ковариации (требуется около {lookback_cov} часов).")
                         st.stop()

                    price_lookback = cov_data_input.pivot_table(index='date', columns='tic', values='close')
                    price_lookback = price_lookback.reindex(columns=DRL_TRAINING_ASSETS_RECO, fill_value=np.nan)
                    return_lookback = price_lookback.pct_change().dropna(how='all')
                    latest_cov_matrix = np.zeros((len(DRL_TRAINING_ASSETS_RECO), len(DRL_TRAINING_ASSETS_RECO)))
                    if len(return_lookback) >= len(DRL_TRAINING_ASSETS_RECO):
                         return_lookback_valid = return_lookback.dropna(axis=1, how='all')
                         if not return_lookback_valid.empty and return_lookback_valid.shape[1] > 1:
                              latest_cov_matrix = return_lookback_valid.cov().reindex(index=DRL_TRAINING_ASSETS_RECO, columns=DRL_TRAINING_ASSETS_RECO).fillna(0).values

                    # --- 4. Construct Latest State --- #
                    print("Constructing latest observation state...")
                    last_processed_data = drl_processed[drl_processed['date'] == last_date_available]
                    if last_processed_data.empty:
                         st.error("Не удалось извлечь данные для последнего состояния после обработки.")
                         st.stop()
                    # Ensure data is sorted by the canonical asset list order for consistency
                    last_processed_data = last_processed_data.set_index('tic').reindex(DRL_TRAINING_ASSETS_RECO).reset_index()

                    # Extract indicators for the last timestamp for all DRL assets
                    indicator_values = last_processed_data[INDICATORS].values.T # Shape (num_indicators, num_assets)
                    # Combine cov matrix and indicators
                    latest_state = np.append(latest_cov_matrix, indicator_values, axis=0) # Shape (stock_dim + num_ind, stock_dim)
                    # Ensure state shape matches expected (15, 7)
                    expected_shape = (len(DRL_TRAINING_ASSETS_RECO) + len(INDICATORS), len(DRL_TRAINING_ASSETS_RECO))
                    if latest_state.shape != expected_shape:
                        st.error(f"Ошибка формы состояния: получено {latest_state.shape}, ожидалось {expected_shape}. Проверьте данные и индикаторы.")
                        st.stop()

                    # --- 5. Load Selected Model --- #
                    print(f"Loading model {selected_model_name}...")
                    model_file = os.path.join(drl_models_dir_reco, f"trained_{selected_model_name.lower()}.zip")
                    if not os.path.exists(model_file):
                         st.error(f"Файл модели не найден: {model_file}")
                         st.stop()
                    model_class_reco = globals()[selected_model_name] # Get class by name
                    model = model_class_reco.load(model_file)

                    # --- 6. Predict Action --- #
                    print("Predicting action...")
                    # Model expects a batch dimension, add it if predicting single state
                    # latest_state_batch = np.expand_dims(latest_state, axis=0)
                    # However, stable-baselines3 predict usually handles single obs automatically
                    raw_actions, _ = model.predict(latest_state, deterministic=True)

                    # --- 7. Normalize Weights --- #
                    print("Normalizing weights...")
                    target_weights_raw = softmax_normalization(raw_actions)
                    target_weights_dict_raw = {asset: weight for asset, weight in zip(DRL_TRAINING_ASSETS_RECO, target_weights_raw)}

                    # --- 8. Filter, Renormalize, Calculate Target Values & Display --- #
                    st.markdown("--- ")
                    st.subheader(f"Рекомендация от {selected_model_name} (для текущих активов + {STABLECOIN_ASSET})")
                    
                    # Remove the informational text about the rebalancing interval
                    # st.info(f"Рекомендация основана на текущих рыночных данных. \nПланируйте следующую ребалансировку через **{reco_rebalance_interval}** дней.")

                    # Get current user assets (only those also in DRL training set)
                    current_user_assets_in_drl_set = set()
                    if portfolio_data_reco and any(portfolio_data_reco["quantities"].values()):
                         current_user_assets_in_drl_set = {
                              asset for asset, quantity in portfolio_data_reco["quantities"].items()
                              if quantity > 1e-9 and asset in DRL_TRAINING_ASSETS_RECO
                         }

                    # Define the set of assets to consider for the final recommendation
                    relevant_assets = current_user_assets_in_drl_set.copy()
                    relevant_assets.add(STABLECOIN_ASSET) # Always include stablecoin

                    # Filter the raw weights to include only relevant assets
                    filtered_weights = {asset: target_weights_dict_raw.get(asset, 0.0)
                                        for asset in relevant_assets}

                    # Renormalize the filtered weights
                    total_filtered_weight = sum(filtered_weights.values())
                    final_target_weights = {}
                    if total_filtered_weight > 1e-9:
                        final_target_weights = {asset: weight / total_filtered_weight
                                                for asset, weight in filtered_weights.items()}
                    elif relevant_assets: # If sum is zero, but we have assets, distribute equally
                        print("Warning: Sum of filtered weights is zero. Distributing equally among relevant assets.")
                        num_relevant = len(relevant_assets)
                        final_target_weights = {asset: 1.0 / num_relevant for asset in relevant_assets}
                    else: # Should not happen if stablecoin is always added
                         st.error("Не удалось определить релевантные активы для рекомендации.")
                         st.stop()

                    st.markdown(f"Модель предлагает следующее распределение капитала (${total_portfolio_value:,.2f}) между **вашими активами из набора DRL и {STABLECOIN_ASSET}**: ")

                    # Calculate final target values
                    results_list = []
                    for asset, weight in final_target_weights.items():
                        target_value = total_portfolio_value * weight
                        results_list.append({
                            "Актив": asset,
                            "Рекомендуемый вес (%)": weight * 100,
                            "Целевая стоимость ($": target_value
                        })

                    results_df_reco = pd.DataFrame(results_list)
                    results_df_reco = results_df_reco.sort_values(by="Рекомендуемый вес (%)", ascending=False).reset_index(drop=True)

                    # Display table
                    st.dataframe(results_df_reco.style.format({
                        "Рекомендуемый вес (%)": "{:.2f}%",
                        "Целевая стоимость ($": "${:,.2f}"
                    }))

                    # Optional: Display as bar chart
                    st.subheader("Визуализация весов")
                    # Use the final filtered/renormalized weights for the chart
                    chart_data = results_df_reco.set_index("Актив")["Рекомендуемый вес (%)"] / 100.0
                    st.bar_chart(chart_data)

                    st.success("Рекомендация успешно сгенерирована!")

                except Exception as e:
                    st.error(f"Произошла ошибка при генерации рекомендации: {e}")
                    traceback.print_exc()
        pass # End of Recommendations page block

    # <<< Add block for the new Data & Analysis page >>>
    elif st.session_state.active_page == "Данные и Анализ":
        st.header("Управление данными и Анализ рынка")
        st.markdown("Здесь вы можете обновить исторические данные активов с Binance и просмотреть базовый анализ рынка.")

        # --- Initialize session state for news analysis if not present ---
        if 'news_analysis_results' not in st.session_state:
            st.session_state.news_analysis_results = None
        # ---------------------------------------------------------------

        # --- Обновление данных --- #
        st.subheader("Обновление данных активов")
        st.warning("**Внимание:** Для обновления данных с Binance необходимо настроить API ключи в секретах Streamlit или переменных окружения (`BINANCE_API_KEY`, `BINANCE_API_SECRET`).")

        # Initialize update status if not present
        if 'last_update_status' not in st.session_state:
            st.session_state.last_update_status = None
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = None
        if 'update_counter' not in st.session_state: # For cache invalidation
            st.session_state.update_counter = 0

        col1_update, col2_update = st.columns([3, 1])
        with col1_update:
            # Rename the button
            if st.button("🔄 Обновить рыночные данные", use_container_width=True):
                st.session_state.last_update_status = None # Reset status
                st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                progress_bar = st.progress(0, text="Запуск обновления...")
                try:
                    success_update, message_update = update_all_asset_data(progress_bar=progress_bar)
                    st.session_state.last_update_status = message_update
                    if success_update:
                        progress_bar.progress(1.0, text="Обновление файлов завершено. Создание объединенного файла...")
                        # If update was successful, recreate combined file
                        success_combine, message_combine = create_combined_data()
                        if success_combine:
                            st.session_state.last_update_status += f" {message_combine}"
                            st.session_state.update_counter += 1 # Increment counter to invalidate cache
                            st.success(f"Обновление успешно завершено! {message_combine}")
                            st.rerun() # Rerun to reload data with new trigger
                        else:
                            st.session_state.last_update_status += f" Ошибка объединения: {message_combine}"
                            st.error(f"Файлы обновлены, но не удалось создать объединенный файл: {message_combine}")
                    else:
                        st.error(f"Ошибка при обновлении CSV файлов: {message_update}")
                    progress_bar.empty() # Remove progress bar
                except Exception as e:
                    st.error(f"Непредвиденная ошибка во время обновления: {e}")
                    traceback.print_exc()
                    st.session_state.last_update_status = f"Критическая ошибка: {e}"
                    progress_bar.empty()

        # Display last update status
        if st.session_state.last_update_status:
             st.info(f"Статус последнего обновления ({st.session_state.last_update_time}): {st.session_state.last_update_status}")

        st.markdown("--- ")
        st.subheader("Анализ рынка")

        # Load data using the cached function with the trigger
        # Display loading spinner while data is loading via cache
        with st.spinner("Загрузка агрегированных данных..."):
             combined_df = load_combined_data_cached(st.session_state.update_counter)

        if not combined_df.empty:
            st.dataframe(combined_df.tail()) # Show recent combined data tail

            # --- Normalized Plot Section ---
            st.markdown("#### Нормализованная динамика цен")
            # Add period selection for normalized plot
            norm_period_options = {"7 дней": 7, "30 дней": 30, "90 дней": 90, "180 дней": 180, "Все время": None}
            selected_norm_period_label = st.radio(
                "Период для нормализации:", 
                options=norm_period_options.keys(), 
                index=3, # Default to 180 days
                horizontal=True, 
                key="norm_period_radio"
            )
            selected_norm_days = norm_period_options[selected_norm_period_label]

            try:
                with st.spinner("Генерация нормализованного графика..."):
                     # Pass selected days to the plotting function
                     fig_norm = generate_normalized_plot(combined_df, days=selected_norm_days)
                     st.plotly_chart(fig_norm, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при построении графика нормализованных цен: {e}")

            # --- Correlation Heatmap Section ---
            st.markdown("#### Корреляция доходностей") # Adjusted title slightly
            # Add period selection for correlation heatmap
            corr_period_options = {
                "1 час": "h", "1 день": "D", "3 дня": "3D", "1 неделя": "W", 
                "1 месяц": "MS", "3 месяца": "3MS", "1 год": "YS"
            }
            selected_corr_period_label = st.radio(
                "Период расчета доходности для корреляции:", 
                options=corr_period_options.keys(), 
                index=3, # Default to 1 неделя
                horizontal=True, 
                key="corr_period_radio"
            )
            selected_corr_freq = corr_period_options[selected_corr_period_label]
            
            try:
                with st.spinner(f"Генерация матрицы корреляций ({selected_corr_period_label})..."):
                     # Pass selected frequency to the plotting function
                     fig_corr = generate_correlation_heatmap(combined_df, frequency=selected_corr_freq)
                     st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при построении матрицы корреляций: {e}")

            # --- NEW: Single Asset Plot Section --- 
            st.markdown("--- ")
            st.subheader("График цены актива")

            if not combined_df.empty:
                # Get available assets from combined_df columns
                available_assets = combined_df.columns.tolist()

                col1_asset, col2_res = st.columns([2,3])
                with col1_asset:
                    selected_asset = st.selectbox("Выберите актив:", available_assets, key="single_asset_select")
                with col2_res:
                    resolution_options = {"1 час": "h", "4 часа": "4h", "1 день": "D", "1 неделя": "W", "1 месяц": "MS"}
                    selected_resolution_label = st.radio("Таймфрейм:", 
                                                       options=resolution_options.keys(), 
                                                       index=0, # Default to 1h
                                                       horizontal=True, 
                                                       key="resolution_radio"
                                                    )
                    selected_resolution_code = resolution_options[selected_resolution_label]
                
                # Generate and display the single asset plot
                if selected_asset:
                    try:
                         with st.spinner(f"Генерация графика {selected_asset} ({selected_resolution_label})..."):
                              # Import the new function if not already imported at the top
                              from portfolios_optimization.data_loader import generate_single_asset_plot 
                              
                              # Call the function to generate the plot
                              fig_single = generate_single_asset_plot(combined_df, selected_asset, selected_resolution_code)
                              st.plotly_chart(fig_single, use_container_width=True)
                              # Remove the placeholder info message
                              # st.info(f"Логика для графика {selected_asset} с таймфреймом {selected_resolution_label} будет добавлена.") 
                    except Exception as e:
                         st.error(f"Ошибка при построении графика для {selected_asset}: {e}")
                         traceback.print_exc() # Print detailed error in console

            # <<< START NEW: FinNLP Analysis Section >>>
            # --- UPDATED: Section title to reflect Transformers usage --- 
            st.markdown("--- ")
            st.subheader(f"Анализ новостей по активу")

            # --- Determine available news assets --- 
            news_data_base_dir = "notebooks/news_data"
            available_news_assets_map = {}
            try:
                # Scan subdirectories in news_data
                asset_dirs = [d for d in os.listdir(news_data_base_dir) if os.path.isdir(os.path.join(news_data_base_dir, d))]
                # Try to map dir name (e.g., 'btc') back to a ticker format (e.g., 'BTCUSDT') if possible
                # This assumes tickers are like {ASSET}USDT
                all_price_tickers = combined_df.columns.tolist() if not combined_df.empty else []
                for asset_dir_name in asset_dirs:
                    # Find a matching ticker from price data
                    matching_ticker = next((ticker for ticker in all_price_tickers if ticker.lower().startswith(asset_dir_name.lower())), None)
                    if matching_ticker:
                        available_news_assets_map[matching_ticker] = asset_dir_name # Store map: BTCUSDT -> btc
                    else:
                        # If no ticker matches, use the directory name as a fallback key (less ideal)
                        available_news_assets_map[asset_dir_name.upper()] = asset_dir_name
            except FileNotFoundError:
                 st.warning(f"Директория с новостями не найдена: {news_data_base_dir}")
                 available_news_assets_map = {}
            
            if not available_news_assets_map:
                 st.warning("Не найдены доступные активы с новостями для анализа.")
            else:
                # --- Get User Input --- 
                col1_news_opts, col2_news_opts = st.columns([2, 3])
                with col1_news_opts:
                    selected_asset_ticker_news = st.selectbox(
                        "Выберите актив для анализа новостей:", 
                        options=list(available_news_assets_map.keys()), # Use tickers as options
                        key="news_asset_select"
                    )
                    # Get the corresponding directory/asset name from the selected ticker
                    selected_asset_name_news = available_news_assets_map.get(selected_asset_ticker_news)
                
                with col2_news_opts:
                     # Date range selection
                     today_date_news = datetime.now().date()
                     default_start_date_news = today_date_news - timedelta(days=7) 
                     news_start_date = st.date_input("Начальная дата новостей", value=default_start_date_news, max_value=today_date_news, key="news_start_date")
                     news_end_date = st.date_input("Конечная дата новостей", value=today_date_news, min_value=news_start_date, max_value=today_date_news, key="news_end_date")
                     # --- UPDATED: Increased max_value and default --- 
                     num_articles_to_analyze = st.number_input("Макс. кол-во новостей", min_value=1, max_value=2000, value=20, step=5, key="num_articles_news")

                # --- Analysis Button --- 
                if st.button("📰 Анализировать новости", key="analyze_news_button"):
                    st.session_state.news_analysis_results = None # Clear previous results before new analysis
                    if selected_asset_name_news:
                        spinner_text = f"Получение и анализ {num_articles_to_analyze} новостей для {selected_asset_ticker_news} ({news_start_date} - {news_end_date})..."
                        with st.spinner(spinner_text):
                            news_fetch_result = fetch_news_from_csv(
                                selected_asset_name_news,
                                start_date=news_start_date,
                                end_date=news_end_date,
                                num_articles=num_articles_to_analyze
                            )
                            
                            if news_fetch_result:
                                summaries = news_fetch_result.get("summaries", [])
                                
                                # --- Perform Sentiment Analysis --- 
                                @st.cache_resource
                                def get_sentiment_pipeline(): # Inner function definition is fine here
                                    model_name = "ProsusAI/finbert"
                                    try:
                                        device = 0 if torch.cuda.is_available() else -1
                                        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
                                        print(f"Sentiment pipeline loaded: {model_name} on device {'cuda:0' if device==0 else 'cpu'}")
                                        return sentiment_pipeline
                                    except Exception as e:
                                        st.error(f"Error loading pipeline '{model_name}': {e}")
                                        # ... (fallback logic remains the same) ...
                                        try:
                                            device = 0 if torch.cuda.is_available() else -1
                                            sentiment_pipeline = pipeline("sentiment-analysis", device=device)
                                            print(f"Default sentiment pipeline loaded on device {'cuda:0' if device==0 else 'cpu'}")
                                            return sentiment_pipeline
                                        except Exception as e_fallback:
                                            st.error(f"Error loading default sentiment pipeline: {e_fallback}")
                                            return None

                                sentiment_pipeline_instance = get_sentiment_pipeline()
                                
                                if summaries and sentiment_pipeline_instance:
                                    try:
                                        # ... (limit summaries, check valid_summaries logic) ...
                                        max_summaries_for_analysis = 100
                                        summaries_to_analyze = summaries[:max_summaries_for_analysis] if len(summaries) > max_summaries_for_analysis else summaries
                                        valid_summaries = [str(s) for s in summaries_to_analyze if pd.notna(s) and isinstance(s, str)]

                                        if valid_summaries:
                                            results = sentiment_pipeline_instance(valid_summaries)
                                            # ... (calculate scores, counts, news_with_sentiment list) ...
                                            sentiment_scores = []
                                            positive_count = 0
                                            negative_count = 0
                                            neutral_count = 0
                                            news_with_sentiment = []
                                            for summary, result in zip(valid_summaries, results):
                                                label = result['label'].upper()
                                                score = result['score']
                                                numeric_score = 0.0
                                                if label == 'POSITIVE' or label == 'POS' or label == 'LABEL_1':
                                                    positive_count += 1
                                                    numeric_score = score
                                                elif label == 'NEGATIVE' or label == 'NEG' or label == 'LABEL_0':
                                                    negative_count += 1
                                                    numeric_score = -score
                                                else:
                                                    neutral_count += 1
                                                    numeric_score = 0.0
                                                sentiment_scores.append(numeric_score)
                                                news_with_sentiment.append({"summary": summary, "label": label, "score": numeric_score})

                                            total_analyzed = len(sentiment_scores)
                                            if total_analyzed > 0:
                                                # --- Store results in session state --- 
                                                st.session_state.news_analysis_results = {
                                                    "ticker": selected_asset_ticker_news,
                                                    "fetched_summaries": summaries, # Store original fetched ones
                                                    "analyzed_summaries_sentiment": news_with_sentiment,
                                                    "avg_score": sum(sentiment_scores) / total_analyzed,
                                                    "positive_pct": (positive_count / total_analyzed) * 100,
                                                    "negative_pct": (negative_count / total_analyzed) * 100,
                                                    "neutral_pct": (neutral_count / total_analyzed) * 100,
                                                    "positive_count": positive_count,
                                                    "negative_count": negative_count,
                                                    "neutral_count": neutral_count,
                                                    "actual_start_date": news_fetch_result["start_date"],
                                                    "actual_end_date": news_fetch_result["end_date"],
                                                    "num_articles_fetched": news_fetch_result["count"],
                                                    "error": None
                                                }
                                                st.success("Анализ новостей завершен.") # Indicate success
                                            else:
                                                st.warning("Не удалось извлечь оценки тональности.")
                                                st.session_state.news_analysis_results = {"error": "Не удалось извлечь оценки тональности."}
                                        else:
                                            st.warning("Нет валидных текстов новостей для анализа.")
                                            st.session_state.news_analysis_results = {"error": "Нет валидных текстов новостей для анализа."}
                                    except Exception as e:
                                        st.error(f"Ошибка во время анализа тональности: {e}")
                                        traceback.print_exc()
                                        st.session_state.news_analysis_results = {"error": f"Ошибка анализа тональности: {e}"}
                                else:
                                     st.warning("Нет сводок новостей или не удалось загрузить модель тональности.")
                                     st.session_state.news_analysis_results = {"error": "Нет сводок/модели тональности."}
                            else:
                                # Error handled in fetch_news_from_csv
                                st.session_state.news_analysis_results = {"error": "Ошибка загрузки новостей."}
                    else:
                        st.warning("Пожалуйста, выберите актив.")
                        st.session_state.news_analysis_results = None # Ensure state is clear

                # --- Display Results Block (conditional on session state) ---
                if st.session_state.get('news_analysis_results') and not st.session_state.news_analysis_results.get('error'):
                    results = st.session_state.news_analysis_results
                    st.markdown(f"**Результаты анализа {results['num_articles_fetched']} новостей для {results['ticker']} ({results['actual_start_date'].strftime('%Y-%m-%d')} - {results['actual_end_date'].strftime('%Y-%m-%d')}):**")
                    
                    # Display combined text summary (optional, can be removed if too long)
                    combined_text = "\n\n---\n\n".join([str(s['summary']) for s in results['analyzed_summaries_sentiment'][:10] if pd.notna(s['summary'])]) + "..."
                    st.text_area("Примеры сводок:", combined_text, height=150, disabled=True, key="news_display_area_results")
                    st.markdown("--- ")

                    # Display Metrics
                    st.markdown("**Общая тональность:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Positive", f"{results['positive_pct']:.1f}%", f"{results['positive_count']} шт.")
                    col2.metric("Negative", f"{results['negative_pct']:.1f}%", f"{results['negative_count']} шт.", delta_color="inverse")
                    col3.metric("Neutral", f"{results['neutral_pct']:.1f}%", f"{results['neutral_count']} шт.", delta_color="off")
                    col4.metric("Средний балл", f"{results['avg_score']:.2f}", help="-1 (Neg) до +1 (Pos)")

                    # AI Summary Button
                    if st.button("💬 Сгенерировать сводку AI", key="generate_ai_summary_sess"):
                        # Agent initialization should still be cached
                        finrobot_agent = initialize_finrobot_agent()
                        if finrobot_agent:
                            with st.spinner("🤖 Llama3 генерирует сводку..."):
                                try:
                                    # Get data from session state
                                    data_for_prompt = st.session_state.news_analysis_results
                                    news_with_sentiment = data_for_prompt['analyzed_summaries_sentiment']
                                    
                                    n_examples = 3
                                    # Sorting is needed again as it's not stored sorted
                                    news_with_sentiment.sort(key=lambda x: x['score'], reverse=True)
                                    positive_examples = [item['summary'] for item in news_with_sentiment if item['score'] > 0][:n_examples]
                                    negative_examples = [item['summary'] for item in news_with_sentiment if item['score'] < 0][::-1][:n_examples]
                                    pos_examples_str = "\n".join([f"- {ex[:150]}..." for ex in positive_examples]) if positive_examples else "Нет ярких позитивных примеров"
                                    neg_examples_str = "\n".join([f"- {ex[:150]}..." for ex in negative_examples]) if negative_examples else "Нет ярких негативных примеров"
                                    
                                    llama_prompt = (
                                        f"Ты - финансовый аналитик. Проанализируй статистику тональности и примеры новостей по активу {data_for_prompt['ticker']}. "
                                        f"Сформируй краткую сводку (2-4 предложения) на русском языке, объясняющую общую тональность новостного фона и ключевые факторы "
                                        f"(события/темы из новостей, если возможно их уловить), влияющие на позитивные и негативные настроения.\n\n"
                                        f"**Общая статистика тональности ({data_for_prompt['actual_start_date'].strftime('%Y-%m-%d')} - {data_for_prompt['actual_end_date'].strftime('%Y-%m-%d')}):**\n"
                                        f"- Доля позитивных новостей: {data_for_prompt['positive_pct']:.1f}% ({data_for_prompt['positive_count']} шт.)\n"
                                        f"- Доля негативных новостей: {data_for_prompt['negative_pct']:.1f}% ({data_for_prompt['negative_count']} шт.)\n"
                                        f"- Доля нейтральных новостей: {data_for_prompt['neutral_pct']:.1f}% ({data_for_prompt['neutral_count']} шт.)\n"
                                        f"- Средний балл тональности (от -1 до +1): {data_for_prompt['avg_score']:.2f}\n\n"
                                        f"**Примеры позитивных новостей:**\n{pos_examples_str}\n\n"
                                        f"**Примеры негативных новостей:**\n{neg_examples_str}\n\n"
                                        f"**Твоя краткая сводка:**"
                                    )
                                    
                                    print("--- PROMPT FOR LLAMA (from session state) ---")
                                    print(llama_prompt)
                                    print("--- END PROMPT ---")
                                    
                                    response = None
                                    try:
                                        # Make sure .chat() is correct method
                                        response = finrobot_agent.chat(llama_prompt)
                                        st.success("Ответ от AI получен.")
                                    except Exception as agent_call_e:
                                        st.error(f"Ошибка при вызове AI агента ({type(agent_call_e).__name__}): {agent_call_e}")
                                        traceback.print_exc()
                                    
                                    # Store response in session state to persist it
                                    st.session_state.news_analysis_ai_summary = response 
                                        
                                except Exception as llm_e:
                                    st.error(f"Ошибка при подготовке запроса AI: {llm_e}")
                                    traceback.print_exc()
                                    st.session_state.news_analysis_ai_summary = {"error": f"Ошибка подготовки запроса: {llm_e}"}
                        else:
                            st.error("Не удалось инициализировать AI-агента.")
                            st.session_state.news_analysis_ai_summary = {"error": "Агент не инициализирован"}
                            
                    # --- Display AI Summary (conditional on its existence in session state) ---
                    if st.session_state.get('news_analysis_ai_summary'):
                        ai_summary = st.session_state.news_analysis_ai_summary
                        st.subheader("Сводка от AI (Llama3):")
                        if isinstance(ai_summary, dict) and ai_summary.get('error'):
                            st.error(f"Ошибка генерации сводки: {ai_summary['error']}")
                        elif isinstance(ai_summary, dict) and 'content' in ai_summary:
                            st.markdown(ai_summary['content'])
                        elif isinstance(ai_summary, str):
                            st.markdown(ai_summary)
                        else:
                            st.warning("Не удалось отобразить ответ AI.")
                            st.write(ai_summary)
                    # -------------------------------------------------------------------------
                    
                    # Display detailed news table
                    with st.expander("Показать новости с оценкой тональности"):
                        df_sentiment = pd.DataFrame(results['analyzed_summaries_sentiment'])
                        # ... (highlighting logic remains the same) ...
                        def highlight_sentiment(row):
                            score = row['score']
                            if score > 0.1: color = 'background-color: #2a4f38'
                            elif score < -0.1: color = 'background-color: #5a2a2a'
                            else: color = ''
                            return [color] * len(row)
                        st.dataframe(
                            df_sentiment[['label', 'score', 'summary']]
                            .style
                            .apply(highlight_sentiment, axis=1)
                            .format({'score': '{:+.2f}'}),
                            use_container_width=True
                        )
                # --- End Display Results Block ---
                elif st.session_state.get('news_analysis_results') and st.session_state.news_analysis_results.get('error'):
                     # Show error if analysis failed
                     st.error(f"Ошибка анализа новостей: {st.session_state.news_analysis_results['error']}")

            # <<< END: News Analysis Section >>>
            
        else:
             st.warning(f"Не удалось загрузить данные из 'data/data_compare_eda.csv'. Запустите обновление данных.")
        pass # End of Data & Analysis page block
        
    # ... (rest of the page elif blocks) ...

    # --- NEW: Function to initialize FinRobot Agent ---
    @st.cache_resource # Cache the agent resource itself
    def initialize_finrobot_agent():
        """Initializes and returns a FinRobot agent configured with OAI_CONFIG_LIST."""
        try:
            llm_config = {
                "config_list": autogen.config_list_from_json(
                    "OAI_CONFIG_LIST", # Assumes file is in the root directory
                    filter_dict={"model": ["gpt-4-0125-preview"]}, # Or whichever model you have in the list
                ),
                "timeout": 120,
                "temperature": 0.2, # Lower temperature for more factual answers
            }
            # Create a basic assistant agent
            # You might want to customize the system message later
            assistant_agent = SingleAssistant(
                name="Portfolio_Analyst_Assistant",
                llm_config=llm_config,
                system_message="You are a helpful AI assistant specialized in analyzing portfolio performance data. Answer the user's questions based on the provided portfolio summary. Be concise and clear.",
                human_input_mode="NEVER", # Agent runs without asking for human input during its process
            )
            return assistant_agent
        except FileNotFoundError:
            st.error("Error: OAI_CONFIG_LIST file not found. Please ensure it exists in the project root.")
            return None
        except Exception as e:
            st.error(f"Error initializing FinRobot agent: {e}")
            traceback.print_exc() # Print detailed traceback to console/log
            return None
    # --- End Function to initialize FinRobot Agent ---

    # --- NEW: Function to format analysis results for LLM ---
    def format_portfolio_data_for_llm(analysis_results):
        """Formats the portfolio analysis results into a string for the LLM agent."""
        if not analysis_results or not isinstance(analysis_results, dict):
            return "Нет данных для анализа или неверный формат."

        summary_parts = []

        # Extract metrics (raw numeric data)
        metrics_df = analysis_results.get('metrics') # Use the raw metrics
        if metrics_df is not None and isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            summary_parts.append("**Основные метрики производительности по стратегиям:**")
            for strategy in metrics_df.index: # Iterate through strategies (index)
                summary_parts.append(f"\n*Стратегия: {strategy}*" )
                for metric, value in metrics_df.loc[strategy].items(): # Iterate through metrics for the strategy
                    if isinstance(value, (int, float)):
                        # Use original metric names from calculation for formatting clues
                        if any(p in metric.lower() for p in ['cagr', 'return']):
                            formatted_value = f"{value:.2%}"
                        elif any(p in metric.lower() for p in ['volatility', 'drawdown']):
                             formatted_value = f"{value:.2%}"
                        elif any(r in metric.lower() for r in ['ratio']):
                             formatted_value = f"{value:.2f}"
                        else: # Default for Final Value, Net Profit etc.
                             formatted_value = f"{value:,.2f}"
                             if 'value' in metric.lower() or 'profit' in metric.lower():
                                  formatted_value = f"${formatted_value}" # Add dollar sign
                    else:
                         formatted_value = str(value)
                    summary_parts.append(f"  - {metric}: {formatted_value}")
            summary_parts.append("\n") 
        else:
            summary_parts.append("Метрики производительности недоступны.")

        # Extract date range and final values from daily values DataFrame
        daily_values_df = analysis_results.get('portfolio_daily_values')
        if daily_values_df is not None and isinstance(daily_values_df, pd.DataFrame) and not daily_values_df.empty:
            start_date = daily_values_df.index.min().strftime('%Y-%m-%d')
            end_date = daily_values_df.index.max().strftime('%Y-%m-%d')
            summary_parts.append(f"**Период анализа:** {start_date} - {end_date}\n")

            summary_parts.append("**Финальная стоимость портфеля по стратегиям:**")
            final_values = daily_values_df.iloc[-1] # Get last row
            # Filter only strategy value columns (usually start with 'Value_')
            strategy_value_cols = [col for col in daily_values_df.columns if col.startswith('Value_')]
            for strategy_col in strategy_value_cols:
                # Try to map column name back to display name if possible (e.g., from metrics index)
                strategy_name = strategy_col.replace('Value_', '').replace('_', ' ') # Basic name cleanup
                if metrics_df is not None and not metrics_df.empty:
                     matching_names = [idx for idx in metrics_df.index if strategy_col.endswith(idx.replace(' ','_').replace('DRL ',''))]
                     if matching_names: strategy_name = matching_names[0]
                
                value = final_values.get(strategy_col, np.nan)
                if pd.notna(value):
                     summary_parts.append(f"  - {strategy_name}: ${value:,.2f}")
            summary_parts.append("\n")
        else:
            summary_parts.append("Данные о ежедневной стоимости портфеля недоступны.")

        # <<< REMOVED holdings formatting logic >>>

        return "\n".join(summary_parts)
    # --- End Function to format analysis results for LLM ---

    # --- HELPER FUNCTIONS (Moved to top after imports) ---

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

    '''
    poetry run streamlit run auth_app.py
    '''