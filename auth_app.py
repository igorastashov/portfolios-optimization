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

# Импорт модулей приложения
from portfolios_optimization.data_loader import load_price_data, load_return_data, load_model_actions
from portfolios_optimization.portfolio_optimizer import optimize_markowitz_portfolio
from portfolios_optimization.portfolio_analysis import calculate_metrics, plot_efficient_frontier
from portfolios_optimization.visualization import plot_portfolio_performance, plot_asset_allocation
from portfolios_optimization.model_trainer import train_model, load_trained_model
from portfolios_optimization.authentication import (
    initialize_users_file, register_user, authenticate_user, get_user_info,
    update_user_portfolio, get_user_portfolios, get_user_portfolio
)

# Импорт страниц приложения
from app_pages import (
    render_dashboard, render_portfolio_optimization, render_model_training, 
    render_model_comparison, render_backtest, render_about, render_account_dashboard
)

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
                    st.session_state.active_page = "Dashboard"
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
    page = st.sidebar.radio(
        "Выберите раздел",
        ["Мой кабинет", "Единый торговый аккаунт", "Dashboard", "Portfolio Optimization", "Model Training", "Model Comparison", "Backtest Results", "About"]
    )
    
    st.session_state.active_page = page
    
    # Страница личного кабинета пользователя
    if page == "Мой кабинет":
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
            
            # Получение списка портфелей пользователя
            portfolios = get_user_portfolios(st.session_state.username)
            
            if portfolios:
                # Отображение существующих портфелей
                tab1, tab2 = st.tabs(["Мои портфели", "Создать новый портфель"])
                
                with tab1:
                    # Выбор портфеля для просмотра
                    selected_portfolio = st.selectbox(
                        "Выберите портфель",
                        options=portfolios
                    )
                    
                    # Загрузка данных выбранного портфеля
                    portfolio_data = get_user_portfolio(st.session_state.username, selected_portfolio)
                    
                    if portfolio_data:
                        st.write(f"**Название:** {selected_portfolio}")
                        st.write(f"**Описание:** {portfolio_data.get('description', 'Нет описания')}")
                        st.write(f"**Тип:** {portfolio_data.get('type', 'Пользовательский')}")
                        st.write(f"**Последнее обновление:** {portfolio_data.get('last_updated', 'Неизвестно')}")
                        
                        # Отображение активов портфеля
                        if "assets" in portfolio_data:
                            # Создаем DataFrame для отображения
                            portfolio_df = pd.DataFrame({
                                'Актив': list(portfolio_data['assets'].keys()),
                                'Вес': list(portfolio_data['assets'].values())
                            })
                            
                            # График распределения активов
                            fig = px.pie(
                                portfolio_df,
                                values='Вес',
                                names='Актив',
                                title=f"Распределение активов портфеля {selected_portfolio}"
                            )
                            st.plotly_chart(fig)
                            
                            # Таблица с активами
                            st.table(portfolio_df)
                            
                            # Анализ портфеля
                            st.subheader("Анализ портфеля")
                            
                            # Если доступны данные цен активов
                            if not price_data.empty:
                                # Фильтрация данных для активов портфеля
                                portfolio_assets = list(portfolio_data['assets'].keys())
                                if all(asset in price_data.columns for asset in portfolio_assets):
                                    # Получение данных цен
                                    asset_data = price_data[portfolio_assets]
                                    
                                    # Расчет доходностей
                                    returns = asset_data.pct_change().dropna()
                                    
                                    # Веса активов
                                    weights = np.array(list(portfolio_data['assets'].values()))
                                    
                                    # Расчет доходности портфеля
                                    portfolio_returns = np.matmul(returns, weights)
                                    portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
                                    
                                    # График накопленной доходности
                                    cum_returns = portfolio_returns_series.cumsum()
                                    
                                    fig = px.line(
                                        cum_returns, 
                                        title="Накопленная доходность портфеля",
                                        labels={"value": "Доходность", "index": "Дата"}
                                    )
                                    st.plotly_chart(fig)
                                    
                                    # Метрики портфеля
                                    total_return = cum_returns.iloc[-1]
                                    annual_return = (1 + total_return) ** (252 / len(cum_returns)) - 1
                                    volatility = portfolio_returns_series.std() * np.sqrt(252)
                                    sharpe_ratio = annual_return / volatility
                                    
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Общая доходность", f"{total_return*100:.2f}%")
                                    col2.metric("Годовая доходность", f"{annual_return*100:.2f}%")
                                    col3.metric("Коэффициент Шарпа", f"{sharpe_ratio:.2f}")
                                else:
                                    st.warning("Некоторые активы портфеля отсутствуют в данных цен")
                            else:
                                st.warning("Данные цен активов недоступны")
                    else:
                        st.error("Не удалось загрузить данные портфеля")
                
                with tab2:
                    # Форма создания нового портфеля
                    st.subheader("Создать новый портфель")
                    
                    with st.form("new_portfolio_form"):
                        portfolio_name = st.text_input("Название портфеля")
                        portfolio_description = st.text_area("Описание портфеля")
                        
                        # Выбор активов
                        selected_assets = st.multiselect(
                            "Выберите активы",
                            options=assets,
                            default=assets[:5] if len(assets) >= 5 else assets
                        )
                        
                        # Динамическое создание слайдеров для весов активов
                        asset_weights = {}
                        
                        if selected_assets:
                            # Равномерное начальное распределение
                            initial_weight = 1.0 / len(selected_assets)
                            
                            for asset in selected_assets:
                                weight = st.slider(
                                    f"Вес {asset}",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=initial_weight,
                                    step=0.01,
                                    key=f"weight_{asset}"
                                )
                                asset_weights[asset] = weight
                            
                            # Проверка суммы весов
                            total_weight = sum(asset_weights.values())
                            st.write(f"Общая сумма весов: {total_weight:.2f}")
                            
                            if abs(total_weight - 1.0) > 0.01:
                                st.warning("Сумма весов должна быть равна 1.0")
                        
                        submit_button = st.form_submit_button("Создать портфель")
                        
                        if submit_button:
                            if not portfolio_name:
                                st.error("Необходимо указать название портфеля")
                            elif not selected_assets:
                                st.error("Необходимо выбрать хотя бы один актив")
                            elif abs(total_weight - 1.0) > 0.01:
                                st.error("Сумма весов должна быть равна 1.0")
                            else:
                                # Создание данных портфеля
                                portfolio_data = {
                                    "description": portfolio_description,
                                    "type": "custom",
                                    "assets": asset_weights,
                                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # Сохранение портфеля
                                success, message = update_user_portfolio(
                                    st.session_state.username, 
                                    portfolio_name, 
                                    portfolio_data
                                )
                                
                                if success:
                                    st.success(message)
                                    st.info("Обновите страницу, чтобы увидеть новый портфель")
                                else:
                                    st.error(message)
            else:
                st.info("У вас еще нет портфелей. Создайте свой первый портфель!")
                
                # Форма создания нового портфеля
                with st.form("first_portfolio_form"):
                    portfolio_name = st.text_input("Название портфеля")
                    portfolio_description = st.text_area("Описание портфеля")
                    
                    # Выбор активов
                    selected_assets = st.multiselect(
                        "Выберите активы",
                        options=assets,
                        default=assets[:5] if len(assets) >= 5 else assets
                    )
                    
                    # Динамическое создание слайдеров для весов активов
                    asset_weights = {}
                    
                    if selected_assets:
                        # Равномерное начальное распределение
                        initial_weight = 1.0 / len(selected_assets)
                        
                        for asset in selected_assets:
                            weight = st.slider(
                                f"Вес {asset}",
                                min_value=0.0,
                                max_value=1.0,
                                value=initial_weight,
                                step=0.01,
                                key=f"weight_{asset}"
                            )
                            asset_weights[asset] = weight
                        
                        # Проверка суммы весов
                        total_weight = sum(asset_weights.values())
                        st.write(f"Общая сумма весов: {total_weight:.2f}")
                        
                        if abs(total_weight - 1.0) > 0.01:
                            st.warning("Сумма весов должна быть равна 1.0")
                    
                    submit_button = st.form_submit_button("Создать портфель")
                    
                    if submit_button:
                        if not portfolio_name:
                            st.error("Необходимо указать название портфеля")
                        elif not selected_assets:
                            st.error("Необходимо выбрать хотя бы один актив")
                        elif abs(total_weight - 1.0) > 0.01:
                            st.error("Сумма весов должна быть равна 1.0")
                        else:
                            # Создание данных портфеля
                            portfolio_data = {
                                "description": portfolio_description,
                                "type": "custom",
                                "assets": asset_weights,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Сохранение портфеля
                            success, message = update_user_portfolio(
                                st.session_state.username, 
                                portfolio_name, 
                                portfolio_data
                            )
                            
                            if success:
                                st.success(message)
                                st.info("Обновите страницу, чтобы увидеть свой портфель")
                            else:
                                st.error(message)
        else:
            st.error("Не удалось загрузить информацию о пользователе")
    
    # Страница единого торгового аккаунта в стиле Bybit
    elif page == "Единый торговый аккаунт":
        render_account_dashboard(st.session_state.username, price_data, assets)
    
    # Подключение страниц из app_pages.py
    elif page == "Dashboard":
        render_dashboard(price_data, model_returns, model_actions, assets)
    
    elif page == "Portfolio Optimization":
        render_portfolio_optimization(price_data, assets)
    
    elif page == "Model Training":
        render_model_training(price_data, assets)
    
    elif page == "Model Comparison":
        render_model_comparison(model_returns, model_actions)
    
    elif page == "Backtest Results":
        render_backtest(model_returns)
    
    elif page == "About":
        render_about() 