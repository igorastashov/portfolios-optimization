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
    update_user_portfolio, get_user_portfolios, get_user_portfolio, get_portfolio_with_quantities
)

# Импорт страниц приложения
from app_pages import (
    render_dashboard, render_portfolio_optimization, render_model_training, 
    render_model_comparison, render_backtest, render_about, render_account_dashboard,
    render_transactions_manager
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
    page_options = ["Мой кабинет", "Управление активами", "Единый торговый аккаунт", "Dashboard", "Portfolio Optimization", "Model Training", "Model Comparison", "Backtest Results", "About"]
    
    # Устанавливаем индекс для radio на основе текущей активной страницы в состоянии сессии
    try:
        current_page_index = page_options.index(st.session_state.active_page)
    except ValueError:
        current_page_index = 0 # По умолчанию первая страница, если значение некорректно
        st.session_state.active_page = page_options[0]

    selected_page = st.sidebar.radio(
        "Выберите раздел",
        page_options,
        index=current_page_index, # Инициализируем radio текущей страницей из session_state
        key="main_nav_radio" # Добавляем ключ для стабильности
    )
    
    # Обновляем состояние сессии, ТОЛЬКО если пользователь выбрал ДРУГУЮ страницу в radio
    if selected_page != st.session_state.active_page:
        st.session_state.active_page = selected_page
        st.rerun() # Перезапускаем, чтобы обновить страницу немедленно

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
        st.header("Единый торговый аккаунт - Динамика стоимости портфеля")

        # --- Portfolio Visualization Code Start ---
        from collections import OrderedDict # Ensure OrderedDict is imported

        # --- Настройки ---
        days_history = 180
        # Укажите "сегодняшнюю" дату для отсчета истории назад
        # TODO: Consider making this dynamic or configurable via st.date_input
        report_date = pd.Timestamp('2025-04-19')

        # --- Шаг 1: Определение данных о портфеле ---
        # Используем данные из вашего последнего запроса
        portfolio_data_viz = {
            "ID": [2, 0, 1, 3],
            # ВАЖНО: Убедитесь, что эти даты соответствуют вашим реальным покупкам для этих сумм!
            "Дата": ["2025-02-09T14:21:24.000", "2025-04-01T14:21:01.000", "2025-03-05T14:21:17.000", "2025-01-12T14:29:48.000"],
            "Актив": ["LTCUSDT", "BNBUSDT", "BTCUSDT", "HBARUSDT"],
            "Общая стоимость": [10000.00, 1000.00, 1000.00, 784.00] # Суммы из вашего примера
        }
        portfolio_df_viz = pd.DataFrame(portfolio_data_viz)
        portfolio_df_viz['Дата'] = pd.to_datetime(portfolio_df_viz['Дата'])
        portfolio_df_viz = portfolio_df_viz.sort_values(by='Дата').reset_index(drop=True)

        # --- Шаг 2: Загрузка и обработка исторических данных активов ---
        # Используйте @st.cache_data для кеширования загрузки и предобработки данных
        @st.cache_data(ttl=3600) # Кешировать данные на 1 час
        def load_and_preprocess_historical_data(assets_list):
            csv_base_path = 'D:\\__projects__\\diploma\\portfolios-optimization\\data' # Путь к вашим данным
            all_prices = {}
            data_found = False
            for asset in assets_list:
                file_path = os.path.join(csv_base_path, f'{asset}_hourly_data.csv')
                try:
                    df = pd.read_csv(file_path)
                    # Простая предобработка (адаптируйте при необходимости)
                    if 'Open time' not in df.columns:
                        # Поиск столбца времени
                        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
                        if not time_col: raise ValueError(f"Missing time column for {asset}")
                        df.rename(columns={time_col: 'Open time'}, inplace=True)
                    if 'Close' not in df.columns:
                         # Поиск столбца цены
                        price_col = next((col for col in df.columns if 'close' in col.lower()), None)
                        if not price_col: raise ValueError(f"Missing close price column for {asset}")
                        df.rename(columns={price_col: 'Close'}, inplace=True)

                    df['Open time'] = pd.to_datetime(df['Open time'])
                    df = df.set_index('Open time')
                    df = df[['Close']].rename(columns={'Close': f'{asset}_Price'})
                    df[f'{asset}_Price'] = df[f'{asset}_Price'].astype(float)
                    all_prices[asset] = df
                    data_found = True
                except FileNotFoundError:
                    st.warning(f"Файл данных для {asset} не найден по пути: {file_path}")
                except Exception as e:
                    st.error(f"Ошибка при обработке файла для {asset}: {e}")
            
            if not data_found:
                return pd.DataFrame() # Возвращаем пустой DataFrame, если не нашли ни одного файла

            # Объединение данных
            historical_data = pd.concat(all_prices.values(), axis=1)
            return historical_data

        required_assets = portfolio_df_viz['Актив'].unique().tolist()
        historical_prices_viz = load_and_preprocess_historical_data(required_assets)

        if historical_prices_viz.empty:
            st.error("Не удалось загрузить исторические данные для активов в портфеле. График не может быть построен.")
            st.stop() # Прерываем выполнение для этой страницы

        # Определяем фактическую дату отчета, если 'today' не задана
        if report_date is None:
             if not historical_prices_viz.empty:
                 report_date = historical_prices_viz.index.max()
             else:
                  st.error("Не удалось определить дату отчета, так как нет исторических данных.")
                  st.stop()

        start_date_history = report_date - pd.Timedelta(days=days_history)

        # --- Шаг 3: Фильтрация, подготовка и расчет стоимости ---
        historical_prices_filtered_viz = historical_prices_viz[
            (historical_prices_viz.index >= start_date_history) &
            (historical_prices_viz.index <= report_date)
        ].copy()

        # Заполнение пропусков
        historical_prices_filtered_viz = historical_prices_filtered_viz.ffill().bfill()

        # Проверка на NaN после заполнения
        if historical_prices_filtered_viz.isnull().values.any():
            st.warning("В исторических данных остались пропуски (NaN) после заполнения. Строки с NaN будут удалены.")
            # st.dataframe(historical_prices_filtered_viz[historical_prices_filtered_viz.isnull().any(axis=1)]) # Отладка
            historical_prices_filtered_viz = historical_prices_filtered_viz.dropna()

        if historical_prices_filtered_viz.empty:
            st.error(f"Нет исторических данных в диапазоне {start_date_history} - {report_date} после обработки. График не может быть построен.")
            st.stop()

        # Поиск цен покупки
        portfolio_df_viz['Purchase_Price_Actual'] = np.nan
        portfolio_df_viz['Actual_Purchase_Time_Index'] = pd.NaT

        st.write("Поиск цен на момент покупки...")
        for index, row in portfolio_df_viz.iterrows():
            asset = row['Актив']
            purchase_date = row['Дата']
            price_col = f'{asset}_Price'

            if price_col not in historical_prices_filtered_viz.columns:
                st.warning(f"Столбец цен {price_col} не найден для актива {asset} в отфильтрованных данных.")
                continue

            relevant_prices = historical_prices_filtered_viz[historical_prices_filtered_viz.index >= purchase_date]

            if not relevant_prices.empty:
                actual_purchase_time_index = relevant_prices.index[0]
                purchase_price = historical_prices_filtered_viz.loc[actual_purchase_time_index, price_col]

                if pd.notna(purchase_price) and purchase_price > 0:
                    portfolio_df_viz.loc[index, 'Purchase_Price_Actual'] = purchase_price
                    portfolio_df_viz.loc[index, 'Actual_Purchase_Time_Index'] = actual_purchase_time_index
                else:
                    st.warning(f"Найдена некорректная цена ({purchase_price}) для {asset} на {actual_purchase_time_index}. Покупка будет проигнорирована.")
                    portfolio_df_viz.loc[index, 'Actual_Purchase_Time_Index'] = pd.NaT
            else:
                st.warning(f"Не найдены исторические данные для {asset} на или после {purchase_date} в диапазоне до {report_date}. Покупка будет проигнорирована.")
                portfolio_df_viz.loc[index, 'Actual_Purchase_Time_Index'] = pd.NaT

        # Удаление недействительных покупок
        initial_portfolio_size_viz = len(portfolio_df_viz)
        portfolio_df_viz.dropna(subset=['Actual_Purchase_Time_Index', 'Purchase_Price_Actual'], inplace=True)
        final_portfolio_size_viz = len(portfolio_df_viz)

        st.write(f"Начальное кол-во покупок: {initial_portfolio_size_viz}, учтено после поиска цен: {final_portfolio_size_viz}")
        if final_portfolio_size_viz == 0:
              st.error("Не найдено действительных покупок для анализа. Проверьте Даты покупок и наличие данных в CSV.")
              st.stop()

        # Расчет относительной стоимости
        historical_prices_filtered_viz['Total_Value_Relative'] = 0.0
        st.write("Расчет относительной стоимости портфеля...")

        progress_bar = st.progress(0)
        total_steps = len(historical_prices_filtered_viz.index)

        for i, current_time_index in enumerate(historical_prices_filtered_viz.index):
            current_total_value = 0.0
            for _, purchase_row in portfolio_df_viz.iterrows():
                purchase_time_index = purchase_row['Actual_Purchase_Time_Index']
                if current_time_index >= purchase_time_index:
                    initial_investment = purchase_row['Общая стоимость']
                    purchase_price = purchase_row['Purchase_Price_Actual']
                    asset = purchase_row['Актив']
                    price_col = f'{asset}_Price'
                    current_price = historical_prices_filtered_viz.loc[current_time_index, price_col]

                    if pd.notna(current_price) and current_price > 0:
                        price_ratio = current_price / purchase_price
                        current_investment_value = initial_investment * price_ratio
                    else:
                        current_investment_value = 0 # Обнуляем вклад, если цена некорректна

                    current_total_value += current_investment_value

            historical_prices_filtered_viz.loc[current_time_index, 'Total_Value_Relative'] = current_total_value
            progress_bar.progress((i + 1) / total_steps) # Обновляем прогресс-бар

        st.write("Расчет завершен.")
        progress_bar.empty() # Убираем прогресс-бар

        # --- Шаг 4: Визуализация в Streamlit ---
        st.subheader("График динамики стоимости")
        fig_viz, ax_viz = plt.subplots(figsize=(12, 6)) # Уменьшил размер для Streamlit
        plt.style.use('seaborn-v0_8-darkgrid')

        # График стоимости
        ax_viz.plot(historical_prices_filtered_viz.index, historical_prices_filtered_viz['Total_Value_Relative'],
                label='Общая стоимость портфеля (Метод относит. изменений)', color='green', linewidth=2)

        # Отметки о покупках
        unique_labels_legend_viz = set()
        portfolio_in_range_viz = portfolio_df_viz[portfolio_df_viz['Actual_Purchase_Time_Index'] >= historical_prices_filtered_viz.index.min()]

        for _, row in portfolio_in_range_viz.iterrows():
            plot_time = row['Actual_Purchase_Time_Index']
            if plot_time in historical_prices_filtered_viz.index:
                value_at_purchase = historical_prices_filtered_viz.loc[plot_time, 'Total_Value_Relative']
                label_text_marker = f'Покупка {row["Актив"]}'
                current_label_for_legend = label_text_marker if label_text_marker not in unique_labels_legend_viz else ""
                if current_label_for_legend: unique_labels_legend_viz.add(label_text_marker)

                ax_viz.scatter(plot_time, value_at_purchase, color='red', s=50, zorder=5,
                           label=current_label_for_legend, marker='o', edgecolors='black')

                ax_viz.annotate(f" +${row['Общая стоимость']:,.2f}\n ({row['Актив']})",
                            xy=(plot_time, value_at_purchase), xytext=(10, 10),
                            textcoords='offset points', ha='left', va='bottom', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='grey'))

        # Настройка графика
        ax_viz.set_title(f'Динамика стоимости портфеля за {days_history} дней', fontsize=12)
        ax_viz.set_xlabel('Дата', fontsize=10)
        ax_viz.set_ylabel('Расчетная стоимость (USDT)', fontsize=10)
        ax_viz.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax_viz.tick_params(axis='x', rotation=30, labelsize=8)
        ax_viz.tick_params(axis='y', labelsize=8)
        ax_viz.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Легенда
        handles_viz, labels_viz = ax_viz.get_legend_handles_labels()
        if handles_viz: # Добавляем проверку, что есть что отображать в легенде
             if 'Общая стоимость портфеля (Метод относит. изменений)' not in labels_viz and ax_viz.get_lines():
                 line_handle_viz = ax_viz.get_lines()[0]
                 handles_viz.insert(0, line_handle_viz)
                 labels_viz.insert(0, line_handle_viz.get_label())
             by_label_viz = OrderedDict(zip(labels_viz, handles_viz))
             ax_viz.legend(by_label_viz.values(), by_label_viz.keys(), loc='best', fontsize=8)

        fig_viz.tight_layout()
        st.pyplot(fig_viz) # Отображаем график в Streamlit

        # --- Вывод данных для проверки (опционально) ---
        with st.expander("Показать данные для отладки"):
            st.write("Данные портфеля с найденными ценами покупки:")
            st.dataframe(portfolio_df_viz[['Дата', 'Актив', 'Общая стоимость', 'Purchase_Price_Actual', 'Actual_Purchase_Time_Index']])

            st.write(f"Рассчитанные данные портфеля (первые 5 записей):")
            price_cols_to_show_viz = sorted([f'{a}_Price' for a in portfolio_df_viz['Актив'].unique() if f'{a}_Price' in historical_prices_filtered_viz.columns])
            st.dataframe(historical_prices_filtered_viz[['Total_Value_Relative'] + price_cols_to_show_viz].head())

            st.write(f"Рассчитанные данные портфеля (последние 5 записей):")
            st.dataframe(historical_prices_filtered_viz[['Total_Value_Relative'] + price_cols_to_show_viz].tail())
        # --- Portfolio Visualization Code End ---

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