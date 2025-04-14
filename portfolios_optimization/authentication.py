import os
import json
import hashlib
import pandas as pd
from datetime import datetime

# Файл с данными пользователей
USERS_FILE = "data/users.json"
PORTFOLIOS_DIR = "data/user_portfolios"
TRANSACTIONS_DIR = "data/user_transactions"

def initialize_users_file():
    """Инициализация файла пользователей, если он не существует"""
    if not os.path.exists(os.path.dirname(USERS_FILE)):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(PORTFOLIOS_DIR):
        os.makedirs(PORTFOLIOS_DIR, exist_ok=True)
    
    if not os.path.exists(TRANSACTIONS_DIR):
        os.makedirs(TRANSACTIONS_DIR, exist_ok=True)

def hash_password(password):
    """Хеширование пароля"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Загрузка данных пользователей"""
    initialize_users_file()
    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_users(users):
    """Сохранение данных пользователей"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username, password, email):
    """Регистрация нового пользователя"""
    users = load_users()
    
    # Проверка на существование пользователя
    if username in users:
        return False, "Пользователь с таким именем уже существует"
    
    # Создание новой записи пользователя
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "",
        "portfolio": {}
    }
    
    # Создание директории для портфелей пользователя
    user_portfolio_dir = os.path.join(PORTFOLIOS_DIR, username)
    os.makedirs(user_portfolio_dir, exist_ok=True)
    
    # Сохранение данных пользователей
    save_users(users)
    
    return True, "Регистрация успешно завершена"

def authenticate_user(username, password):
    """Аутентификация пользователя"""
    users = load_users()
    
    # Проверка на существование пользователя
    if username not in users:
        return False, "Неверное имя пользователя"
    
    # Проверка пароля
    if users[username]["password"] != hash_password(password):
        return False, "Неверный пароль"
    
    # Обновление времени последнего входа
    users[username]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_users(users)
    
    return True, "Вход выполнен успешно"

def get_user_info(username):
    """Получение информации о пользователе"""
    users = load_users()
    
    if username not in users:
        return None
    
    # Копируем информацию без пароля
    user_info = users[username].copy()
    user_info.pop("password", None)
    
    return user_info

def update_user_portfolio(username, portfolio_name, portfolio_data):
    """Обновление портфеля пользователя"""
    users = load_users()
    
    if username not in users:
        return False, "Пользователь не найден"
    
    # Обновление информации о портфеле в данных пользователя
    if "portfolios" not in users[username]:
        users[username]["portfolios"] = {}
    
    users[username]["portfolios"][portfolio_name] = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": portfolio_data.get("description", ""),
        "type": portfolio_data.get("type", "custom")
    }
    
    save_users(users)
    
    # Сохранение данных портфеля в отдельный файл
    portfolio_file = os.path.join(PORTFOLIOS_DIR, username, f"{portfolio_name}.json")
    with open(portfolio_file, 'w') as f:
        json.dump(portfolio_data, f, indent=2)
    
    return True, "Портфель успешно обновлен"

def get_user_portfolios(username):
    """Получение списка портфелей пользователя"""
    users = load_users()
    
    if username not in users:
        return []
    
    # Получение списка портфелей из данных пользователя
    if "portfolios" not in users[username]:
        return []
    
    return list(users[username]["portfolios"].keys())

def get_user_portfolio(username, portfolio_name):
    """Получение данных портфеля пользователя"""
    portfolio_file = os.path.join(PORTFOLIOS_DIR, username, f"{portfolio_name}.json")
    
    if not os.path.exists(portfolio_file):
        return None
    
    with open(portfolio_file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def add_transaction(username, transaction_data):
    """Добавление новой транзакции для пользователя
    
    Args:
        username (str): Имя пользователя
        transaction_data (dict): Данные транзакции в формате:
            {
                "asset": "BTCUSDT",
                "type": "buy" or "sell",
                "quantity": float,
                "price": float,
                "fee": float,
                "date": "YYYY-MM-DD HH:MM:SS",
                "note": "Описание транзакции"
            }
    
    Returns:
        tuple: (success, message)
    """
    if not transaction_data.get("asset") or not transaction_data.get("type") or not transaction_data.get("quantity"):
        return False, "Необходимо указать актив, тип операции и количество"
    
    if transaction_data["type"] not in ["buy", "sell"]:
        return False, "Тип операции должен быть 'buy' или 'sell'"
    
    # Получение списка транзакций пользователя
    transactions = get_user_transactions(username)
    
    # Добавление ID и даты транзакции
    transaction_id = len(transactions) + 1
    transaction_data["id"] = transaction_id
    if not transaction_data.get("date"):
        transaction_data["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Добавление новой транзакции
    transactions.append(transaction_data)
    
    # Сохранение транзакций
    user_transactions_dir = os.path.join(TRANSACTIONS_DIR, username)
    os.makedirs(user_transactions_dir, exist_ok=True)
    
    transactions_file = os.path.join(user_transactions_dir, "transactions.json")
    with open(transactions_file, 'w') as f:
        json.dump(transactions, f, indent=2)
    
    # Обновление портфеля пользователя на основе транзакций
    update_portfolio_from_transactions(username)
    
    return True, f"Транзакция №{transaction_id} успешно добавлена"

def get_user_transactions(username):
    """Получение всех транзакций пользователя
    
    Args:
        username (str): Имя пользователя
    
    Returns:
        list: Список транзакций
    """
    user_transactions_dir = os.path.join(TRANSACTIONS_DIR, username)
    os.makedirs(user_transactions_dir, exist_ok=True)
    
    transactions_file = os.path.join(user_transactions_dir, "transactions.json")
    
    if not os.path.exists(transactions_file):
        return []
    
    with open(transactions_file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def get_transaction_by_id(username, transaction_id):
    """Получение транзакции по ID
    
    Args:
        username (str): Имя пользователя
        transaction_id (int): ID транзакции
    
    Returns:
        dict: Данные транзакции или None, если не найдена
    """
    transactions = get_user_transactions(username)
    for transaction in transactions:
        if transaction.get("id") == transaction_id:
            return transaction
    return None

def delete_transaction(username, transaction_id):
    """Удаление транзакции по ID
    
    Args:
        username (str): Имя пользователя
        transaction_id (int): ID транзакции
    
    Returns:
        tuple: (success, message)
    """
    transactions = get_user_transactions(username)
    
    # Фильтрация транзакций, исключая удаляемую
    updated_transactions = [t for t in transactions if t.get("id") != transaction_id]
    
    if len(updated_transactions) == len(transactions):
        return False, f"Транзакция с ID {transaction_id} не найдена"
    
    # Сохранение обновленных транзакций
    user_transactions_dir = os.path.join(TRANSACTIONS_DIR, username)
    transactions_file = os.path.join(user_transactions_dir, "transactions.json")
    with open(transactions_file, 'w') as f:
        json.dump(updated_transactions, f, indent=2)
    
    # Обновление портфеля пользователя на основе транзакций
    update_portfolio_from_transactions(username)
    
    return True, f"Транзакция с ID {transaction_id} успешно удалена"

def update_portfolio_from_transactions(username):
    """Обновление портфеля пользователя на основе его транзакций
    
    Args:
        username (str): Имя пользователя
    
    Returns:
        dict: Обновленный портфель
    """
    transactions = get_user_transactions(username)
    
    # Словарь для хранения баланса активов
    portfolio = {}
    
    # Словарь для хранения средней цены покупки
    avg_prices = {}
    
    # Словарь для общей стоимости активов
    total_values = {}
    
    # Обработка всех транзакций
    for transaction in transactions:
        asset = transaction.get("asset")
        quantity = float(transaction.get("quantity", 0))
        price = float(transaction.get("price", 0))
        fee = float(transaction.get("fee", 0))
        
        if transaction.get("type") == "buy":
            # Для покупки: увеличиваем количество и обновляем среднюю цену
            current_quantity = portfolio.get(asset, 0)
            current_value = total_values.get(asset, 0)
            
            # Новое количество и общая стоимость
            new_quantity = current_quantity + quantity
            transaction_value = quantity * price
            new_value = current_value + transaction_value
            
            # Обновление портфеля
            portfolio[asset] = new_quantity
            total_values[asset] = new_value
            
            # Обновление средней цены покупки
            if new_quantity > 0:
                avg_prices[asset] = new_value / new_quantity
        
        elif transaction.get("type") == "sell":
            # Для продажи: уменьшаем количество
            current_quantity = portfolio.get(asset, 0)
            
            # Новое количество
            new_quantity = max(0, current_quantity - quantity)  # Не позволяет уйти в минус
            
            # Если продаем всё, сбрасываем среднюю цену
            if new_quantity == 0:
                avg_prices[asset] = 0
                total_values[asset] = 0
            
            # Обновление портфеля
            portfolio[asset] = new_quantity
    
    # Создание данных для сохранения
    portfolio_data = {
        "description": "Портфель на основе транзакций",
        "type": "transactions",
        "assets": {},
        "avg_prices": avg_prices,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Расчет весов активов
    total_balance = sum(portfolio.get(asset, 0) * avg_prices.get(asset, 0) for asset in portfolio)
    
    if total_balance > 0:
        for asset in portfolio:
            if portfolio.get(asset, 0) > 0:
                asset_value = portfolio.get(asset, 0) * avg_prices.get(asset, 0)
                weight = asset_value / total_balance
                portfolio_data["assets"][asset] = weight
    
    # Сохранение портфеля
    update_user_portfolio(username, "transactions_portfolio", portfolio_data)
    
    return portfolio_data

def get_portfolio_with_quantities(username):
    """Получение портфеля с количествами активов
    
    Args:
        username (str): Имя пользователя
    
    Returns:
        dict: Портфель с количествами активов
    """
    transactions = get_user_transactions(username)
    
    # Словарь для хранения баланса активов
    portfolio = {}
    
    # Словарь для хранения средней цены покупки
    avg_prices = {}
    
    # Словарь для общей стоимости активов
    total_values = {}
    
    # Обработка всех транзакций
    for transaction in transactions:
        asset = transaction.get("asset")
        quantity = float(transaction.get("quantity", 0))
        price = float(transaction.get("price", 0))
        fee = float(transaction.get("fee", 0))
        
        if transaction.get("type") == "buy":
            # Для покупки: увеличиваем количество и обновляем среднюю цену
            current_quantity = portfolio.get(asset, 0)
            current_value = total_values.get(asset, 0)
            
            # Новое количество и общая стоимость
            new_quantity = current_quantity + quantity
            transaction_value = quantity * price
            new_value = current_value + transaction_value
            
            # Обновление портфеля
            portfolio[asset] = new_quantity
            total_values[asset] = new_value
            
            # Обновление средней цены покупки
            if new_quantity > 0:
                avg_prices[asset] = new_value / new_quantity
        
        elif transaction.get("type") == "sell":
            # Для продажи: уменьшаем количество
            current_quantity = portfolio.get(asset, 0)
            
            # Новое количество
            new_quantity = max(0, current_quantity - quantity)  # Не позволяет уйти в минус
            
            # Если продаем всё, сбрасываем среднюю цену
            if new_quantity == 0:
                avg_prices[asset] = 0
                total_values[asset] = 0
            
            # Обновление портфеля
            portfolio[asset] = new_quantity
    
    return {
        "quantities": portfolio,
        "avg_prices": avg_prices,
        "total_values": total_values
    } 