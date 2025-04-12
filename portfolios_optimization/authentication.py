import os
import json
import hashlib
import pandas as pd
from datetime import datetime

# Файл с данными пользователей
USERS_FILE = "data/users.json"
PORTFOLIOS_DIR = "data/user_portfolios"

def initialize_users_file():
    """Инициализация файла пользователей, если он не существует"""
    if not os.path.exists(os.path.dirname(USERS_FILE)):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(PORTFOLIOS_DIR):
        os.makedirs(PORTFOLIOS_DIR, exist_ok=True)

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