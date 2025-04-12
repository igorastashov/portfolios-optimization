@echo off

:: Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Run the Streamlit application with authentication
streamlit run auth_app.py 