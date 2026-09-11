# Jalankan app.py (default) atau tentukan file lain: .\run.ps1 app_final.py
param(
    [string]$File = "app.py",
    [int]$Port = 8501
)

$python = "C:\Users\THINKPAD\AppData\Local\Python\pythoncore-3.14-64\python.exe"
& $python -m streamlit run $File --server.port $Port
