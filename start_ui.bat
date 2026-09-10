@echo off
rem 启动 APF-SFC MuJoCo 抑振实验 UI（双击本文件即可）
rem 用项目自带 venv 的解释器，pythonw 不弹黑窗；日志都进 UI 文本框。
rem 若 venv 路径与本机不同，改下面 PY 一行即可。
set "PY=C:\Users\PC\Desktop\code\code\Myproject-1\venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=python"
start "" "%PY%" "%~dp0launcher_ui.py"
