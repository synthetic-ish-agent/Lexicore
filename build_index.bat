@echo off
setlocal
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% ingest.py --data .\data --db .\chroma_db
pause
