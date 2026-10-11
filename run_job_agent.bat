@echo off
cd /d "%~dp0"
python job_agent.py >> run_log.txt 2>&1
