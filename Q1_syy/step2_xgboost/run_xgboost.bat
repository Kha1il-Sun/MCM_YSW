@echo off
echo 正在激活pytorch环境...
call conda activate pytorch

echo 正在安装依赖包...
pip install -r requirements.txt

echo 正在运行XGBoost模型...
python run_xgboost.py

echo 程序执行完成！
pause
