@echo off
echo 激活pytorch环境...
call conda activate pytorch

echo 安装依赖包...
pip install -r requirements.txt

echo 开始运行随机森林模型...
python rf_model.py

echo 运行完成！
pause
