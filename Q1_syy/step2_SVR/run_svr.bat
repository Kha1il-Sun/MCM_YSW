@echo off
echo 激活pytorch环境...
call conda activate pytorch

echo 安装依赖包...
pip install -r requirements.txt

echo 开始运行SVR模型训练...
python svr_model.py

echo 训练完成！
pause
