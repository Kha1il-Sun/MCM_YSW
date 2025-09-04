@echo off
echo ========================================
echo 数据预处理脚本
echo ========================================
echo.

echo 激活pytorch环境...
call conda activate pytorch

echo.
echo 检查环境...
python -c "import torch; print('PyTorch版本:', torch.__version__)"

echo.
echo 开始运行数据预处理...
python run_preprocessing.py

echo.
echo 按任意键退出...
pause
