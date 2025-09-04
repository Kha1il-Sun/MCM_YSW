@echo off
echo 激活pytorch环境并运行特征工程XGBoost模型...
echo.

REM 激活conda环境
call conda activate pytorch

REM 检查环境是否激活成功
if errorlevel 1 (
    echo 错误：无法激活pytorch环境，请检查conda安装和环境名称
    pause
    exit /b 1
)

echo 环境激活成功，开始运行代码...
echo.

REM 运行Python脚本
python feature_engineering_xgboost.py

echo.
echo 运行完成！
pause
