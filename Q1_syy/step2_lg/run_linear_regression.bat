@echo off
echo 正在激活pytorch环境并运行线性回归模型...
echo.

REM 激活conda环境
call conda activate pytorch

REM 检查环境是否激活成功
if errorlevel 1 (
    echo 错误：无法激活pytorch环境，请检查conda是否正确安装
    pause
    exit /b 1
)

echo pytorch环境已激活
echo.

REM 运行线性回归模型
echo 开始运行线性回归模型...
python linear_regression_model.py

REM 检查运行结果
if errorlevel 1 (
    echo 错误：程序运行失败
    pause
    exit /b 1
) else (
    echo.
    echo 程序运行完成！
    echo 请查看生成的结果文件和图表
)

echo.
pause
