@echo off
echo 运行 Problem 2 分析...
echo.

REM 检查Python环境
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 安装依赖
echo 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告: 依赖包安装可能有问题，继续运行...
)

REM 运行基本测试
echo.
echo 运行基本功能测试...
python test_basic.py
if %errorlevel% neq 0 (
    echo 错误: 基本功能测试失败
    pause
    exit /b 1
)

REM 运行主分析
echo.
echo 运行主分析...
python p2.py
if %errorlevel% neq 0 (
    echo 错误: 主分析失败
    pause
    exit /b 1
)

echo.
echo 分析完成！请查看 outputs 目录中的结果文件。
pause
