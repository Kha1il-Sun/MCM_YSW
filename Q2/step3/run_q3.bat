@echo off
REM 问题3：综合多因素的NIPT最佳时点优化
REM Windows 批处理运行脚本

echo ============================================================
echo 问题3：综合多因素的NIPT最佳时点优化
echo ============================================================

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：Python未安装或不在PATH中
    echo 请安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 检查是否在正确目录
if not exist "p3.py" (
    echo 错误：请在step3目录中运行此脚本
    echo 当前目录应包含p3.py文件
    pause
    exit /b 1
)

REM 检查配置文件
if not exist "config\step3_config.yaml" (
    echo 错误：配置文件不存在
    echo 请确保config\step3_config.yaml文件存在
    pause
    exit /b 1
)

REM 检查数据文件
if not exist "..\step2_1\step1_long_records.csv" (
    echo 错误：数据文件不存在
    echo 请先运行step2_1中的数据预处理
    pause
    exit /b 1
)

echo 检查完成，开始运行分析...
echo.

REM 运行分析
python run_analysis.py %*

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo 分析完成！请查看outputs目录中的结果文件
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo 分析过程中出现错误，请查看错误信息
    echo ============================================================
)

pause
