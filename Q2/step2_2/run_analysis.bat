@echo off
echo 运行基于经验数据的理论模型分析...
echo.

REM 检查Python环境
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

REM 运行主程序
echo 开始分析...
python p2.py --config config/step2_config.yaml --data-dir ../step2_1

if %errorlevel% equ 0 (
    echo.
    echo 分析完成！结果已保存到 outputs/ 目录
    echo.
    echo 主要输出文件:
    echo - p2_group_recommendation.csv: BMI分组推荐
    echo - p2_wstar_curve.csv: w*(b)曲线数据
    echo - p2_report.txt: 详细分析报告
    echo - empirical_model_params.yaml: 模型参数
    echo.
) else (
    echo.
    echo 分析失败！请检查错误信息
    echo.
)

pause
