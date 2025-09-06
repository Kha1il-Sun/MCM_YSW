@echo off
echo 开始检测误差影响分析...
echo.

REM 设置Python路径
set PYTHONPATH=%CD%\src;%PYTHONPATH%

REM 运行分析
python run_error_analysis.py --data-dir ../../step2_1 --config ../../config/step2_config.yaml

echo.
echo 分析完成！请查看 outputs 目录中的结果文件。
pause
