# MCM Q3 项目 Makefile

.PHONY: help install test lint format clean run docker-build docker-run

# 默认目标
.DEFAULT_GOAL := help

# 项目变量
PROJECT_NAME := mcm-q3
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := $(PROJECT_NAME):latest

help: ## 显示帮助信息
	@echo "MCM Q3 项目 - 可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## 安装开发依赖
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

test: ## 运行测试
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit: ## 运行单元测试
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## 运行集成测试
	$(PYTHON) -m pytest tests/integration/ -v

lint: ## 代码检查
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format: ## 代码格式化
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean: ## 清理临时文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf outputs/ logs/ .cache/

run: ## 运行主管道
	$(PYTHON) scripts/run_pipeline.py --config configs/q3_default.yaml

run-sigma-sweep: ## 运行sigma敏感性分析
	$(PYTHON) scripts/run_sigma_sweep.py --config configs/mi_sigma_sweep.yaml

run-grouping: ## 运行分组分析
	$(PYTHON) scripts/run_grouping_dp.py

docker-build: ## 构建Docker镜像
	docker build -t $(DOCKER_IMAGE) .

docker-run: ## 运行Docker容器
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE)

docker-shell: ## 进入Docker容器shell
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/logs:/app/logs \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE)

setup-data: ## 设置示例数据
	mkdir -p data/schemas
	@echo "请将数据文件放置在 data/ 目录中"
	@echo "- step1_long_records.csv"
	@echo "- step1_surv_dat_fit.csv (可选)"

validate-config: ## 验证配置文件
	$(PYTHON) -c "from src.config import load_config; load_config('configs/q3_default.yaml'); print('配置文件有效')"

profile: ## 性能分析
	$(PYTHON) -m cProfile -o profile.stats scripts/run_pipeline.py --config configs/q3_default.yaml
	$(PYTHON) -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

memory-profile: ## 内存分析
	$(PYTHON) -m memory_profiler scripts/run_pipeline.py --config configs/q3_default.yaml

docs: ## 生成文档
	sphinx-build -b html docs/ docs/_build/

serve-docs: ## 启动文档服务器
	$(PYTHON) -m http.server 8000 --directory docs/_build/

check-deps: ## 检查依赖安全性
	$(PIP) check
	safety check

update-deps: ## 更新依赖
	$(PIP) list --outdated
	@echo "运行 'pip install --upgrade <package>' 来更新特定包"

init-project: install setup-data validate-config ## 初始化项目
	@echo "项目初始化完成！"
	@echo "运行 'make run' 开始分析"

# 开发工作流
dev-setup: install-dev ## 设置开发环境
	@echo "开发环境设置完成"

dev-test: format lint test ## 开发测试流程

dev-ci: dev-test ## CI流程
	@echo "CI检查通过"

# 发布相关
build: clean ## 构建分发包
	$(PYTHON) -m build

upload-test: build ## 上传到测试PyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

upload: build ## 上传到PyPI
	$(PYTHON) -m twine upload dist/*

# 监控和调试
monitor: ## 监控运行状态
	watch -n 5 'ps aux | grep python | grep -v grep'

logs: ## 查看日志
	tail -f logs/mcm_q3.log

debug: ## 调试模式运行
	$(PYTHON) -m pdb scripts/run_pipeline.py --config configs/q3_default.yaml