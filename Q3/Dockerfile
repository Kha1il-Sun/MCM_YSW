FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt pyproject.toml ./

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 复制源代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# 创建必要的目录
RUN mkdir -p data outputs logs .cache

# 设置权限
RUN chmod +x scripts/*.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.config; print('OK')" || exit 1

# 默认命令
CMD ["python", "scripts/run_pipeline.py", "--config", "configs/q3_default.yaml"]