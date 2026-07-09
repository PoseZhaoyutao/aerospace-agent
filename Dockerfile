# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt setup.py ./
COPY aerospace_agent/ ./aerospace_agent/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /build/ ./
# 环境变量默认值
ENV AEROSPACE_LOG_LEVEL=INFO
ENV AEROSPACE_DATA_DIR=/app/data
ENV PYTHONPATH=/app
# 数据卷
VOLUME ["/app/data", "/app/reports"]
# 入口
ENTRYPOINT ["aerospace-agent"]
CMD ["--help"]
