#!/usr/bin/env bash
# 启动 IOPaint(8080) + 去水印 API(8000)
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
mkdir -p models logs

# 优先用项目内的 .venv，其次用环境变量 PYTHON，最后回落到系统 python3
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
echo "使用解释器：$("$PY" -V 2>&1)"

IOPAINT_PORT="${IOPAINT_PORT:-8080}"
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${IOPAINT_PORT}/api/v1/model"; then
  echo "IOPaint 已在 ${IOPAINT_PORT} 端口运行，跳过启动"
else
  nohup "$PY" -m iopaint start --model=lama --device=cpu \
    --port "${IOPAINT_PORT}" --model-dir "${ROOT}/models" >> logs/iopaint.log 2>&1 &
  echo $! > logs/iopaint.pid
  echo "IOPaint 启动中（首次运行需下载约 200MB 模型）..."
  for _ in $(seq 1 90); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${IOPAINT_PORT}/api/v1/model"; then
      echo "IOPaint 就绪"
      break
    fi
    sleep 2
  done
fi

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${API_PORT}/health"; then
  echo "API 已在 ${API_PORT} 端口运行，跳过启动"
else
  cd api
  IOPAINT_URL="http://127.0.0.1:${IOPAINT_PORT}" nohup "$PY" -m uvicorn main:app \
    --host "${API_HOST}" --port "${API_PORT}" >> ../logs/api.log 2>&1 &
  echo $! > ../logs/api.pid
  cd ..
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${API_PORT}/health"; then
      echo "API 就绪：http://${API_HOST}:${API_PORT}"
      break
    fi
    sleep 1
  done
fi

curl -s "http://127.0.0.1:${API_PORT}/health" && echo
