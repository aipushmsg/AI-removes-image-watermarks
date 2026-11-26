#!/usr/bin/env bash
# 停止 start.sh 启动的两个服务
set -uo pipefail

cd "$(dirname "$0")"

for name in api iopaint; do
  pid_file="logs/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" && echo "已停止 ${name} (pid ${pid})"
    else
      echo "${name} 未在运行"
    fi
    rm -f "${pid_file}"
  else
    echo "${name} 未找到 pid 文件"
  fi
done
