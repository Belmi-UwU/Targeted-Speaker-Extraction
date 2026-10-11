#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")" && pwd)"
exec "$task_root/speakerbeam_ss_verification/.venv/bin/python" -u "$task_root/local_web/server.py" "$@"
