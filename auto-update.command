#!/bin/bash
# F1 Auto-Update — double-click to run, or use with launchd/cron
cd "$(dirname "$0")"
source venv/bin/activate
python -m data.auto_update "$@"
