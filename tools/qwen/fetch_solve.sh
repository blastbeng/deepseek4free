#!/bin/bash
# usage: fetch_solve.sh <challenge-id>   (challenge dir number default 873)
ID="$1"; DIR="${2:-873}"
B="https://static-captcha-sgp.aliyuncs.com/qst/PUZZLE/online/$DIR/$ID"
curl -s -o /tmp/qwen_back.png  -w "back: %{http_code} %{size_download}\n" "$B/back.png"
curl -s -o /tmp/qwen_shadow.png -w "shdw: %{http_code} %{size_download}\n" "$B/shadow.png"
python3 - <<'PY'
import base64, json
def b64(p): return 'data:image/png;base64,' + base64.b64encode(open(p,'rb').read()).decode()
json.dump({'bg': b64('/tmp/qwen_back.png'), 'piece': b64('/tmp/qwen_shadow.png'),
           'bgRect': [450,340,300,200], 'pieceRect': [450,340,52,200]}, open('/tmp/qwen_cap.json','w'))
PY
python3 /opt/docker/compose/deepseek4free/tools/qwen/solve_fast.py
