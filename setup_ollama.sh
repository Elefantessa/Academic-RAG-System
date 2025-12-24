
set -euo pipefail


export OLLAMA_DIR="/project_antwerp/.ollama"
export OLLAMA_MODELS="$OLLAMA_DIR/models"
export OLLAMA_HOST="127.0.0.1:11434"
export LOG_DIR="/project_antwerp/logs"
export BIN_DIR="/usr/bin"
mkdir -p "$OLLAMA_MODELS" "$LOG_DIR"

if [ -f /project_antwerp/venv/bin/activate ]; then
  source /project_antwerp/venv/bin/activate
fi


if ! command -v ollama >/dev/null 2>&1; then
  echo "[+] Installing ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi


mkdir -p "$OLLAMA_DIR"
if [ -d "/root/.ollama" ] && [ ! -L "/root/.ollama" ]; then

  rsync -a --remove-source-files /root/.ollama/ "$OLLAMA_DIR"/ || true
  rm -rf /root/.ollama
fi
ln -sfn "$OLLAMA_DIR" /root/.ollama

echo "[+] Starting ollama server..."
pkill -f "ollama serve" >/dev/null 2>&1 || true
nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
  "$BIN_DIR/ollama" serve > "$LOG_DIR/ollama_server.log" 2>&1 &

echo -n "[+] Waiting for ollama to be ready"
for i in {1..30}; do
  sleep 1
  if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo " ... OK"
    break
  fi
  echo -n "."
  if [ $i -eq 30 ]; then
    echo "  (timeout)"
    exit 1
  fi
done

echo "[+] Pulling model: llama3.1"
env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull llama3.1


echo "[+] Verifying storage location:"
echo "OLLAMA_MODELS=$OLLAMA_MODELS"
du -sh "$OLLAMA_DIR" "$OLLAMA_MODELS" || true
echo "[+] Done. Try:"
echo "env OLLAMA_HOST=$OLLAMA_HOST OLLAMA_MODELS=$OLLAMA_MODELS ollama run llama3.1 -p 'Hello'"
