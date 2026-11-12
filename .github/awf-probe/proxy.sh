#!/usr/bin/env bash
# Runs INSIDE the AWF sandbox against the host's echo upstream (dummy key).
# Mode "turns": path mapping, auth headers, allowedModels, maxTurns.
# Mode "rpm":   rate limit, Anthropic authHeader override, agentTimeout.
# Every probe is non-fatal.

MODE="${1:-turns}"
OAI="${AWF_API_PROXY_OPENAI_URL:-http://172.30.0.30:10000}"
ANT="${AWF_API_PROXY_ANTHROPIC_URL:-http://172.30.0.30:10001}"
TMP="$(mktemp -d)"
N=0
say() { printf 'PROBE %s\n' "$*"; }

req() {  # req <label> <method> <url> <model|-> [extra curl args...]
  local label="$1" method="$2" url="$3" model="$4"; shift 4
  N=$((N + 1))
  local out="$TMP/$N.json" data="" st
  if [ "$model" != "-" ]; then
    data=$(printf '{"model":"%s","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' "$model")
    st=$(curl -sS -m 30 -o "$out" -w '%{http_code}' -X "$method" "$url" -H 'content-type: application/json' "$@" --data "$data" 2>&1)
  else
    st=$(curl -sS -m 30 -o "$out" -w '%{http_code}' -X "$method" "$url" "$@" 2>&1)
  fi
  local body
  body=$(python3 -I - "$out" <<'PY'
import json, sys
raw = open(sys.argv[1], "rb").read()
try:
    d = json.loads(raw)
except Exception:
    print("raw:", raw[:240]); sys.exit()
if isinstance(d, dict) and "choices" in d:
    print("upstream-echo:", d["choices"][0]["message"]["content"][:400])
elif isinstance(d, dict) and d.get("type") == "message":
    print("upstream-echo:", d["content"][0]["text"][:400])
elif isinstance(d, dict) and "data" in d:
    print("models:", [m.get("id") for m in d["data"]][:10])
else:
    print("proxy/other:", json.dumps(d)[:400])
PY
)
  say "#$N $label $method ${url#http://} model=$model -> $st | $body"
}

say "mode=$MODE listeners openai=$OAI anthropic=$ANT"
say "env: $(env | grep -E '^(OPENAI|ANTHROPIC)_(BASE_URL|API_KEY|AUTH_TOKEN)=' | sort | tr '\n' ' ')"
curl -sS -m 10 "$OAI/reflect" -o "$TMP/reflect.json" 2>/dev/null
say "reflect: $(head -c 2500 "$TMP/reflect.json" | tr '\n' ' ')"

if [ "$MODE" = turns ]; then
  req models-v1      GET  "$OAI/v1/models" -
  req models-nov1    GET  "$OAI/models" -
  req chat-v1-provider-scoped POST "$OAI/v1/chat/completions" zai-org/GLM-5.3-Flash
  req chat-nov1      POST "$OAI/chat/completions" zai-org/GLM-5.3-Flash
  req chat-glob-case POST "$OAI/v1/chat/completions" qwen/qwen3.8-flash-next-fp8
  req msgs-ant-v1    POST "$ANT/v1/messages" Qwen/Qwen3.8-Flash-Next-FP8 -H 'anthropic-version: 2023-06-01'
  req msgs-ant-nov1  POST "$ANT/messages" Qwen/Qwen3.8-Flash-Next-FP8 -H 'anthropic-version: 2023-06-01'
  req msgs-oai-v1    POST "$OAI/v1/messages" zai-org/GLM-5.3-Flash -H 'anthropic-version: 2023-06-01'
  req star-crosses-slash POST "$OAI/v1/chat/completions" swiss-ai/Apertus-70B-Instruct
  req deny-ant-scoped-to-openai POST "$ANT/v1/messages" zai-org/GLM-5.3-Flash -H 'anthropic-version: 2023-06-01'
  req deny-bare-slash-pattern POST "$OAI/v1/chat/completions" zai-org/GLM-5.2
  req deny-other     POST "$OAI/v1/chat/completions" other/model-x
  req deny-no-org    POST "$OAI/v1/chat/completions" GLM-5.3-Flash
  # client-supplied credentials must be stripped and replaced
  req client-auth-stripped POST "$OAI/v1/chat/completions" zai-org/GLM-5.3-Flash -H 'Authorization: Bearer client-supplied-value'
  for i in $(seq 1 10); do
    req "turn-fill-$i" POST "$OAI/v1/chat/completions" zai-org/GLM-5.3-Flash
  done
else
  req ant-authheader POST "$ANT/v1/messages" zai-org/GLM-5.3-Flash -H 'anthropic-version: 2023-06-01'
  for i in $(seq 1 6); do
    req "rpm-$i" POST "$OAI/v1/chat/completions" zai-org/GLM-5.3-Flash
  done
  say "agentTimeout test: sleeping 150 s with container.agentTimeout=1 (expect termination ~60 s)"
  start=$(date +%s)
  trap 'say "received signal after $(( $(date +%s) - start )) s"' TERM INT
  sleep 150 &
  wait $!
  say "sleep finished after $(( $(date +%s) - start )) s (agentTimeout did NOT stop us)"
fi
say "done"
exit 0
