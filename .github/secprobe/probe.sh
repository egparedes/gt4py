#!/usr/bin/env bash
# Runs INSIDE the AWF sandbox against the host's echo upstream (dummy key).
# Usage: probe.sh <variant>
#   paths   methods, bodies and path variants; what is forwarded and counted
#   race    100 concurrent chat calls with maxTurns 20 (echo delays 3 s)
#   prod    the production limits: 100 concurrent chat calls
#   rph     120 sequential requests on uncounted paths with requestsPerHour 100
# Every probe is non-fatal; output is status codes and counters only.

VARIANT=${1:?}
OAI="${AWF_API_PROXY_OPENAI_URL:-http://172.30.0.30:10000}"
M=zai-org/GLM-5.3-Flash
TMP=$(mktemp -d)
say() { printf 'PROBE %s %s\n' "${VARIANT}" "$*"; }

# /reflect has the 'maxTurns' counter ('runs'), /health the rate limiter's.
reflect() { # label
  curl -sS -m 10 "${OAI}/reflect" -o "${TMP}/reflect.json" 2>/dev/null
  curl -sS -m 10 "${OAI}/health" -o "${TMP}/health.json" 2>/dev/null
  say "reflect[$1]: $(python3 -I - "${TMP}/reflect.json" "${TMP}/health.json" <<'PY'
import json, sys
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
r, h = load(sys.argv[1]), load(sys.argv[2])
rl = h.get("rate_limits")
if isinstance(rl, dict):
    rl = rl.get("openai", rl)
print(json.dumps({"runs": r.get("runs"), "rate_limits": rl}, sort_keys=True)[:600])
PY
)"
}

# req <label> <method> <path> <body|-> [curl args...]; prints status and a short proxy error type.
req() {
  local label=$1 method=$2 path=$3 body=$4
  shift 4
  local args=(-sS -m 30 --path-as-is -o "${TMP}/out" -w '%{http_code}' -X "${method}" "$@")
  [ "${body}" = - ] || args+=(-H 'content-type: application/json' --data-binary "${body}")
  local st
  st=$(curl "${args[@]}" "${OAI}${path}" 2>/dev/null) || st="curl-failed:${st}"
  local kind
  kind=$(python3 -I -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("non-json"); sys.exit()
e = d.get("error") if isinstance(d, dict) else None
if isinstance(e, dict):
    print("error:" + str(e.get("type") or e.get("code")))
elif isinstance(e, str):
    print("error:" + e[:60])
elif isinstance(d, dict) and "choices" in d:
    print("upstream-echo")
else:
    print("other")' "${TMP}/out" 2>/dev/null)
  say "${label}: ${method} ${path} -> ${st} ${kind}"
}

chat() { printf '{"model":"%s","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' "${1:-$M}"; }

say "listener ${OAI}"
reflect start
case "${VARIANT}" in
  paths)
    req chat POST /v1/chat/completions "$(chat)"
    reflect after-chat
    for i in 1 2 3; do req "completions-$i" POST /v1/completions "{\"model\":\"$M\",\"prompt\":\"hi\",\"max_tokens\":8}"; done
    for i in 1 2 3; do req "embeddings-$i" POST /v1/embeddings "{\"model\":\"$M\",\"input\":\"hi\"}"; done
    reflect after-completions-embeddings
    req embeddings-disallowed-model POST /v1/embeddings '{"model":"other/model","input":"hi"}'
    req chat-no-model POST /v1/chat/completions '{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
    req chat-model-number POST /v1/chat/completions '{"model":42,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
    req chat-model-array POST /v1/chat/completions '{"model":["other/model"],"messages":[]}'
    req chat-non-json POST /v1/chat/completions 'model=other/model&prompt=hi'
    req chat-disallowed POST /v1/chat/completions "$(chat other/model)"
    reflect after-bodies
    req put-chat PUT /v1/chat/completions "$(chat other/model)"
    req patch-chat PATCH /v1/chat/completions "$(chat other/model)"
    req delete-chat DELETE /v1/chat/completions -
    req get-files GET /v1/files -
    req get-batches GET /v1/batches -
    req get-models GET /v1/models -
    reflect after-methods
    req dot-dot POST '/v1/../admin' "$(chat)"
    req dot-dot-deep POST '/v1/chat/completions/../../../x' "$(chat)"
    req enc-dot-dot POST '/v1/%2e%2e/admin' "$(chat)"
    req enc-dot-dot-upper POST '/v1/%2E%2E/%2E%2E/admin' "$(chat)"
    req dot-dot-enc-slash POST '/v1/..%2fadmin' "$(chat)"
    req enc-slash POST '/v1/chat%2fcompletions' "$(chat)"
    req double-slash-lead POST '//v1/chat/completions' "$(chat)"
    req double-slash-mid POST '/v1//chat/completions' "$(chat)"
    req no-v1 POST '/chat/completions' "$(chat)"
    req root-dot-dot POST '/../../x' "$(chat)"
    req backslash POST '/v1/..\admin' "$(chat)"
    req query-dot-dot POST '/v1/chat/completions?x=/../../admin' "$(chat)"
    req chat-suffix-trick POST '/v1/x/chat/completions' "$(chat)"
    req absolute-form POST '/v1/chat/completions' "$(chat)" --request-target 'http://evil.example/v1/chat/completions'
    reflect after-paths
    ;;
  race | prod)
    for i in $(seq 1 100); do
      curl -sS -m 60 -o /dev/null -w '%{http_code}\n' -X POST -H 'content-type: application/json' \
        --data-binary "$(chat)" "${OAI}/v1/chat/completions" >"${TMP}/c.$i" 2>/dev/null &
    done
    wait
    say "100 concurrent chat calls: $(cat "${TMP}"/c.* | sort | uniq -c | tr -s ' \n' ' ')"
    reflect after-concurrent
    req chat-after POST /v1/chat/completions "$(chat)"
    req embeddings-after POST /v1/embeddings "{\"model\":\"$M\",\"input\":\"hi\"}"
    req put-after PUT /v1/chat/completions "$(chat)"
    reflect end
    ;;
  rph)
    : >"${TMP}/codes"
    for i in $(seq 1 40); do
      for p in /v1/embeddings /v1/completions /v1/x; do
        curl -sS -m 30 -o /dev/null -w "${p} %{http_code}\n" -X POST -H 'content-type: application/json' \
          --data-binary "{\"model\":\"$M\",\"input\":\"hi\"}" "${OAI}${p}" >>"${TMP}/codes" 2>/dev/null
      done
    done
    say "120 sequential uncounted requests: $(sort "${TMP}/codes" | uniq -c | tr -s ' \n' ' ')"
    reflect end
    ;;
esac
say done
