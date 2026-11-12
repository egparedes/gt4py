#!/usr/bin/env bash
# Runs INSIDE the AWF sandbox. Exercises the API-proxy listeners against CSCS
# with the real key held by the sidecar. Prints only HTTP status, model id,
# token usage and truncated response text. Never prints env values.
# Every probe is non-fatal.

OAI="${AWF_API_PROXY_OPENAI_URL:-http://172.30.0.30:10000}"
ANT="${AWF_API_PROXY_ANTHROPIC_URL:-http://172.30.0.30:10001}"
TMP="$(mktemp -d)"
CANDIDATES="${PROBE_MODEL_CANDIDATES:-zai-org/GLM-5.3-Flash Qwen/Qwen3.8-Flash-Next-FP8}"

say() { printf 'PROBE %s\n' "$*"; }

summ() {  # summ <label> <status> <body-file>
  python3 -I - "$1" "$2" "$3" <<'PY'
import json, sys
label, status, path = sys.argv[1:4]
raw = open(path, "rb").read()
try:
    d = json.loads(raw)
except Exception:
    print(f"PROBE {label}: status={status} non-json body[:200]={raw[:200]!r}")
    sys.exit()
info = {"status": status}
if isinstance(d, dict):
    info["model"] = d.get("model")
    info["usage"] = d.get("usage")
    if "choices" in d:
        m = (d["choices"] or [{}])[0].get("message", {}) or {}
        info["finish_reason"] = (d["choices"] or [{}])[0].get("finish_reason")
        info["text"] = (m.get("content") or "")[:120]
        tc = m.get("tool_calls") or []
        info["tool_calls"] = [
            {"name": t.get("function", {}).get("name"),
             "arguments": (t.get("function", {}).get("arguments") or "")[:120]}
            for t in tc
        ]
        if m.get("reasoning_content") is not None or m.get("reasoning") is not None:
            info["has_reasoning"] = True
    elif d.get("type") == "message":
        info["stop_reason"] = d.get("stop_reason")
        info["content"] = [
            {"type": c.get("type"), "text": (c.get("text") or "")[:120],
             "name": c.get("name"), "input": c.get("input")}
            for c in d.get("content", [])
        ]
    elif "error" in d or "detail" in d or "message" in d:
        info["error"] = json.dumps(d)[:300]
    else:
        info["keys"] = sorted(d)[:20]
print(f"PROBE {label}: " + json.dumps(info))
PY
}

req() {  # req <label> <method> <url> [data] [extra curl args...]
  local label="$1" method="$2" url="$3" data="${4:-}"
  shift 4 || shift $#
  local out="$TMP/$label.json" status
  if [ -n "$data" ]; then
    status=$(curl -sS -m 120 -o "$out" -w '%{http_code}' -X "$method" "$url" \
      -H 'content-type: application/json' "$@" --data "$data" 2>"$TMP/$label.err")
  else
    status=$(curl -sS -m 60 -o "$out" -w '%{http_code}' -X "$method" "$url" "$@" 2>"$TMP/$label.err")
  fi
  [ -s "$TMP/$label.err" ] && say "$label: curl-stderr=$(head -c 200 "$TMP/$label.err")"
  summ "$label" "$status" "$out"
}

say "env names (values not shown): $(env | cut -d= -f1 | grep -E 'API|BASE_URL|AWF_|TOKEN|KEY' | sort | tr '\n' ' ')"
say "listener urls: openai=$OAI anthropic=$ANT"

curl -sS -m 10 "$OAI/health" -o "$TMP/health.json" -w '' 2>/dev/null
say "health: $(head -c 600 "$TMP/health.json" | tr '\n' ' ')"

curl -sS -m 30 "$OAI/reflect" -o "$TMP/reflect.json" 2>/dev/null
python3 -I - "$TMP/reflect.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"PROBE reflect: unparsable ({e})"); sys.exit()
print("PROBE reflect top-level keys:", sorted(d)[:40])
print("PROBE reflect models_fetch_complete:", d.get("models_fetch_complete"))
for ep in d.get("endpoints", []) or []:
    models = ep.get("models")
    if isinstance(models, list):
        ids = [m if isinstance(m, str) else m.get("id") for m in models]
    else:
        ids = models
    print("PROBE reflect endpoint:", json.dumps({k: ep.get(k) for k in ("provider", "url", "port", "configured", "base_url", "target")}), "models:", ids)
PY

# Model list straight from the upstream, through the OpenAI listener.
for p in /v1/models /models; do
  st=$(curl -sS -m 30 -o "$TMP/models.json" -w '%{http_code}' "$OAI$p" 2>/dev/null)
  ids=$(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(sorted(m["id"] for m in d.get("data",[]))))' "$TMP/models.json" 2>/dev/null)
  say "openai-listener GET $p: status=$st ids=[$ids]"
  [ "$st" = 200 ] && cp "$TMP/models.json" "$TMP/models-ok.json"
done

MODEL=""
for c in $CANDIDATES; do
  if [ -f "$TMP/models-ok.json" ] && grep -qF "\"$c\"" "$TMP/models-ok.json"; then MODEL="$c"; break; fi
done
[ -z "$MODEL" ] && MODEL="${CANDIDATES%% *}"
say "selected model: $MODEL"

# A real CSCS model that is NOT in apiProxy.allowedModels (for the 403 check).
UNLISTED=$(python3 -I - "$TMP/models-ok.json" <<'PY' 2>/dev/null
import fnmatch, json, sys
allowed = ["zai-org/glm-5.3-flash", "qwen/*"]
ids = [m["id"] for m in json.load(open(sys.argv[1])).get("data", [])]
print(next((i for i in ids if not any(fnmatch.fnmatchcase(i.lower(), a) for a in allowed)), ""))
PY
)
say "unlisted real model for the 403 check: ${UNLISTED:-<none found>}"

CHAT=$(printf '{"model":"%s","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' "$MODEL")
req chat-v1 POST "$OAI/v1/chat/completions" "$CHAT"
req chat-nov1 POST "$OAI/chat/completions" "$CHAT"

TOOL=$(printf '{"model":"%s","max_tokens":256,"tool_choice":"auto","tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"messages":[{"role":"user","content":"What is the weather in Zurich? Call the get_weather tool."}]}' "$MODEL")
req tool-call POST "$OAI/v1/chat/completions" "$TOOL"

MSG=$(printf '{"model":"%s","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' "$MODEL")
req messages-anthropic-listener POST "$ANT/v1/messages" "$MSG" -H 'anthropic-version: 2023-06-01'
req messages-openai-listener POST "$OAI/v1/messages" "$MSG" -H 'anthropic-version: 2023-06-01'

ATOOL=$(printf '{"model":"%s","max_tokens":256,"tools":[{"name":"get_weather","description":"Get the current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"messages":[{"role":"user","content":"What is the weather in Zurich? Call the get_weather tool."}]}' "$MODEL")
req messages-tool-anthropic-listener POST "$ANT/v1/messages" "$ATOOL" -H 'anthropic-version: 2023-06-01'

# allowedModels enforcement (proxy-side, should not reach the upstream).
req denied-fake POST "$OAI/v1/chat/completions" '{"model":"not-allowed/model-x","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
if [ -n "$UNLISTED" ]; then
  req denied-real POST "$OAI/v1/chat/completions" "$(printf '{"model":"%s","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' "$UNLISTED")"
fi
req denied-anthropic POST "$ANT/v1/messages" '{"model":"not-allowed/model-x","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' -H 'anthropic-version: 2023-06-01'

say "done"
exit 0
