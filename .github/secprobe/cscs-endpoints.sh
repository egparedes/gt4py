#!/usr/bin/env bash
# TRUSTED context only: no agent, no PR content. Checks which CSCS inference
# endpoints the key in $KEY can use, printing status codes and response sizes
# only, never bodies or headers. Minimal requests (max_tokens 1, GETs for
# objects that a POST would create).
set -uo pipefail
set +x
: "${KEY:?}"
BASE=https://api.inference.cscs.ch
M=zai-org/GLM-5.3-Flash

probe() { # method path [body]
  local method=$1 path=$2 body=${3:-} with without
  local args=(-sS -m 60 -o /dev/null -w '%{http_code}/%{size_download}B' -X "${method}")
  [ -z "${body}" ] || args+=(-H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' --data-binary "${body}")
  without=$(curl "${args[@]}" "${BASE}${path}" 2>/dev/null) || without="curl-error"
  # The key is passed in a header file, so it never appears in a command line.
  with=$(curl "${args[@]}" -H @"${HDR}" "${BASE}${path}" 2>/dev/null) || with="curl-error"
  printf 'CSCS %-6s %-45s with-key=%-14s without-key=%s\n' "${method}" "${path}" "${with}" "${without}"
}

HDR=$(mktemp)
chmod 600 "${HDR}"
printf 'Authorization: Bearer %s\n' "${KEY}" >"${HDR}"
unset KEY

probe GET /v1/models
probe GET "/v1/models/${M}"
probe POST /v1/chat/completions "{\"model\":\"${M}\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
probe POST /v1/completions "{\"model\":\"${M}\",\"prompt\":\"hi\",\"max_tokens\":1}"
probe POST /v1/embeddings "{\"model\":\"${M}\",\"input\":\"hi\"}"
probe POST /v1/messages "{\"model\":\"${M}\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
probe POST /v1/responses "{\"model\":\"${M}\",\"input\":\"hi\",\"max_output_tokens\":16}"
probe POST /v1/rerank "{\"model\":\"${M}\",\"query\":\"a\",\"documents\":[\"b\"]}"
probe POST /v1/score "{\"model\":\"${M}\",\"text_1\":\"a\",\"text_2\":\"b\"}"
probe POST /tokenize "{\"model\":\"${M}\",\"prompt\":\"hi\"}"
probe POST /v1/tokenize "{\"model\":\"${M}\",\"prompt\":\"hi\"}"
probe POST /v1/audio/transcriptions "{\"model\":\"${M}\"}"
probe POST /v1/moderations "{\"model\":\"${M}\",\"input\":\"hi\"}"
probe GET /v1/files
probe GET /v1/batches
probe GET /v1/fine_tuning/jobs
probe GET /v1/assistants
probe GET /v1/vector_stores
probe GET /v1/usage
probe GET /v1/key/info
probe GET /key/info
probe GET /v1/user/info
probe GET /user/info
probe GET /v1/dashboard/billing/usage
probe GET /v1/organization/usage/completions
probe GET /v1/me
probe GET /health
probe GET /version
probe GET /metrics
probe GET /v1
probe GET /
rm -f "${HDR}"
