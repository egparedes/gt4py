#!/usr/bin/env python3
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fake OpenAI/Anthropic upstream for the AWF probe; runs on the HOST.

It records every request the API-proxy sidecar forwards (method, path, header
names, and which header carries the dummy key) as JSON lines, and returns
minimal well-formed responses so the proxy's accounting works. The dummy key is
read from the environment and only ever reported as "<DUMMY>".

Usage: python3 -I echo_server.py <port> <log.jsonl>   (env: PROBE_ECHO_KEY)
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


KEY = os.environ.get("PROBE_ECHO_KEY", "")
LOG = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"
MODELS = [
    "zai-org/GLM-5.3-Flash",
    "zai-org/GLM-5.3",
    "Qwen/Qwen3.8-Flash-Next-FP8",
    "swiss-ai/Apertus-70B-Instruct",
]


def classify(value: str) -> dict:
    # Split "Scheme credential" so log masking of "Bearer ..." cannot hide it.
    parts = value.split(" ", 1)
    scheme, cred = (parts[0], parts[1]) if len(parts) == 2 else ("", value)
    if KEY and cred == KEY:
        cred = "<DUMMY>"
    elif KEY and KEY in cred:
        cred = cred.replace(KEY, "<DUMMY>")
    elif len(cred) > 60:
        cred = cred[:20] + "...(" + str(len(cred)) + " chars)"
    else:
        cred = "client:" + cred.replace("-", "_")
    return {"scheme": scheme or None, "credential": cred}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence default stderr logging
        pass

    def _record(self, body: bytes) -> dict:
        model = None
        try:
            model = json.loads(body or b"{}").get("model")
        except Exception:
            pass
        auth = {
            k.lower(): (v if k.lower() == "anthropic-version" else classify(v))
            for k, v in self.headers.items()
            if k.lower() in ("authorization", "x-api-key", "api-key", "anthropic-version")
            or (KEY and KEY in v)
        }
        rec = {
            "t": time.time(),
            "method": self.command,
            "path": self.path,
            "model": model,
            "auth": auth,
            "header_names": sorted(k.lower() for k in self.headers.keys()),
        }
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def _send(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        rec = self._record(b"")
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
        else:
            self._send({"echo": {"path": rec["path"], "auth": rec["auth"]}})

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        rec = self._record(body)
        echo = json.dumps({"path": rec["path"], "auth": rec["auth"]})
        if self.path.split("?")[0].endswith("/messages"):
            self._send(
                {
                    "id": "msg_probe",
                    "type": "message",
                    "role": "assistant",
                    "model": rec["model"],
                    "content": [{"type": "text", "text": echo}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                }
            )
        else:
            self._send(
                {
                    "id": "chatcmpl-probe",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": rec["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": echo},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                }
            )


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
