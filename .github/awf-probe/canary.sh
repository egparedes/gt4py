#!/usr/bin/env bash
# Runs INSIDE the AWF sandbox with a DUMMY key in the sidecar.
# Searches everything the agent can reach for the dummy value and runs escape
# probes. The dummy value itself is never given to this script: it gets only
# its SHA-256 (PROBE_KEY_SHA256) and searches for the shape
# AWFCANARY-k<32 hex>. The regex below cannot match itself.
# A decoy with prefix AWFCANARY-d is planted on purpose as a positive control.
# Every probe is non-fatal and prints one "PROBE <name>: ..." line.

say() { printf 'PROBE %s\n' "$*"; }
OUT="${PROBE_OUT:-/tmp/awf-probe-out}"
mkdir -p "$OUT" 2>/dev/null
TS="${PROBE_TS:?}"
KEY_RE='AWFCANARY[-]k[0-9a-f]{32}'
DECOY_RE='AWFCANARY[-]d[0-9a-f]{32}'

say "id: $(id)"
say "caps: $(grep -E '^Cap(Inh|Prm|Eff|Bnd|Amb)' /proc/self/status | tr '\t\n' '  ')"
say "no_new_privs: $(grep NoNewPrivs /proc/self/status | tr '\t' ' ')"
say "seccomp: $(grep -E '^Seccomp' /proc/self/status | tr '\t\n' '  ')"
say "HOME=$HOME PWD=$PWD"
say "env names: $(env | cut -d= -f1 | sort | tr '\n' ' ')"
say "env api-related (values): $(env | grep -E '^(OPENAI|ANTHROPIC|CODEX|AWF_API|HTTPS?_PROXY|NO_PROXY|CLAUDE)' | sed -E "s/$KEY_RE/<CANARY!>/g" | sort | tr '\n' ' ')"
cp /proc/self/mountinfo "$OUT/mountinfo.txt" 2>/dev/null
say "mountinfo (mount point, options, source):"
awk '{print "PROBE   mnt", $5, $6, $(NF-2), $(NF-1)}' /proc/self/mountinfo

# --------------------------------------------------------------------------
# 1. Canary search
# --------------------------------------------------------------------------
hits_file="$OUT/candidates.txt"; : > "$hits_file"
hash_check() {  # reads candidate strings on stdin, prints matching ones
  while read -r c; do
    [ -z "$c" ] && continue
    h=$(printf '%s' "$c" | sha256sum | cut -d' ' -f1)
    if [ "$h" = "$PROBE_KEY_SHA256" ]; then echo "KEY"; fi
  done
}

# 1a. /proc/*/environ and cmdline of every visible process
proc_hits=0; proc_decoy=0; nproc=0
for p in /proc/[0-9]*; do
  nproc=$((nproc + 1))
  for f in environ cmdline; do
    c=$(tr '\0' '\n' < "$p/$f" 2>/dev/null | grep -aoE "$KEY_RE" | hash_check | wc -l)
    proc_hits=$((proc_hits + c))
    d=$(tr '\0' '\n' < "$p/$f" 2>/dev/null | grep -acE "$DECOY_RE")
    proc_decoy=$((proc_decoy + d))
  done
done
say "canary /proc/*/{environ,cmdline}: processes=$nproc key_hits=$proc_hits decoy_hits=$proc_decoy"
say "processes visible: $(ps -eo pid,user,comm 2>/dev/null | tail -n +2 | tr '\n' ';' | head -c 1500)"

# 1b. full grep of the small writable / volatile trees
scan_tree() {  # scan_tree <label> <paths...>
  local label="$1"; shift
  local key=0 decoy=0 files
  key=$(grep -rIaohE "$KEY_RE" "$@" 2>/dev/null | hash_check | wc -l)
  # -I skips binaries; repeat with -a for binary files
  key=$((key + $(grep -raohE "$KEY_RE" "$@" 2>/dev/null | hash_check | wc -l)))
  files=$(grep -rlaE "$DECOY_RE" "$@" 2>/dev/null | head -5 | tr '\n' ' ')
  decoy=$(grep -rlaE "$DECOY_RE" "$@" 2>/dev/null | wc -l)
  say "canary scan[$label] $*: key_hits=$key decoy_files=$decoy ($files)"
  if [ "$key" -gt 0 ]; then
    say "CANARY FOUND in $label: files: $(grep -rlaE "$KEY_RE" "$@" 2>/dev/null | head -20 | tr '\n' ' ')"
  fi
}
scan_tree home "$HOME"
scan_tree tmp /tmp /var/tmp /dev/shm /run
scan_tree etc /etc
[ -n "$PROBE_WORKSPACE" ] && scan_tree workspace "$PROBE_WORKSPACE"
[ -n "$PROBE_WORK_DIR" ] && scan_tree workdir "$PROBE_WORK_DIR"

# 1c. every file under / that was created or modified after the key was generated
find / \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
  -type f \( -newermt "@$TS" -o -newerct "@$TS" \) -print 2>/dev/null > "$OUT/new-files.txt"
say "files newer than key generation: $(wc -l < "$OUT/new-files.txt")"
unreadable=0; key_new=0; decoy_new=0
while IFS= read -r f; do
  if [ ! -r "$f" ]; then unreadable=$((unreadable + 1)); echo "$f" >> "$OUT/new-unreadable.txt"; continue; fi
  k=$(grep -aohE "$KEY_RE" "$f" 2>/dev/null | hash_check | wc -l)
  if [ "$k" -gt 0 ]; then key_new=$((key_new + k)); say "CANARY FOUND in new file $f"; fi
  grep -qaE "$DECOY_RE" "$f" 2>/dev/null && decoy_new=$((decoy_new + 1))
done < "$OUT/new-files.txt"
say "canary scan[new files under /]: key_hits=$key_new decoy_files=$decoy_new unreadable=$unreadable"
[ -s "$OUT/new-unreadable.txt" ] && say "unreadable new files: $(head -20 "$OUT/new-unreadable.txt" | tr '\n' ' ')"
say "new files (first 60): $(grep -vE '^/(tmp/awf-probe-out)' "$OUT/new-files.txt" | head -60 | tr '\n' ' ')"

# 1d. the API proxy's own endpoints
for path in /health /reflect /metrics /debug /config; do
  b=$(curl -sS -m 10 "${AWF_API_PROXY_OPENAI_URL:-http://172.30.0.30:10000}$path" 2>/dev/null)
  k=$(printf '%s' "$b" | grep -aoE "$KEY_RE" | hash_check | wc -l)
  say "canary api-proxy GET $path: bytes=${#b} key_hits=$k"
done

# --------------------------------------------------------------------------
# 2. Privilege and host-escape probes
# --------------------------------------------------------------------------
sudo -n true >/dev/null 2>&1; say "sudo -n true: exit=$? (expect non-zero)"
say "sudo binary: $(ls -la "$(command -v sudo 2>/dev/null)" 2>&1)"
for s in /var/run/docker.sock /run/docker.sock /run/containerd/containerd.sock; do
  say "socket $s: $(ls -la "$s" 2>&1 | head -1); test -S: $([ -S "$s" ] && echo yes || echo no)"
done
r=$(curl -sS -m 5 --unix-socket /var/run/docker.sock http://localhost/version 2>&1 | head -c 200)
say "docker API via socket: $r"

# Runner internals (paths passed in by the host; never print file contents)
for v in PROBE_RUNNER_TEMP PROBE_GITHUB_ENV PROBE_GITHUB_OUTPUT PROBE_GITHUB_PATH PROBE_RUNNER_DIR PROBE_HOST_HOME; do
  p="${!v}"; [ -z "$p" ] && continue
  if [ -d "$p" ]; then
    say "$v=$p: dir listing: $(ls -la "$p" 2>&1 | head -15 | tr '\n' ';')"
  else
    say "$v=$p: exists=$([ -e "$p" ] && echo y || echo n) readable=$([ -r "$p" ] && echo y || echo n) writable=$([ -w "$p" ] && echo y || echo n)"
  fi
done
if [ -n "$PROBE_GITHUB_ENV" ] && [ -w "$PROBE_GITHUB_ENV" ]; then
  echo "AWF_PROBE_INJECTED=from-sandbox" >> "$PROBE_GITHUB_ENV" && say "GITHUB_ENV append: SUCCEEDED (host should see AWF_PROBE_INJECTED)"
else
  say "GITHUB_ENV append: not possible"
fi
if [ -n "$PROBE_RUNNER_DIR" ]; then
  for f in .credentials .credentials_rsaparams .runner; do
    say "runner file $PROBE_RUNNER_DIR/$f: exists=$([ -e "$PROBE_RUNNER_DIR/$f" ] && echo y || echo n) readable=$([ -r "$PROBE_RUNNER_DIR/$f" ] && echo y || echo n)"
  done
fi

# Write tests: a marker in each candidate path; the host checks persistence.
MARK=".awf-probe-write-marker"
for d in "$PROBE_WORKSPACE" /tmp /var/tmp /dev/shm "$HOME" "$HOME/.config" "$HOME/.config/git" \
         "$HOME/.cache" "$HOME/.local/bin" "$HOME/.local/share" "$HOME/.npm" "$HOME/.cargo" \
         "$HOME/.claude" "$HOME/.copilot" "$HOME/.ssh" /opt /opt/awf-probe /usr/local/bin /etc /root \
         "$PROBE_RUNNER_TEMP" "$PROBE_WORK_DIR" "$PROBE_HOST_HOME"; do
  [ -z "$d" ] && continue
  mkdir -p "$d" 2>/dev/null
  if ( echo probe > "$d/$MARK" ) 2>/dev/null; then say "write $d: WRITABLE"; else say "write $d: denied"; fi
done

# --------------------------------------------------------------------------
# 3. Network probes
# --------------------------------------------------------------------------
code() { curl -sS -m 8 -o /dev/null -w '%{http_code}' "$@" 2>&1 | tail -c 160 | tr '\n' ' '; }
say "net via proxy github.com (allowed): $(code https://github.com)"
say "net via proxy example.com (not allowed): $(code https://example.com)"
say "net via proxy api.inference.cscs.ch: $(code https://api.inference.cscs.ch/v1/models)"
say "net direct (no proxy) example.com:443: $(env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy curl -sS -m 8 --noproxy '*' -o /dev/null -w '%{http_code}' https://example.com 2>&1 | tail -c 160)"
say "net direct raw IP 1.1.1.1:443: $(env -u HTTPS_PROXY -u https_proxy curl -sS -m 8 --noproxy '*' -o /dev/null -w '%{http_code}' https://1.1.1.1 2>&1 | tail -c 160)"
say "net direct raw IP 1.1.1.1:80: $(env -u HTTP_PROXY -u http_proxy curl -sS -m 8 --noproxy '*' -o /dev/null -w '%{http_code}' http://1.1.1.1 2>&1 | tail -c 160)"
say "net direct tcp 140.82.112.3:22: $(timeout 6 bash -c '</dev/tcp/140.82.112.3/22' 2>&1 && echo CONNECTED || echo failed)"
say "net direct tcp 8.8.8.8:853: $(timeout 6 bash -c '</dev/tcp/8.8.8.8/853' 2>&1 && echo CONNECTED || echo failed)"
say "net direct cscs: $(env -u HTTPS_PROXY -u https_proxy curl -sS -m 8 --noproxy '*' -o /dev/null -w '%{http_code}' https://api.inference.cscs.ch/v1/models 2>&1 | tail -c 160)"
say "net IMDS 169.254.169.254 direct: $(curl -sS -m 5 --noproxy '*' -H Metadata:true -o /dev/null -w '%{http_code}' 'http://169.254.169.254/metadata/instance?api-version=2021-02-01' 2>&1 | tail -c 160)"
say "net IMDS 169.254.169.254 via proxy: $(code -H Metadata:true 'http://169.254.169.254/metadata/instance?api-version=2021-02-01')"
say "net wireserver 168.63.129.16 direct: $(curl -sS -m 5 --noproxy '*' -H 'x-ms-version: 2012-11-30' -o /dev/null -w '%{http_code}' 'http://168.63.129.16/machine/?comp=goalstate' 2>&1 | tail -c 160)"
say "net wireserver 168.63.129.16 via proxy: $(code -H 'x-ms-version: 2012-11-30' 'http://168.63.129.16/machine/?comp=goalstate')"
say "net host gateway 172.30.0.1:18080 direct: $(curl -sS -m 5 --noproxy '*' -o /dev/null -w '%{http_code}' http://172.30.0.1:18080/ 2>&1 | tail -c 160)"
say "net host gateway 172.30.0.1:18080 via proxy: $(code http://172.30.0.1:18080/)"
say "net docker tcp 172.30.0.1:2375: $(timeout 5 bash -c '</dev/tcp/172.30.0.1/2375' 2>&1 && echo CONNECTED || echo failed)"
say "net dns (resolver) github.com: $(getent hosts github.com | head -1)"
say "net dns direct to 1.1.1.1: $(python3 -I -c '
import socket,struct
q=struct.pack(">HHHHHH",0x1234,0x0100,1,0,0,0)+b"".join(bytes([len(p)])+p.encode() for p in "example.com".split("."))+b"\0"+struct.pack(">HH",1,1)
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(4)
try:
    s.sendto(q,("1.1.1.1",53)); d=s.recv(512); print("ANSWERED",len(d),"bytes")
except Exception as e: print("failed:",e)
' 2>&1)"
say "resolv.conf: $(grep -v '^#' /etc/resolv.conf 2>/dev/null | tr '\n' ';')"

# --------------------------------------------------------------------------
# 4. Work-dir visibility
# --------------------------------------------------------------------------
say "/tmp listing: $(ls -la /tmp 2>&1 | tail -n +2 | awk '{print $1, $3, $NF}' | tr '\n' ';')"
for d in /tmp/awf-* "$PROBE_WORK_DIR"; do
  [ -e "$d" ] || continue
  say "workdir candidate $d: $(ls -la "$d" 2>&1 | tail -n +2 | awk '{print $1, $NF}' | tr '\n' ';' | head -c 800)"
done
say "done"
exit 0
