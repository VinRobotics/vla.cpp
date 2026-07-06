#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Shared helpers for the CI runners. Per-arch GGUF/env resolution mirrors
# eval/run_libero.sh and eval/run_libero_server.sh so the CI and the manual
# sweeps stay in lock-step. The mem sampler is the same Tegra-aware sampler as
# eval/run_libero_server.sh, plus a macOS variant (no /proc, no nvidia-smi).

# ---- bind address normalisation --------------------------------------------
# "5555" | "*:5555" | "host:5555" | "tcp://..." -> a full tcp:// URL.
normalize_bind() {
    local a="$1"
    case "$a" in
        tcp://*)  echo "$a" ;;
        *:*)      echo "tcp://$a" ;;
        *[!0-9]*) echo "ERROR: cannot parse bind '$a'" >&2; return 1 ;;
        *)        echo "tcp://*:$a" ;;
    esac
}

# ---- platform -> connection vars (orchestrator side) -----------------------
# Resolve a platform key to its hosts.env settings, setting the globals RROOT /
# RMODELS / SRV_HOST / SRV_PORT / CTRL_PORT / CMAKE_FLAGS / BUILD_ENV. Sourced
# from ci/config/hosts.env (orchestrator only). Used by run_remote.sh,
# check_commits.sh, build_servers.sh.
resolve_platform() {
    case "$1" in
        rtx3090) RROOT="${RTX_REPO_ROOT}"; RMODELS="${RTX_MODELS_ROOT}"; SRV_HOST="${RTX_SERVER_HOST}"
                 SRV_PORT="${RTX_DATA_PORT}"; CTRL_PORT="${RTX_CTRL_PORT}"; CMAKE_FLAGS="${RTX_CMAKE_FLAGS:-}"
                 BUILD_ENV="${RTX_BUILD_ENV:-}" ;;
        orin)    RROOT="${ORIN_REPO_ROOT}"; RMODELS="${ORIN_MODELS_ROOT}"; SRV_HOST="${ORIN_SERVER_HOST}"
                 SRV_PORT="${ORIN_DATA_PORT}"; CTRL_PORT="${ORIN_CTRL_PORT}"; CMAKE_FLAGS="${ORIN_CMAKE_FLAGS:-}"
                 BUILD_ENV="${ORIN_BUILD_ENV:-}" ;;
        m4)      RROOT="${M4_REPO_ROOT}"; RMODELS="${M4_MODELS_ROOT}"; SRV_HOST="${M4_SERVER_HOST}"
                 SRV_PORT="${M4_DATA_PORT}"; CTRL_PORT="${M4_CTRL_PORT}"; CMAKE_FLAGS="${M4_CMAKE_FLAGS:-}"
                 BUILD_ENV="${M4_BUILD_ENV:-}" ;;
        *) echo "ERROR: unknown platform '$1'" >&2; return 1 ;;
    esac
}

# ---- per-suite model layout -------------------------------------------------
# Most archs ship ONE suite-agnostic checkpoint (a flat <repo>/<file>.gguf dir).
# bitvla and gr00t_n1_7 instead partition weights per LIBERO suite, one GGUF +
# dataset_statistics.json under <repo>/<suite-subdir>/. Those two MUST be served
# per-suite (the runner respawns vla-server for each suite they cover).
is_per_suite_model() { case "$1" in bitvla|gr00t_n1_7) return 0 ;; *) return 1 ;; esac; }

# Map a suite key to the (subdir, filename-token) a per-suite model uses. The
# token is the suite minus the "libero_" prefix (spatial/object/goal/10), EXCEPT
# bitvla names the long suite "libero_long/...-long" while gr00t_n1_7 uses the
# canonical "libero_10/...-10". Echoes "<subdir> <token>".
suite_dir_token() {
    local arch="$1" suite="$2"
    if [[ "$arch" == "bitvla" && "$suite" == "libero_10" ]]; then
        echo "libero_long long"
    else
        echo "${suite} ${suite#libero_}"
    fi
}

# ---- per-arch GGUF positional args (paths live where the SERVER runs) -------
# Echoes the positional args for `vla-server`, one per line. For per-suite models
# the suite selects the checkpoint; single-checkpoint models ignore it.
server_args_for() {
    local arch="$1" root="$2" suite="${3:-${DEFAULT_SUITE:-libero_object}}" sd tok
    case "$arch" in
        pi0)
            echo "${root}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf" ;;
        pi05)
            echo "${root}/pi05-libero-gguf/pi05-libero.gguf" ;;
        vla_adapter)
            echo "${root}/vla-adapter-libero-object-gguf/libero_object/vla-adapter-libero-object.gguf" ;;
        openvla_oft)
            echo "${root}/openvla-oft-libero-gguf/openvla-oft-libero.gguf" ;;
        smolvla)
            echo "${root}/smolvla-libero-gguf/smolvla-libero.gguf" ;;
        evo1)
            echo "${root}/evo1-libero-gguf/evo1-libero.gguf" ;;
        bitvla)
            read -r sd tok <<<"$(suite_dir_token bitvla "$suite")"
            echo "${root}/bitvla-libero-gguf/${sd}/bitvla-libero-${tok}.gguf" ;;
        gr00t_n1_5)
            echo "${root}/gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf" ;;
        gr00t_n1_6)
            echo "${root}/gr00tn1d6-libero-gguf/gr00tn1d6-libero.gguf" ;;
        gr00t_n1_7)
            read -r sd tok <<<"$(suite_dir_token gr00t_n1_7 "$suite")"
            echo "${root}/gr00tn1d7-libero-gguf/${sd}/gr00tn1d7-libero-${tok}.gguf" ;;
        *) echo "ERROR: unknown arch '$arch'" >&2; return 1 ;;
    esac
}

# Client-side dataset_statistics.json for the GR00T family ("" otherwise).
# This is a CLIENT arg (un-normalisation) so it is read on the orchestrator.
# gr00t_n1_7 is per-suite, so its stats live under the suite subdir.
stats_json_for() {
    local arch="$1" root="$2" suite="${3:-${DEFAULT_SUITE:-libero_object}}" sd tok
    case "$arch" in
        bitvla)
            read -r sd tok <<<"$(suite_dir_token bitvla "$suite")"
            echo "${root}/bitvla-libero-gguf/${sd}/dataset_statistics.json" ;;
        gr00t_n1_5) echo "${root}/gr00tn1d5-libero-object-gguf/dataset_statistics.json" ;;
        gr00t_n1_6) echo "${root}/gr00tn1d6-libero-gguf/dataset_statistics.json" ;;
        openvla_oft) echo "${root}/openvla-oft-libero-gguf/dataset_statistics.json" ;;
        gr00t_n1_7)
            read -r sd tok <<<"$(suite_dir_token gr00t_n1_7 "$suite")"
            echo "${root}/gr00tn1d7-libero-gguf/${sd}/dataset_statistics.json" ;;
        *) echo "" ;;
    esac
}

# Client-side local tokenizer DIR for archs that vendor their tokenizer alongside
# the GGUF instead of using an HF repo. Only gr00t_n1_6 does (its Eagle tokenizer
# lives in the model dir); "" => use the client's preset/HF tokenizer. Read on the
# orchestrator and passed to the client via --tokenizer.
tokenizer_for() {
    local arch="$1" root="$2"
    case "$arch" in
        gr00t_n1_6) echo "${root}/gr00tn1d6-libero-gguf" ;;
        *) echo "" ;;
    esac
}

# Export the GR00T env the SERVER needs (BF16 weights + per-arch embodiment).
# Honours a pre-set VLA_GR00T_EMBODIMENT; otherwise applies the per-arch default.
apply_gr00t_env() {
    local arch="$1"
    case "$arch" in
        gr00t_n1_5|gr00t_n1_6|gr00t_n1_7)
            export VLA_GR00T_BF16_WEIGHTS="${VLA_GR00T_BF16_WEIGHTS:-1}" ;;
    esac
    case "$arch" in
        gr00t_n1_5) export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-new_embodiment}" ;;
        gr00t_n1_6) export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-libero_panda}" ;;
        gr00t_n1_7) : ;;  # N1.7 LIBERO uses the GGUF's baked default embodiment
    esac
}

# OpenVLA-OFT selects its action-unnorm (server side) and proprio-norm (client
# side) stats by suite key; the two ends MUST agree and match the suite under
# test. Echo the key for an (arch, suite) ("" for every other arch).
openvla_unnorm_key() {
    local arch="$1" suite="${2:-${DEFAULT_SUITE:-libero_object}}"
    [[ "$arch" == openvla_oft ]] && echo "${suite}_no_noops" || echo ""
}

# Export the suite-matched unnorm key for the SERVER (no-op for other archs).
# Honours a pre-set VLA_OPENVLA_OFT_UNNORM_KEY.
apply_openvla_env() {
    local k; k="$(openvla_unnorm_key "$1" "${2:-}")"
    if [[ -n "$k" ]]; then
        export VLA_OPENVLA_OFT_UNNORM_KEY="${VLA_OPENVLA_OFT_UNNORM_KEY:-$k}"
    fi
}

# ---- ready-waits ------------------------------------------------------------
# Block until the server log prints the "bound ... ready" banner (or dies).
wait_ready_log() {
    local log="$1" pid="$2" timeout="${3:-600}"
    for _ in $(seq 1 "$timeout"); do
        grep -q "bound to .* ready" "$log" 2>/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || { echo "ERROR: server exited before ready; see $log" >&2; tail -n 40 "$log" >&2 || true; return 1; }
        sleep 1
    done
    echo "ERROR: server not ready within ${timeout}s; see $log" >&2; return 1
}

# Block until a remote host:port accepts a TCP connection (bash /dev/tcp; no nc).
wait_ready_tcp() {
    local host="$1" port="$2" timeout="${3:-600}"
    for i in $(seq 1 "$timeout"); do
        (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
        [[ $((i % 10)) -eq 1 ]] && echo "[wait] ${host}:${port} not up yet ..." >&2
        sleep 1
    done
    echo "ERROR: ${host}:${port} not reachable within ${timeout}s" >&2; return 1
}

# ---- memory sampler (Linux / Tegra) ----------------------------------------
# Identical metric set to eval/run_libero_server.sh: peak VmHWM (RSS), peak
# per-PID VRAM (nvidia-smi; null on Tegra), and peak/baseline/Δ system-used RAM
# (the only figure that captures the unified-memory iGPU carveout on Tegra).
# Atomic write (tmp+mv) so the gate never reads a half-written file.
_sys_used_kib() {
    awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{ if(t&&a) print t-a; else print 0 }' \
        /proc/meminfo 2>/dev/null || echo 0
}
mem_sampler_linux() {
    local pid="$1" out="$2" poll="${3:-1}"
    local peak_vram=0 peak_rss_kib=0 samples=0 vram_seen=0 peak_sys_kib=0 baseline_sys_kib=0
    local is_tegra=0; [[ -f /etc/nv_tegra_release ]] && is_tegra=1
    export LC_ALL=C
    trap 'stop=1' TERM INT HUP
    local stop=0
    baseline_sys_kib=$(_sys_used_kib); peak_sys_kib=$baseline_sys_kib
    _emit() {
        local vram_json="null"; (( vram_seen )) && vram_json="$peak_vram"
        local rss peak_sys base_sys delta_sys
        rss=$(awk      -v k="$peak_rss_kib"     'BEGIN{printf "%.1f",k/1024}')
        peak_sys=$(awk -v k="$peak_sys_kib"     'BEGIN{printf "%.1f",k/1024}')
        base_sys=$(awk -v k="$baseline_sys_kib" 'BEGIN{printf "%.1f",k/1024}')
        delta_sys=$(awk -v p="$peak_sys_kib" -v b="$baseline_sys_kib" 'BEGIN{d=p-b;if(d<0)d=0;printf "%.1f",d/1024}')
        printf '{"pid": %d, "peak_vram_mib": %s, "peak_rss_mib": %s, "peak_sys_used_mib": %s, "baseline_sys_used_mib": %s, "sys_used_delta_mib": %s, "is_tegra": %d, "samples": %d}\n' \
            "$pid" "$vram_json" "$rss" "$peak_sys" "$base_sys" "$delta_sys" "$is_tegra" "$samples" > "${out}.tmp" && mv -f "${out}.tmp" "$out"
    }
    while [[ $stop -eq 0 ]] && kill -0 "$pid" 2>/dev/null; do
        local rss; rss=$(awk '/^VmHWM:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)
        [[ -n "$rss" ]] && (( rss > peak_rss_kib )) && peak_rss_kib=$rss
        local vram; vram=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
                           | awk -F, -v p="$pid" '$1==p{gsub(/ /,"",$2);print $2;exit}' || true)
        if [[ -n "$vram" ]]; then vram_seen=1; (( vram > peak_vram )) && peak_vram=$vram; fi
        local sys; sys=$(_sys_used_kib); [[ -n "$sys" ]] && (( sys > peak_sys_kib )) && peak_sys_kib=$sys
        samples=$((samples+1)); _emit; sleep "$poll"
    done
    _emit
    echo "[mem-sampler] wrote $out (tegra=${is_tegra} samples=${samples})"
}

# ---- memory sampler (macOS / Apple Silicon) --------------------------------
# No /proc, no nvidia-smi, and Metal weights live in unified memory. Best effort:
#   peak_rss_mib       - max RSS via `ps -o rss=` (KiB). Undercounts GPU/wired.
#   peak_sys_used_mib  - (pages active+wired+compressed)*pagesize from vm_stat.
#   sys_used_delta_mib - rise over the sampler-start baseline.
# Emitted with is_tegra=0 and peak_vram_mib=null. NOT gated (m4 mem_metric=null);
# recorded for trend visibility only.
_mac_sys_used_mib() {
    local ps; ps=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
    vm_stat 2>/dev/null | awk -v p="$ps" '
        /Pages active/   {gsub(/\./,"",$3); a=$3}
        /Pages wired/    {gsub(/\./,"",$4); w=$4}
        /Pages occupied by compressor/ {gsub(/\./,"",$5); c=$5}
        END { printf "%.1f", (a+w+c)*p/1048576 }'
}
mem_sampler_macos() {
    local pid="$1" out="$2" poll="${3:-1}"
    local peak_rss_kib=0 samples=0
    export LC_ALL=C
    trap 'stop=1' TERM INT HUP
    local stop=0
    local baseline_sys peak_sys
    baseline_sys=$(_mac_sys_used_mib); peak_sys=$baseline_sys
    _emit() {
        local rss_mib delta
        rss_mib=$(awk -v k="$peak_rss_kib" 'BEGIN{printf "%.1f",k/1024}')
        delta=$(awk -v p="$peak_sys" -v b="$baseline_sys" 'BEGIN{d=p-b;if(d<0)d=0;printf "%.1f",d}')
        printf '{"pid": %d, "peak_vram_mib": null, "peak_rss_mib": %s, "peak_sys_used_mib": %s, "baseline_sys_used_mib": %s, "sys_used_delta_mib": %s, "is_tegra": 0, "samples": %d}\n' \
            "$pid" "$rss_mib" "$peak_sys" "$baseline_sys" "$delta" "$samples" > "${out}.tmp" && mv -f "${out}.tmp" "$out"
    }
    while [[ $stop -eq 0 ]] && kill -0 "$pid" 2>/dev/null; do
        local rss; rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        [[ -n "$rss" ]] && (( rss > peak_rss_kib )) && peak_rss_kib=$rss
        local sys; sys=$(_mac_sys_used_mib)
        awk -v s="$sys" -v p="$peak_sys" 'BEGIN{exit !(s>p)}' && peak_sys=$sys
        samples=$((samples+1)); _emit; sleep "$poll"
    done
    _emit
    echo "[mem-sampler] wrote $out (macos samples=${samples})"
}

# Dispatch by OS.
mem_sampler() {
    if [[ "$(uname -s)" == "Darwin" ]]; then mem_sampler_macos "$@"; else mem_sampler_linux "$@"; fi
}
