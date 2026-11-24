#!/usr/bin/env bash
# 端到端 A/B: text/vl × 优化栈(eager / 仅target graph / 全部优化Full / SVD r128/256/512)。
# 每组 bench 内置 HF 原生贪心基线, 输出各自对 HF 的加速比与 lossless。
set -u
cd "$(dirname "$0")/../../../.."
export HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
BENCH=spec_decoding.eagle3_sgl.example.bench_accept_tpot
COMMON="--explain --max-new-tokens 130 --num-prompts 10"

for mode in text vl; do
    for spec in \
        "eager|--no-cuda-graph" \
        "target_graph|--cuda-graph --no-draft-cuda-graph" \
        "full|--cuda-graph" \
        "svd_r128|--cuda-graph --draft-svd-rank 128" \
        "svd_r256|--cuda-graph --draft-svd-rank 256" \
        "svd_r512|--cuda-graph --draft-svd-rank 512"
    do
        label="${spec%%|*}"
        flags="${spec#*|}"
        echo "===== mode=$mode config=$label ====="
        # shellcheck disable=SC2086
        $PY -m $BENCH --mode $mode $flags $COMMON || echo "FAILED mode=$mode config=$label"
    done
done
echo "===== ALL DONE ====="
