"""Qwen3-VL EAGLE3 的无训练 SVD 候选 Head 实验脚本。"""

"""

cd /root/autodl-tmp/code/spec-decoding-main

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=.
PY=.venv/bin/python

# 1. 正确性验证
$PY -m spec_decoding.eagle3_sgl.example.svd_draft.validate_svd -v

# 2. 真实 LM Head 微基准
$PY -m spec_decoding.eagle3_sgl.example.svd_draft.bench_svd_head \
  --rank 128 --num-candidates 128 --topk 8 \
  --warmup 10 --repeats 50

# 3. 真实 hidden 上的完整草稿树实验
$PY -m spec_decoding.eagle3_sgl.example.svd_draft.bench_svd_tree \
  --rank 128 --num-candidates 128 \
  --topk 8 --num-steps 8 --max-tree-nodes 64 \
  --warmup 3 --repeats 15

"""
