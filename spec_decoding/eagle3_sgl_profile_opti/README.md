# eagle3_sgl_profile_opti 安装与运行

本目录是 eagle3_sgl（EAGLE-3）的性能优化版本，Python 导入路径
spec_decoding.eagle3_sgl_profile_opti。算法与接受率与原版完全一致（acc/round 不变），
改动全部在执行效率上，分五步：

| 步骤 | 改动 | 效果 |
|------|------|------|
| Step 1 | tree_verify_full 的接受选择张量化，去掉逐节点 argmax().item() | 消除重复 argmax |
| Step 2 | tree_draft 的 AR 展开按 BFS 深度批量化，去掉每层 64 次微前向 | 累计 +7% |
| Step 3 | 用 forward_pre_hook 只采集 eagle 层激活，去掉 output_hidden_states=True | 几乎无新增收益 |
| Step 4 | 树掩码由 Python 双重循环改为祖先矩阵向量化构造 | 掩码 34.8ms → 0.41ms/轮，累计 2.95× |
| Step 5 | target 换 StaticCache + 固定 shape masked SDPA + verify 前向 CUDA graph | 再 +30%，累计约 3.8× |

优化的完整推理过程与每一步的实测数据见 tps_opti_path.md，profile 方法与阶段划分见
example/opti_process.md，与 SGLang 的对比见 example/profile_compare_report.md，
StaticCache 与 DynamicCache 的差别见 DynamicCache-StaticCache.md。

---

## 1. 环境

全部操作在 docker 容器 xxxx 内进行。

| 项 | 值 |
|----|----|
| 镜像 | nvcr.io/nvidia/pytorch:25.04-py3 |
| Python | 3.12.3 |
| torch | 2.7.0a0+79aa17489c.nv25.04 |
| transformers | 5.3.0 |
| triton | 3.7.1（triton_tree verify backend 用） |
| huggingface_hub | 1.13.0（端到端脚本用 hf_hub_download 拉 draft config 与权重，必需） |
| GPU | H100 80GB；单卡即可，8B fp16 target + draft + static cache 约 20GB |

进入容器：

```bash
docker exec -it xxxx bash
```

---

## 2. 安装

本目录自带 pyproject.toml，只把当前目录映射为包 spec_decoding.eagle3_sgl_profile_opti
（及 example、kernels 子包），不打包上级 spec_decoding 下的其它组件。

```bash
cd /home/spec-decoding/spec_decoding/eagle3_sgl_profile_opti
python3 -m pip install -e .
```

安装后发行包名是 eagle3-sgl-profile-opti，验证：

```bash
pip list | grep -i eagle3-sgl-profile-opti
# eagle3-sgl-profile-opti  0.1.0  /home/spec-decoding/spec_decoding/eagle3_sgl_profile_opti

cd /tmp && python3 -c "import spec_decoding.eagle3_sgl_profile_opti as m; print(m.__file__)"
```

装好后在任意路径都能用 python3 -m 跑，不需要设置 PYTHONPATH。example 下的 phase
子目录没有 __init__.py，但作为 namespace 包同样能用 -m 直接跑，已验证。

额外依赖：原版 eagle3_sgl 组件。因为 example 里做 A/B 对照的脚本（phase0 的两个
profile/timing 脚本、phase1 的 ab_speed 与 validate_step1）以及 run_smoke_eagle3_sgl、
run_mtbench_eagle3_sgl 会 import spec_decoding.eagle3_sgl。只跑本文第 4、5 节的话不需要它。

```bash
cd /home/spec-decoding/spec_decoding/eagle3_sgl
python3 -m pip install -e .
```

不想安装也可以用 PYTHONPATH 跑：

```bash
cd /home/spec-decoding
PYTHONPATH=/home/spec-decoding python3 -m spec_decoding.eagle3_sgl_profile_opti.example.run_eagle3_sgl_llama31 --help
```

---

## 3. 不需要 GPU 的快速校验

Step 4 把树掩码从 Python 双重循环改成了向量化构造，这个脚本用 200 棵随机树逐元素比对
新旧两种实现，是本组件最值得先跑的正确性校验：

```bash
CUDA_VISIBLE_DEVICES="" python3 -m spec_decoding.eagle3_sgl_profile_opti.example.phase2_real_bottleneck.check_mask_equiv
```

实测输出：

```
ALL EQUAL ✓
```

另有一个 tiny-gpt2 冒烟脚本：

```bash
python3 -m spec_decoding.eagle3_sgl_profile_opti.example.run_smoke_eagle3_sgl --device cpu
```

实测输出：

```
[eagle3_sgl-smoke] eagle_layers=[0, 1] in=3 out=9 ids=[1, 2, 3, 5087, 5087, 5087, 5087, 5087, 5087]
[eagle3_sgl-smoke] OK
```

注意这个冒烟脚本 import 的是原版 spec_decoding.eagle3_sgl，不是本组件，所以它只能验证
环境和原版流程；要验证本组件请用上面的 check_mask_equiv 和下面的端到端。

---

## 4. 端到端测试（真实权重）

脚本 example/run_eagle3_sgl_llama31.py：真 Llama-3.1-8B-Instruct target + 官方 EAGLE-3
draft 权重，跑完与干净贪心逐 token 比对，并把性能追加进 perf.log。

| 角色 | 模型 |
|------|------|
| target | meta-llama/Llama-3.1-8B-Instruct，无 HF token 时自动回退到 NousResearch/Meta-Llama-3.1-8B-Instruct |
| draft | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B（含 fc / midlayer / norm / lm_head / d2t） |

推荐用 Step 5 的 CUDA graph 路径跑，它是无损的：

```bash
cd /home/spec-decoding/spec_decoding/eagle3_sgl_profile_opti
CUDA_VISIBLE_DEVICES=0 python3 -m spec_decoding.eagle3_sgl_profile_opti.example.run_eagle3_sgl_llama31 \
    --max-new-tokens 128 --topk 8 --num-steps 8 --max-tree-nodes 64 --cuda-graph
```

2026-07-29 实测（单张 H100，float16，成功）：

```
[eagle3_sgl] target=NousResearch/Meta-Llama-3.1-8B-Instruct draft=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B verify=full_model_tree(backend=auto) tree=cumulative ar_draft=True inc_acts=True inc_draft_ext=True eagle_layers=[2, 16, 29] topk=8 num_steps=8 max_nodes=64 dtype=float16
  new_tokens=70  rounds=27  accept/round=2.59
  wall=0.81s  spec_tok/s=86.47  lossless=True
  baseline(HF greedy) new_tokens=80  wall=1.22s  base_tok/s=48.74  speedup=1.77x
[eagle3sgl-e2e] OK (lossless vs clean greedy)
```

结尾出现 OK (lossless vs clean greedy) 即为端到端通过。

去掉 --cuda-graph 就是 Step 4 的 eager-opti 路径，吞吐更低：

```
  new_tokens=81  rounds=31  accept/round=2.61
  wall=1.20s  spec_tok/s=67.50  lossless=False
  baseline(HF greedy) new_tokens=80  wall=1.21s  base_tok/s=48.36  speedup=1.39x
```

这两条与 tps_opti_path.md 末尾记录的 86.3 / 66.6 tok/s，acc/round 不变（2.59 / 2.61）相符，属于已知且预期的行为

常用参数：

| 参数 | 说明 |
|------|------|
| --cuda-graph | 打开 Step 5：StaticCache + 固定 shape masked SDPA + verify 前向 CUDA graph |
| --topk / --num-steps / --max-tree-nodes | 树的宽度、深度、节点上限，默认 8 / 8 / 64 |
| --verify-mode | full_model_tree（默认，CUDA graph 要求它）或 reference_paths |
| --verify-attn-backend | auto / eager / flash_attn / triton_tree |
| --no-autoregressive-draft | 关掉 AR draft（CUDA graph 要求 AR draft，两者不能同时用） |
| --dtype | float16（默认）/ bfloat16 / float32 |
| --perf-log | 性能记录路径，默认 cwd 下 perf.log |

--cuda-graph 要求 full_model_tree + autoregressive_draft，否则会直接抛
ValueError: decode_cuda_graph requires full_model_tree + autoregressive_draft。

---

## 5. example 下各 phase 脚本

按优化流程分五个阶段，都可以用 python3 -m 加完整路径跑。phase0 与 phase1 里带 A/B 的
脚本需要先装原版 eagle3_sgl用于profile。

阶段 0，在原版上定位基线瓶颈：

| 脚本 | 作用 |
|------|------|
| phase0_profile.profile_eagle3_sgl | torch.profiler：分阶段 device 时间 + Top 算子 + 同步/拷贝计数 |
| phase0_profile.time_stages_eagle3_sgl | CUDA event 分阶段真实墙钟（prefill / draft_build / verify / replay / draft_extend） |
| phase0_profile.parse_sglang_trace | 解析 SGLang chrome trace 做对照，example/sglang_prof/ 下有一份 |

阶段 1，每步改动后的正确性与吞吐校验：

| 脚本 | 作用 |
|------|------|
| phase1_after_eachstep_opti.ab_speed | 无 profiler 的干净 A/B 吞吐，原版 vs opti |
| phase1_after_eachstep_opti.validate_step1 | Step1 输出逐位一致 + 吞吐收益 |
| phase1_after_eachstep_opti.validate_step5 | Step5（CUDA graph）输出逐位一致 + 吞吐收益 |

阶段 2，微观定位真正的瓶颈：

| 脚本 | 作用 |
|------|------|
| phase2_real_bottleneck.probe_fwd_cost | 微基准：target 前向墙钟随 token 数几乎不变，证明 launch-bound |
| phase2_real_bottleneck.probe_verify_internals | 把每轮 verify 拆成建掩码 / 前向 / accept，发现掩码构建才是大头 |
| phase2_real_bottleneck.check_mask_equiv | 向量化掩码与原双重循环逐元素等价 |

阶段 3，优化后复测两张表：

| 脚本 | 作用 |
|------|------|
| phase3_reprofile_after_bottleneck.time_stages_opti | 优化版的分阶段墙钟 |
| phase3_reprofile_after_bottleneck.profile_opti | 优化版的算子耗时与同步计数 |

阶段 4，CUDA graph 的前置校验与计时：

| 脚本 | 作用 |
|------|------|
| phase4_cudagraph.spike_static_verify | 前置要求：StaticCache + cache_position + padded 4D mask + SDPA 的 node_logits 与 DynamicCache extend verify 逐元素一致 |
| phase4_cudagraph.stage_events_opti | event 分阶段计时 |
| phase4_cudagraph.probe_graph_replay | 隔离测 verify graph.replay() 的纯 GPU 时间 vs eager triton verify |

例如：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m spec_decoding.eagle3_sgl_profile_opti.example.phase1_after_eachstep_opti.ab_speed --help
```

example 下还有 run_mtbench_eagle3_sgl（MT-Bench 风格吞吐，题目默认
example/mtbench/question.jsonl，注意它跑的是原版 eagle3_sgl）。各脚本都支持 --help。

---

## 6. 常见问题

1. ModuleNotFoundError: spec_decoding.eagle3_sgl —— 本组件的 A/B 与冒烟脚本要用原版组件，
   按第 2 节把 eagle3_sgl 也装上。
2. 启动时打印 meta-llama 401 gated repo —— 正常，脚本随后自动回退到 NousResearch 镜像；
   想用官方仓库就先 export HF_TOKEN=... 或 huggingface-cli login。
3. ValueError: decode_cuda_graph requires full_model_tree + autoregressive_draft ——
   --cuda-graph 不能和 --verify-mode reference_paths 或 --no-autoregressive-draft 一起用。
4. lossless=False 且打印 NOTE 分叉 —— eager-opti 路径的已知 fp16 平局差异，决策等价；
   要无损就加 --cuda-graph。
5. torch_dtype is deprecated 警告 —— transformers 5.x 的提示，不影响结果。
6. CUDA out of memory —— 8B fp16 加 static cache 约需 20GB，用 CUDA_VISIBLE_DEVICES
   指定一张空卡，先用 nvidia-smi 看占用。

## 7. 进一步提升性能

1. 去掉target replay，复用target verify
2. kernel部分全部与sglang kernel对齐，以优化draft推理和target verify的耗时
