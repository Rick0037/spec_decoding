# eagle3_sgl (EAGLE-3) 安装与运行

本目录是 EAGLE-3 投机解码组件,Python 导入路径 spec_decoding.eagle3_sgl。

相对 EAGLE-2,EAGLE-3 的三处关键差别:draft 的条件不再是 target 最后一层 hidden,而是
低/中/高多层激活拼接后过 fc(n*H→H)(eagle_layers,Llama-3.1-8B 上解析为 [2, 16, 29]);
draft 有自己的小词表 lm_head(draft_vocab_size=32000)并通过 d2t 映射回 target token id;
draft 按 KV cache 自回归展开(autoregressive_draft,默认开启)。

本目录已有一份更细的脚本清单在 example/README.md,本文只讲安装和跑通端到端。

---

## 1. 环境

全部操作在 docker 容器 xxxx 内进行。

| 项 | 值 |
|----|----|
| 镜像 | nvcr.io/nvidia/pytorch:25.04-py3 |
| Python | 3.12.3 |
| torch | 2.7.0a0+79aa17489c.nv25.04 |
| transformers | 5.3.0 |
| huggingface_hub | 1.13.0(端到端脚本用 hf_hub_download 拉 draft config 与权重,必需) |
| GPU | H100 80GB;单卡即可,8B fp16 target + draft 约 20GB |

进入容器:

```bash
docker exec -it xxxx bash
```

---

## 2. 安装

本目录自带 pyproject.toml,只把当前目录映射为包 spec_decoding.eagle3_sgl(及 example、
kernels 子包),不打包上级 spec_decoding 下的其它组件。

```bash
cd /home/spec-decoding/spec_decoding/eagle3_sgl
python3 -m pip install -e .
```

安装后发行包名是 eagle3-sgl,验证:

```bash
pip list | grep -i eagle3
# eagle3-sgl  0.1.0  /home/spec-decoding/spec_decoding/eagle3_sgl

cd /tmp && python3 -c "import spec_decoding.eagle3_sgl as m; print(m.__file__)"
```

装好后在任意 cwd 都能用 python3 -m 跑,不需要设置 PYTHONPATH。

若不想安装,也可以在仓库根目录用 PYTHONPATH 跑:

```bash
cd /home/spec-decoding
PYTHONPATH=/home/spec-decoding python3 -m spec_decoding.eagle3_sgl.example.run_smoke_eagle3_sgl --device cpu
```

---

## 3. CPU 冒烟(不下载大模型,先确认装对了)

用 sshleifer/tiny-gpt2 + 随机 draft 权重,只验证流程能跑通。

```bash
python3 -m spec_decoding.eagle3_sgl.example.run_smoke_eagle3_sgl --device cpu
```

实测输出:

```
[eagle3_sgl-smoke] eagle_layers=[0, 1] in=3 out=10 ids=[1, 2, 3, 5087, 5087, 5087, 5087, 5087, 5087, 5087]
[eagle3_sgl-smoke] OK
```

加载 tiny-gpt2 时会打印 attn.masked_bias UNEXPECTED 提示,属正常,可忽略。

---

## 4. 端到端测试(真实权重)

脚本 run_eagle3_sgl_llama31.py:真 Llama-3.1-8B-Instruct target + 官方 EAGLE-3 draft
权重,跑完后与干净贪心逐 token 比对,不一致就 assert 失败,并把性能追加进 perf.log。

| 角色 | 模型 |
|------|------|
| target | meta-llama/Llama-3.1-8B-Instruct,无 HF token 时自动回退到 NousResearch/Meta-Llama-3.1-8B-Instruct |
| draft | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B(含 fc / midlayer / norm / lm_head / d2t) |

推荐用下面这条命令跑端到端,verify 走 reference_paths,96 token 无损通过:

```bash
cd /home/spec-decoding/spec_decoding/eagle3_sgl
CUDA_VISIBLE_DEVICES=1 python3 -m spec_decoding.eagle3_sgl.example.run_eagle3_sgl_llama31 \
    --max-new-tokens 96 --topk 4 --num-steps 4 --max-tree-nodes 24 \
    --dtype bfloat16 --verify-mode reference_paths
```

2026-07-28 实测(单张 H100,成功):

```
[eagle3_sgl] target=NousResearch/Meta-Llama-3.1-8B-Instruct draft=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B verify=reference_paths(backend=auto) tree=cumulative ar_draft=True inc_acts=True inc_draft_ext=True eagle_layers=[2, 16, 29] topk=4 num_steps=4 max_nodes=24 dtype=bfloat16
  new_tokens=78  rounds=30  accept/round=2.60
  wall=9.20s  spec_tok/s=8.48  lossless=True
  baseline(HF greedy) new_tokens=80  wall=1.25s  base_tok/s=64.15  speedup=0.13x
[eagle3-e2e] OK (lossless)
```

结尾出现 OK (lossless) 即为端到端通过。reference_paths 是逐叶干净前向,慢但最稳;
它的 spec_tok/s 只有参考意义,要看吞吐请用下面的 full_model_tree。

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m spec_decoding.eagle3_sgl.example.run_eagle3_sgl_llama31 \
    --max-new-tokens 48 --topk 4 --num-steps 4 --max-tree-nodes 24 --dtype bfloat16
```

实测(成功):

```
  new_tokens=49  rounds=20  accept/round=2.45
  wall=1.92s  spec_tok/s=25.53  lossless=True
[eagle3-e2e] OK (lossless)
```

perf.log 默认写在当前 cwd;也可以用 --perf-log 指定绝对路径。

常用参数:

| 参数 | 说明 |
|------|------|
| --verify-mode | full_model_tree(树掩码 verify,快)或 reference_paths(逐叶干净前向,慢但无损) |
| --topk / --num-steps / --max-tree-nodes | 树的宽度、深度、节点上限,默认 8 / 8 / 64 |
| --no-autoregressive-draft | 关掉 KV cache 自回归 draft,回退单 token draft |
| --tree-expand-mode | cumulative(默认);static 在 AR draft 下被忽略 |
| --dtype | float16(默认)/ bfloat16 / float32 |
| --perf-log | 性能记录路径,默认 cwd 下 perf.log |

---

## 5. example 目录其它脚本

| 脚本 | 作用 |
|------|------|
| run_smoke_eagle3_sgl | CPU 冒烟,tiny-gpt2 + 随机 draft |
| run_eagle3_sgl_llama31 | 本文第 4 节的端到端 |
| run_mtbench_eagle3_sgl | MT-Bench 风格吞吐,题目默认 example/mtbench/question.jsonl |
| profile_eagle3_sgl | torch.profiler 算子级剖析,--trace-out 指定 trace 路径 |
| time_stages_eagle3_sgl | CUDA event 分阶段墙钟计时 |
| parse_chrome_trace | 聚合 Chrome trace(gzip JSON) |

更细的说明见 example/README.md。各脚本都支持 --help。