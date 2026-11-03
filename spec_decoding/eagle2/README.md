# eagle2 (EAGLE-2) 安装与运行

本目录是 EAGLE-2 投机解码组件,Python 导入路径 spec_decoding.eagle2。

EAGLE-2 与 EAGLE-1 共用同一份 draft checkpoint(draft 网络结构完全一致:fc(2H→H) +
1 层 GQA decoder + 复用 target embedding/lm_head),差别只在推理期的树展开策略:
EAGLE-2 用累计 log 概率的 beam + 全局剪枝(tree_expand_mode=cumulative),把节点预算
花在更可能被接受的分支上,因此通常比 EAGLE-1 的静态 top-k 树接受更多 token。

---

## 1. 环境

全部操作在 docker 容器 xxxx 内进行。

| 项 | 值 |
|----|----|
| 镜像 | nvcr.io/nvidia/pytorch:25.04-py3 |
| Python | 3.12.3 |
| torch | 2.7.0a0+79aa17489c.nv25.04 |
| transformers | 5.3.0 |
| huggingface_hub | 1.13.0(端到端脚本用 hf_hub_download 拉 draft 权重,必需) |
| GPU | H100 80GB;单卡即可,8B fp16 target + draft 约 20GB |

进入容器:

```bash
docker exec -it xxxx bash
```

---

## 2. 安装

本目录自带 pyproject.toml,只把当前目录映射为包 spec_decoding.eagle2(及 example、
kernels 子包),不打包上级 spec_decoding 下的其它组件。

```bash
cd /home/spec-decoding/spec_decoding/eagle2
python3 -m pip install -e .
```

安装后发行包名是 eagle2,验证:

```bash
pip list | grep -i eagle2
# eagle2  0.1.0  /home/spec-decoding/spec_decoding/eagle2

cd /tmp && python3 -c "import spec_decoding.eagle2 as m; print(m.__file__)"
```

装好后在任意 cwd 都能用 python3 -m 跑,不需要设置 PYTHONPATH。

若不想安装,也可以在仓库根目录用 PYTHONPATH 跑:

```bash
cd /home/spec-decoding
PYTHONPATH=/home/spec-decoding python3 -m spec_decoding.eagle2.example.run_smoke_tree
```

---

## 3. CPU 冒烟(不下载大模型,先确认装对了)

两个冒烟脚本,都不需要 GPU:

```bash
# 只验证建树 + verify layout / SDPA bias,纯 mock,最快
python3 -m spec_decoding.eagle2.example.run_smoke_tree

# 走完整 generator:tiny-gpt2 + 随机 draft 权重
python3 -m spec_decoding.eagle2.example.run_smoke_eagle2_draft --device cpu
```

实测输出:

```
[eagle2-smoke] tree nodes=4 tokens=[100, 200, 101, 201]
[eagle2-smoke] verify bias shape=(4, 4)
[eagle2-smoke] OK

[eagle2-draft-smoke] in=3 out=9 ids=[1, 2, 3, 5087, 5087, 5087, 5087, 5087, 5087]
[eagle2-draft-smoke] OK
```

加载 tiny-gpt2 时会打印 attn.masked_bias UNEXPECTED 提示,属正常,可忽略。

---

## 4. 端到端测试(真实权重)

脚本 run_eagle2_llama3.py:真 Llama-3-8B-Instruct target + 官方 EAGLE draft 权重,
跑完后与干净贪心逐 token 比对,不一致就 assert 失败,并把性能追加进 perf.log。

| 角色 | 模型 |
|------|------|
| target | meta-llama/Meta-Llama-3-8B-Instruct,无 HF token 时自动回退到 NousResearch/Meta-Llama-3-8B-Instruct |
| draft | yuhuili/EAGLE-LLaMA3-Instruct-8B(与 EAGLE-1 同一份权重) |

运行:

```bash
cd /home/spec-decoding/spec_decoding/eagle2/example
CUDA_VISIBLE_DEVICES=1 python3 -m spec_decoding.eagle2.example.run_eagle2_llama3 \
    --max-new-tokens 96 --topk 4 --num-steps 5 --max-tree-nodes 32
```

perf.log 默认写在当前 cwd,所以上面先 cd 到 example 目录;也可以用 --perf-log 指定绝对路径。

2026-07-28 实测(单张 H100,float16,成功):

```
[eagle2] target=NousResearch/Meta-Llama-3-8B-Instruct draft=yuhuili/EAGLE-LLaMA3-Instruct-8B verify=full_model_tree(backend=auto) tree=cumulative topk=4 num_steps=5 max_nodes=32 dtype=float16
  new_tokens=92  rounds=44  accept/round=2.09
  wall=5.02s  spec_tok/s=18.32  lossless=True
  baseline(HF greedy) new_tokens=91  wall=1.45s  base_tok/s=62.89  speedup=0.29x
[eagle2-e2e] OK (lossless)
```

--max-new-tokens 128 也通过(提前遇到 EOS,同样是 92 token,lossless=True)。

结尾出现 OK (lossless) 即为端到端通过。speedup 小于 1 是预期的:这是 HF eager 的教学
实现,每轮有 Python 建树与逐节点同步开销,接受率(accept/round)才是这里要看的指标。

常用参数:

| 参数 | 说明 |
|------|------|
| --tree-expand-mode | cumulative 是 EAGLE-2 的累计 log 概率 beam 剪枝;bfs 是退化成静态树的调试对照 |
| --topk / --num-steps / --max-tree-nodes | 树的宽度、深度、节点上限 |
| --verify-mode | full_model_tree(复用前缀 KV 的树掩码 verify,快)或 reference_paths(逐叶干净前向,慢但最稳) |
| --verify-attn-backend | auto / eager / flash_attn / triton_tree(triton_tree 走 SGLang 风格 kernel) |
| --dtype | float16(默认)/ bfloat16 / float32 |
| --perf-log | 性能记录路径,默认 cwd 下 perf.log |

想直观看到 EAGLE-2 剪枝的收益,可以固定其它参数,对比
--tree-expand-mode cumulative 与 --tree-expand-mode bfs 的 accept/round。

---

## 5. example 目录其它脚本

| 脚本 | 作用 | 命令 |
|------|------|------|
| run_smoke_tree | 纯 mock 建树冒烟,最快 | python3 -m spec_decoding.eagle2.example.run_smoke_tree |
| run_smoke_eagle2_draft | CPU 冒烟,tiny-gpt2 + 随机 draft | python3 -m spec_decoding.eagle2.example.run_smoke_eagle2_draft --device cpu |
| run_eagle2_llama3 | 本文第 4 节的端到端 | 见上 |
| perf.log | 历次运行的性能追加记录 | — |

各脚本都支持 --help。目录下另有 setup.txt 与 running_log.txt,是迁移前的环境与运行记录。
