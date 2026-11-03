# eagle (EAGLE-1) 安装与运行

本目录是 EAGLE-1 投机解码组件,Python 导入路径 spec_decoding.eagle。

EAGLE-1 在链式投机之上加了三件事:target 特征条件的轻量 draft(复用 target 的
embedding/lm_head + fc(2H→H) + 1 层 decoder)、topk>1 的静态 top-k 候选树、以及一次
target 前向验证整棵树的树掩码 verify。树形状只由 topk / num_steps / max_tree_nodes
静态决定,不做 EAGLE-2 的累计概率剪枝。

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

本目录自带 pyproject.toml,只把当前目录映射为包 spec_decoding.eagle(及 example、
kernels 子包),不打包上级 spec_decoding 下的其它组件。

```bash
cd /home/spec-decoding/spec_decoding/eagle
python3 -m pip install -e .
```

安装后发行包名是 eagle1,验证:

```bash
pip list | grep -i eagle1
# eagle1  0.1.0  /home/spec-decoding/spec_decoding/eagle

cd /tmp && python3 -c "import spec_decoding.eagle as m; print(m.__file__)"
```

装好后在任意 cwd 都能用 python3 -m 跑,不需要设置 PYTHONPATH。

若不想安装,也可以在仓库根目录用 PYTHONPATH 跑:

```bash
cd /home/spec-decoding
PYTHONPATH=/home/spec-decoding python3 -m spec_decoding.eagle.example.run_smoke_eagle1 --device cpu
```

---

## 3. CPU 冒烟(不下载大模型,先确认装对了)

用 sshleifer/tiny-gpt2 + 随机 draft 权重,只验证流程能跑通(随机 draft 接受率低,
但 target verify 保证输出与普通贪心一致)。

```bash
python3 -m spec_decoding.eagle.example.run_smoke_eagle1 --device cpu
```

实测输出:

```
[eagle1-smoke] in=3 out=9 ids=[1, 2, 3, 5087, 5087, 5087, 5087, 5087, 5087]
[eagle1-smoke] OK
```

加载 tiny-gpt2 时会打印 attn.masked_bias UNEXPECTED 提示,属正常(GPT2 旧权重字段),可忽略。

---

## 4. 端到端测试(真实权重)

脚本 run_eagle1_llama3.py:真 Llama-3-8B-Instruct target + 官方 EAGLE-1 draft 权重,
跑完后与干净贪心逐 token 比对,不一致就 assert 失败,并把性能追加进 perf.log。

| 角色 | 模型 |
|------|------|
| target | meta-llama/Meta-Llama-3-8B-Instruct,无 HF token 时自动回退到 NousResearch/Meta-Llama-3-8B-Instruct |
| draft | yuhuili/EAGLE-LLaMA3-Instruct-8B(官方权重,单个 pytorch_model.bin) |

运行:

```bash
cd /home/spec-decoding/spec_decoding/eagle/example
CUDA_VISIBLE_DEVICES=1 python3 -m spec_decoding.eagle.example.run_eagle1_llama3 \
    --max-new-tokens 64 --topk 4 --num-steps 4 --max-tree-nodes 24
```

perf.log 默认写在当前 cwd,所以上面先 cd 到 example 目录;也可以用 --perf-log 指定绝对路径。

2026-07-28 实测(单张 H100,float16,成功):

```
[eagle1] target=NousResearch/Meta-Llama-3-8B-Instruct draft=yuhuili/EAGLE-LLaMA3-Instruct-8B verify=full_model_tree(backend=auto) topk=4 num_steps=4 max_nodes=24 dtype=float16
  new_tokens=64  rounds=29  accept/round=2.21
  wall=2.63s  spec_tok/s=24.35  lossless=True
  baseline(HF greedy) new_tokens=64  wall=1.02s  base_tok/s=62.59  speedup=0.39x
xxxx[eagle1-e2e] OK (lossless)
```

--max-new-tokens 96 同样通过(生成到 EOS 共 92 token,accept/round=2.19,
spec_tok/s=24.57,lossless=True)。

结尾出现 OK (lossless) 即为端到端通过。speedup 小于 1 是预期的:这是 HF eager 的教学
实现,每轮有 Python 建树与逐节点同步开销,接受率(accept/round)才是这里要看的指标。

常用参数:

| 参数 | 说明 |
|------|------|
| --topk / --num-steps / --max-tree-nodes | 静态树的宽度、深度、节点上限 |
| --verify-mode | full_model_tree(复用前缀 KV 的树掩码 verify,快)或 reference_paths(逐叶干净前向,慢但最稳) |
| --verify-attn-backend | auto / eager / flash_attn / triton_tree |
| --dtype | float16(默认)/ bfloat16 / float32 |
| --prompt / --max-new-tokens | 输入与生成长度 |
| --perf-log | 性能记录路径,默认 cwd 下 perf.log |

---

## 5. example 目录其它脚本

| 脚本 | 作用 | 命令 |
|------|------|------|
| run_smoke_eagle1 | CPU 冒烟,tiny-gpt2 + 随机 draft | python3 -m spec_decoding.eagle.example.run_smoke_eagle1 --device cpu |
| run_eagle1_llama3 | 本文第 4 节的端到端 | 见上 |
| run_eagle1_qwen25 | Qwen2.5-32B + mit-han-lab/Qwen2.5-32B-Eagle-RL,bf16 单卡 H100 约 64GB | CUDA_VISIBLE_DEVICES=1 python3 -m spec_decoding.eagle.example.run_eagle1_qwen25 --max-new-tokens 64 --dtype bfloat16 |
| perf.log | 历次运行的性能追加记录 | — |

各脚本都支持 --help。
