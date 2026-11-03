# `eagle3_sgl` 示例说明

本目录(`eagle3_sgl/example/`)提供 **EAGLE3 投机解码**(Python 导入路径:`spec_decoding.eagle3_sgl`)的冒烟测试、端到端 Llama、基准与性能分析脚本。

---

## 路径约定

| 名称 | 含义 |
|------|------|
| **eagle3_sgl 目录** | 本 README 的父目录:内含 `pyproject.toml`、`runner.py`、`draft_model.py` 等,即本组件根目录。 |
| **example 目录** | 本 README 所在目录:`eagle3_sgl/example/`。 |

磁盘布局:本组件常位于 `spec_decoding/eagle3_sgl/`(`spec_decoding` 为 Python 包目录名);**仅就安装而言**,`eagle3_sgl/pyproject.toml` 只把**当前目录**映射为包 `spec_decoding.eagle3_sgl`(及 `example`、`kernels`),**不会**把 `spec_decoding` 下其它子包一并打包或声明为依赖。

---

## 1. 安装

### 1.1 环境要求

- Python ≥ 3.9  
- `torch`、`transformers`(由本组件目录下的 `pyproject.toml` 声明)

### 1.2 可编辑安装(推荐)

在 **`eagle3_sgl` 目录**(含 `pyproject.toml` 的那一层)执行:

```bash
cd /path/to//eagle3_sgl
python3 -m pip install -e .
```

安装完成后,可在**任意当前工作目录**下使用:

```bash
python3 -m spec_decoding.eagle3_sgl.example.<模块名> [参数...]
```

### 1.3 不安装、直接运行示例

若未执行 `pip install -e .`,则需把 **`spec_decoding` 文件夹的上一级目录** 加入 `PYTHONPATH`(使解释器能找到包 `spec_decoding`),再运行上述 `-m` 命令。

### 1.4 运行大模型示例时的额外依赖

| 依赖 | 用途 |
|------|------|
| `huggingface_hub` | 从 Hub 下载 Llama / EAGLE3 draft 配置与 `pytorch_model.bin` 等 |
| Hugging Face 登录(`huggingface-cli login` 或 `HF_TOKEN`) | 部分 Llama 门禁;脚本多会回退到 `NousResearch/Meta-Llama-3.1-8B-Instruct` |
| NVIDIA GPU(推荐) | Llama 8B + EAGLE3、MT-Bench、profiler 等默认走 CUDA;仅 `run_smoke_eagle3_sgl` 适合 CPU 小模型 |

```bash
python3 -m pip install huggingface_hub
```

默认 EAGLE3 draft:`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`。其它权重格式需与 `build_eagle3_sgl_draft` 的加载方式对齐。

---

## 2. 运行 example

| 方式 | 建议 cwd | 说明 |
|------|-----------|------|
| `python3 -m spec_decoding.eagle3_sgl.example.<模块名>` | 任意(已 `pip install -e .`) | 组件已装入环境。 |
| 同上(未安装) | 任意 | 需事先设置好 `PYTHONPATH`(见 1.3)。 |


**冒烟(安装后):**

```bash
python3 -m spec_decoding.eagle3_sgl.example.run_smoke_eagle3_sgl --device cpu
```

---

## 3. 各脚本做什么、怎么跑

以下命令在安装 `eagle3-sgl` 后可在任意目录执行;未安装时需自行配置 `PYTHONPATH`。

| 模块 | 作用简述 | 典型命令 |
|------|----------|----------|
| `run_smoke_eagle3_sgl` | 最小冒烟:小 GPT-2 + 随机 draft + `reference_paths` verify,可 CPU。 | `python3 -m spec_decoding.eagle3_sgl.example.run_smoke_eagle3_sgl --device cpu` |
| `run_eagle3_sgl_llama31` | 真 EAGLE3:Llama-3.1-8B + Hub draft;`--perf-log`(默认 `perf.log`)写在 cwd。 | `python3 -m spec_decoding.eagle3_sgl.example.run_eagle3_sgl_llama31 --max-new-tokens 96 --device cuda` |
| `run_mtbench_eagle3_sgl` | MT-Bench 风格吞吐;题目默认 `eagle3_sgl/example/mtbench/question.jsonl`。 | `python3 -m spec_decoding.eagle3_sgl.example.run_mtbench_eagle3_sgl --num-questions 10 --max-new-tokens 256 --skip-baseline` |
| `profile_eagle3_sgl` | `torch.profiler`;`--trace-out` 建议写到 example 下(相对 **eagle3_sgl 目录**):`example/eagle3_trace.json`。 | `python3 -m spec_decoding.eagle3_sgl.example.profile_eagle3_sgl --max-new-tokens 128 --trace-out example/eagle3_trace.json` |
| `time_stages_eagle3_sgl` | 分阶段墙钟。 | `python3 -m spec_decoding.eagle3_sgl.example.time_stages_eagle3_sgl --max-new-tokens 128` |
| `parse_chrome_trace` | 聚合 Chrome trace(gzip JSON)。在 **eagle3_sgl 目录** 下:`python3 example/parse_chrome_trace.py eagle3_trace.json.gz` | 见左栏 |

各脚本可加 `--help` 查看参数。

---

## 4. example 目录内其它文件

- `mtbench/question.jsonl`:mtbench任务默认题目路径。  

---

## 5. 常见问题

1. **`ModuleNotFoundError: spec_decoding`**:在 `eagle3_sgl` 目录执行 `python3 -m pip install -e .`;或按 1.3 设置 `PYTHONPATH`。  
2. **Llama 403**:登录 Hugging Face 或使用脚本的 `NousResearch/...` 回退。  
3. **CPU 上跑大模型**:默认面向 GPU;CPU 请用 `run_smoke_eagle3_sgl`。  
4. **`parse_chrome_trace`**:脚本使用 `gzip.open`;若得到明文 `.json`,先 `gzip -k your_trace.json` 再传 `.json.gz`。  
