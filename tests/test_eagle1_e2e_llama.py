# SPDX-License-Identifier: Apache-2.0
"""
Llama 系列 target + 轻量 EAGLE1 draft(复用 target 的 embedding/lm_head + 1 层
decoder,首层跳过 input_layernorm)与静态 top-k 树的端到端测试.

draft 为随机初始化(没有训练过的 EAGLE1 checkpoint),因此接受率很低 —— 但
greedy 树验证仍保证输出相对纯 greedy 是无损的.

需要下载权重(建议用 GPU),用环境变量做开关:

HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RUN_SPEC_E2E=1 PYTHONPATH=. \
.venv/bin/python -m unittest tests.test_eagle1_e2e_llama -v

"""

from __future__ import annotations

import os
import unittest

import torch

from spec_decoding.eagle import (
    Eagle1Config,
    Eagle1Generator,
    build_eagle1_draft,
    make_eagle1_draft_topk_fn,
)

_RUN = os.environ.get("RUN_SPEC_E2E") == "1"
_TARGET_ID = os.environ.get("SPEC_E2E_TARGET", "JackFram/llama-160m")
_PROMPT = os.environ.get("SPEC_E2E_PROMPT", "The capital of France is")
_MAX_NEW = int(os.environ.get("SPEC_E2E_MAX_NEW", "24"))


@unittest.skipUnless(_RUN, "set RUN_SPEC_E2E=1 to run the EAGLE1 Llama end-to-end test")
class TestEagle1E2ELlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if cls.device.type == "cuda" else torch.float32
        cls.tok = AutoTokenizer.from_pretrained(_TARGET_ID)
        cls.target = (
            AutoModelForCausalLM.from_pretrained(_TARGET_ID, torch_dtype=dtype)
            .to(cls.device)
            .eval()
        )
        # * from draft.py build draft.model
        cls.draft = build_eagle1_draft(cls.target, num_layers=1)

    def test_lossless_vs_greedy(self) -> None:
        enc = self.tok(_PROMPT, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        plen = input_ids.shape[1]

        with torch.inference_mode():
            base = self.target.generate(
                input_ids,
                max_new_tokens=_MAX_NEW,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )
        base_cont = base[0, plen:]

        cfg = Eagle1Config(
            # topk=2,
            # num_steps=2,
            # max_tree_nodes=8,
            # verify_mode="reference_paths",
            topk=2,
            num_steps=2,
            max_tree_nodes=8,
            verify_mode="full_model_tree",
        )
        # * draft function in draft.py
        fn = make_eagle1_draft_topk_fn(self.draft, topk=cfg.topk)
        gen = Eagle1Generator(self.target, cfg, fn)
        out = gen.generate(
            input_ids, max_new_tokens=_MAX_NEW, eos_token_id=self.tok.eos_token_id
        )
        spec_cont = out[0, plen:]

        # * 对比 speculative decoding and base model res
        m = min(int(base_cont.numel()), int(spec_cont.numel()))
        self.assertGreater(m, 0)
        if not torch.equal(base_cont[:m].cpu(), spec_cont[:m].cpu()):
            self.fail(
                "EAGLE1 output diverged from greedy baseline:\n"
                f"  base: {base_cont[:m].tolist()}\n  spec: {spec_cont[:m].tolist()}"
            )


if __name__ == "__main__":
    unittest.main()
