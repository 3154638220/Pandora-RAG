"""
LLM 调用客户端。
支持：
  - OpenAI / 兼容 OpenAI 接口的任意后端（Ollama、vLLM 等）
  - mock 模式（无 API Key 时，用简单抽取式答案模拟，方便离线 pipeline 测试）
"""
import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """\
你是一个严谨的问答助手。请根据以下检索到的上下文回答问题。
如无法确定，仍需给出最佳猜测，请直接输出答案（一个词组或短语），不要解释。

问题：{question}

已检索上下文：
{context}

答案："""


class LLMClient:
    def __init__(self, config):
        self.cfg = config
        self._client = None
        self._mock = False

        use_local_compat = bool(config.api_base)
        key_missing = not config.api_key or config.api_key in ("EMPTY", "")

        if key_missing and not use_local_compat:
            logger.warning(
                "未检测到 API Key 且未配置 OPENAI_API_BASE，将使用 mock 模式（抽取式答案）。"
                "使用云端 API 请在 .env 中设置 OPENAI_API_KEY；"
                "使用本机 Ollama 请设置 OPENAI_API_BASE（如 http://localhost:11434/v1）。"
            )
            self._mock = True
        else:
            try:
                from openai import OpenAI

                # Ollama 等 OpenAI 兼容接口通常接受任意非空 api_key
                api_key = config.api_key if not key_missing else "ollama"
                kwargs: Dict = {"api_key": api_key}
                if config.api_base:
                    kwargs["base_url"] = config.api_base
                self._client = OpenAI(**kwargs)
            except ImportError:
                logger.warning("openai 包未安装，降级为 mock 模式。")
                self._mock = True

    # ──────────────────────────────────────────────────────────
    def generate(
        self,
        question: str,
        context: str,
        extract_features: bool = True,
    ) -> Dict:
        """
        调用 LLM 生成答案，同时提取置信度特征。

        返回字典：
          answer       : str   生成的答案文本
          mean_logprob : float 生成 token 的平均对数概率（越高越自信）
          entropy      : float top-logprobs 估计的平均熵（越低越自信）
          token_count  : int   生成 token 数
        """
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        if self._mock:
            return self._mock_generate(context)

        try:
            kwargs = dict(
                model=self.cfg.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
            if extract_features:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as e_inner:
                if extract_features:
                    logger.warning(
                        "带 logprobs 的请求失败（%s），改用无 logprobs 重试（部分本地服务不支持）。",
                        e_inner,
                    )
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                    response = self._client.chat.completions.create(**kwargs)
                    extract_features = False
                else:
                    raise
            answer = (response.choices[0].message.content or "").strip()
            mean_logprob, entropy, token_count = self._extract_features(
                response.choices[0].logprobs
            )
            if not extract_features:
                mean_logprob, entropy, token_count = 0.0, 0.0, len(answer.split())
            return {
                "answer": answer,
                "mean_logprob": mean_logprob,
                "entropy": entropy,
                "token_count": token_count,
            }
        except Exception as e:
            logger.error("LLM 调用失败：%s，降级为 mock。", e)
            return self._mock_generate(context)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_features(logprobs_obj) -> tuple:
        """从 OpenAI logprobs 对象中提取平均对数概率和熵。"""
        if logprobs_obj is None or not logprobs_obj.content:
            return 0.0, 0.0, 0

        token_logprobs = []
        token_entropies = []

        for tok in logprobs_obj.content:
            lp = tok.logprob
            token_logprobs.append(lp)

            if tok.top_logprobs:
                raw_lps = np.array([t.logprob for t in tok.top_logprobs])
                # 对 top-k 概率做 softmax 归一化后估计熵
                log_probs_shifted = raw_lps - raw_lps.max()
                probs = np.exp(log_probs_shifted)
                probs /= probs.sum()
                entropy_val = float(-np.sum(probs * np.log(probs + 1e-12)))
                token_entropies.append(entropy_val)

        mean_logprob = float(np.mean(token_logprobs)) if token_logprobs else 0.0
        mean_entropy = float(np.mean(token_entropies)) if token_entropies else 0.0
        return mean_logprob, mean_entropy, len(token_logprobs)

    @staticmethod
    def _mock_generate(context: str) -> Dict:
        """离线模拟：从上下文抽取首个名词短语作为答案。"""
        words = context.split()
        answer = " ".join(words[:3]) if len(words) >= 3 else context[:20]
        return {
            "answer": answer,
            "mean_logprob": -1.0,
            "entropy": 1.5,
            "token_count": len(answer.split()),
        }
