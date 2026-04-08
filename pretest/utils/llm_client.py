"""
LLM 调用客户端。
支持：
  - OpenAI / 兼容 OpenAI 接口的任意后端（Ollama、vLLM 等）
  - mock 模式（无 API Key 时，用简单抽取式答案模拟，方便离线 pipeline 测试）
"""
import logging
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """\
你是一个严谨的问答助手。请根据以下检索到的上下文回答问题。
如无法确定，仍需给出最佳猜测，请直接输出答案（一个词组或短语），不要解释。

问题：{question}

已检索上下文：
{context}

答案："""

SELF_EVAL_PROMPT = """\
你是答案质量评审器。请根据问题、上下文和候选答案，给出 1 到 5 的整数分数：
- 1 = 明显错误或无关
- 2 = 大概率错误
- 3 = 部分正确/不完整
- 4 = 基本正确
- 5 = 高度正确且完整

只输出一个数字（1/2/3/4/5），不要输出其他文本。

问题：{question}

已检索上下文：
{context}

候选答案：{answer}

评分："""


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

    def generate_n(
        self,
        question: str,
        context: str,
        n: int,
        temperature: float,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        单次请求生成 n 个完成（Pass 1 语义熵 / 自一致性），与 Stage1 的 temperature、n 对齐。
        返回 (answers, meta)，meta 含 token_count（近似每完成一次）、latency_ms、answer_logprob。
        其中 answer_logprob 对应首个采样答案（通常即 Stage1 当前答案）的平均 token logprob。
        """
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        n = max(1, int(n))
        t0 = time.perf_counter()

        if self._mock:
            base = self._mock_generate(context)["answer"]
            rng = random.Random(hash(context) & 0xFFFFFFFF)
            out: List[str] = []
            parts = base.split()
            for _ in range(n):
                if len(parts) <= 2:
                    out.append(base)
                else:
                    k = max(1, int(len(parts) * rng.uniform(0.6, 1.0)))
                    shuffled = parts[:]
                    rng.shuffle(shuffled)
                    out.append(" ".join(shuffled[:k]))
            lat = (time.perf_counter() - t0) * 1000.0
            tc = max(1, sum(len(a.split()) for a in out) // n)
            return out, {"token_count": float(tc), "latency_ms": lat, "answer_logprob": -1.0}

        try:
            kwargs = dict(
                model=self.cfg.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.cfg.max_tokens,
                temperature=float(temperature),
                n=n,
                logprobs=True,
                top_logprobs=5,
            )
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as e_inner:
                logger.warning(
                    "批量 n=%d 请求失败（%s），先改无 logprobs 重试，再回退逐条采样。",
                    n,
                    e_inner,
                )
                try:
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                    response = self._client.chat.completions.create(**kwargs)
                except Exception:
                    texts = []
                    tok_sum = 0
                    for _ in range(n):
                        one = self._client.chat.completions.create(
                            model=self.cfg.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=self.cfg.max_tokens,
                            temperature=float(temperature),
                        )
                        texts.append((one.choices[0].message.content or "").strip())
                        u = getattr(one, "usage", None)
                        if u is not None and getattr(u, "completion_tokens", None):
                            tok_sum += int(u.completion_tokens)
                    lat = (time.perf_counter() - t0) * 1000.0
                    tc = tok_sum / max(1, n) if tok_sum else float(len(texts[0].split()) if texts else 1)
                    return texts, {"token_count": tc, "latency_ms": lat, "answer_logprob": 0.0}

            texts = [(c.message.content or "").strip() for c in response.choices]
            u = getattr(response, "usage", None)
            ct = int(getattr(u, "completion_tokens", 0) or 0) if u is not None else 0
            tc = (ct / float(n)) if ct > 0 else float(len(texts[0].split()) if texts else 1)
            first_logprob = self._extract_mean_logprob_from_choice(response.choices[0]) if texts else 0.0
            lat = (time.perf_counter() - t0) * 1000.0
            return texts, {"token_count": tc, "latency_ms": lat, "answer_logprob": first_logprob}
        except Exception as e:
            logger.error("generate_n 失败：%s，降级为 mock。", e)
            base = self._mock_generate(context)["answer"]
            rng = random.Random(42)
            parts = base.split()
            out: List[str] = []
            for _ in range(n):
                if len(parts) <= 2:
                    out.append(base)
                else:
                    k = max(1, int(len(parts) * rng.uniform(0.6, 1.0)))
                    shuffled = parts[:]
                    rng.shuffle(shuffled)
                    out.append(" ".join(shuffled[:k]))
            lat = (time.perf_counter() - t0) * 1000.0
            tc = max(1, sum(len(a.split()) for a in out) // n)
            return out, {"token_count": float(tc), "latency_ms": lat, "answer_logprob": -1.0}

    def self_evaluate_score(self, question: str, context: str, answer: str) -> float:
        """让模型对当前答案打 1~5 分；失败时返回 0。"""
        if self._mock:
            return 0.0
        prompt = SELF_EVAL_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.cfg.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            text = (response.choices[0].message.content or "").strip()
            m = None
            if text:
                import re

                m = re.search(r"([1-5])", text)
            if not m:
                return 0.0
            return float(int(m.group(1)))
        except Exception as e:
            logger.warning("self_evaluate_score 调用失败：%s", e)
            return 0.0

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_mean_logprob_from_choice(choice) -> float:
        lp_obj = getattr(choice, "logprobs", None)
        mean_lp, _, _ = LLMClient._extract_features(lp_obj)
        return float(mean_lp)

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
