import os
import asyncio
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer

from .prompts import gen_stop_decision_prompt, gen_stop_decision_docs_only_prompt
from ..modules import AsyncOpenAIProcessor
from ...train.stop_rag_train import MultiheadModel


class StopDecider:
    def __init__(
        self,
        llm,
        max_gen_length=200,
        temperature=0.3,
        top_p=0.9,
        provider="vllm",
        use_docs_only=False,
        checkpoint_path=None,
        threshold=0.0,
        max_length=2048,
        bf16=False,
    ):
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        self.llm = llm

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.provider = provider

        if self.provider == "vllm" and llm is not None:
            self.tokenizer = llm.get_tokenizer()
        elif self.provider == "multihead":
            if not checkpoint_path:
                raise ValueError("checkpoint_path is required when provider='multihead'")

            # 勿用 device_map="auto"：与 vLLM 同可见 GPU 时 encoder 会占 cuda:0，与 LLM 显存冲突。
            # 由 STOP_RAG_SD_DEVICE 指定单卡（常与 STOP_RAG_CONTRIEVER_DEVICE 同为 TP 之后的空闲卡）。
            _sd_dev = (os.environ.get("STOP_RAG_SD_DEVICE") or "").strip()
            if _sd_dev:
                _dev = torch.device(_sd_dev)
                if _dev.type == "cuda" and _dev.index is not None:
                    _enc_map = {"": _dev.index}
                elif _dev.type == "cuda":
                    _enc_map = {"": 0}
                else:
                    _enc_map = {"": "cpu"}
            else:
                _enc_map = "auto"

            self.model = MultiheadModel.from_pretrained(
                checkpoint_path,
                encoder_kwargs={
                    "device_map": _enc_map,
                },
                dtype=torch.bfloat16 if bf16 else torch.float32,
                inference_mode=True,
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.encoder_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            self.threshold = threshold
            self.max_length = max_length

        self.use_docs_only = use_docs_only

        print(f"Stop Decider - {self.provider} initialized successfully.")

    def _process_prompts_vllm(self, prompts):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        outputs = self.llm.chat(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]

    async def _process_prompts_openai_async(self, prompts):
        async with AsyncOpenAIProcessor(self.llm) as processor:
            return await processor.process_prompts_async(
                prompts,
                max_gen_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def _batch_decide_multihead(
        self,
        questions: List[Dict[str, Any]],
        histories: List[List[Dict[str, Any]]],
        fields: Dict[str, str],
        log_trace: bool = False,
    ) -> List[str]:
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"
        batch_texts = [
            sep_token.join([question[fields["question"]]] + [f"{doc['title']}: {doc['text']}" for doc in history])
            for question, history in zip(questions, histories)
        ]

        # 与 Contriever/Reranker 同卡时，整批前向易触发 DeBERTa 注意力 OOM；按微批切分。
        micro = int(os.environ.get("STOP_RAG_SD_MICRO_BATCH", "4"))
        micro = max(1, micro)
        device = self.model.device
        scores1: List[float] = []
        scores2: List[float] = []
        for start in range(0, len(batch_texts), micro):
            chunk = batch_texts[start : start + micro]
            inputs = self.tokenizer(
                chunk,
                truncation=True,
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            s1 = outputs["preds_head1"].squeeze(-1).detach().flatten()
            s2 = outputs["preds_head2"].squeeze(-1).detach().flatten()
            scores1.extend(s1.tolist())
            scores2.extend(s2.tolist())

        decisions = []
        for question, score1, score2 in zip(questions, scores1, scores2):
            margin = score1 - score2
            decision = "STOP" if margin > self.threshold else "CONTINUE"
            if log_trace:
                print(
                    f"| MULTIHEAD STOP: {question[fields['id']]} margin={margin:.4f} "
                    f"threshold={self.threshold:.4f} => {decision}"
                )
            decisions.append(decision)

        return decisions

    def extract_decision(self, text, log_trace=False):
        text_upper = text.strip().upper()

        for line in text.strip().splitlines():
            if line.upper().startswith("DECISION:"):
                decision_part = line[9:].strip().upper()
                if "STOP" in decision_part:
                    if log_trace:
                        print("| STOP/CONTINUE: <STOP>")
                    return "STOP"
                elif "CONTINUE" in decision_part:
                    if log_trace:
                        print("| STOP/CONTINUE: <CONTINUE>")
                    return "CONTINUE"

        # Fallback
        if "STOP" in text_upper and "CONTINUE" not in text_upper:
            if log_trace:
                print("| STOP/CONTINUE: <STOP>")
            return "STOP"
        elif "CONTINUE" in text_upper:
            if log_trace:
                print("| STOP/CONTINUE: <CONTINUE>")
            return "CONTINUE"
        else:
            if log_trace:
                print("| STOP/CONTINUE: <DEFAULT CONTINUE>")
                print(f" DECISION: {text.strip()}")
            return "CONTINUE"

    def batch_decide(
        self,
        questions: List[Dict[str, Any]],
        traces: List[str],
        fields: Dict[str, str],
        log_trace: bool = False,
        histories: List[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if self.provider == "nostop":
            decisions = []
            for question in questions:
                if log_trace:
                    qid = question[fields["id"]]
                    print(f"| NOSTOP CONTINUE: {qid} - Always continue until max iterations")
                decision = "CONTINUE"
                decisions.append(decision)
            return decisions
        if self.provider == "multihead":
            if histories is None:
                raise ValueError("histories are required when provider='multihead'")
            return self._batch_decide_multihead(questions, histories, fields, log_trace)

        if self.use_docs_only:
            prompts = [
                gen_stop_decision_docs_only_prompt(question[fields["question"]], trace)
                for question, trace in zip(questions, traces)
            ]
        else:
            prompts = [
                gen_stop_decision_prompt(question[fields["question"]], trace)
                for question, trace in zip(questions, traces)
            ]

        if self.provider == "vllm":
            outputs = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            outputs = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        decisions = []
        for output in outputs:
            decision = self.extract_decision(output, log_trace)
            decisions.append(decision)

        return decisions


def test(llm, provider):
    stop_decider = StopDecider(
        llm=llm,
        max_gen_length=50,
        temperature=0.3,
        top_p=0.9,
        provider=provider,
    )

    questions = [{"question": "What county is the city where Peter Kern died in?"}]

    # continue
    # traces = ["Query: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany\nIntermediate answer: Peter Kern died in Knoxville, Tennessee."]

    # stop
    traces = [
        "Query: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany.\nIntermediate answer: Peter Kern died in Knoxville, Tennessee.\nQuery: In what county is Knoxville, Tennessee located?\nDocument: Knoxville is a city in the U.S. state of Tennessee, and the county seat of Knox County.\nIntermediate answer: Knoxville is located in Knox County."
    ]

    fields = {"question": "question"}

    decisions = stop_decider.batch_decide(questions, traces, fields)
    for question, trace, decision in zip(questions, traces, decisions):
        print(f"Question: {question['question']}")
        print(f"Trace: {trace}")
        print(f"Decision: {decision}")
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    # Test vLLM
    import torch
    from vllm import LLM

    from ...pandora_repo_defaults import resolve_default_vllm_model_id

    llm = LLM(
        model=resolve_default_vllm_model_id(),
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    test(llm, "vllm")

    # Test OpenAI
    from ..modules import OpenAIConfig

    llm = OpenAIConfig(
        model_id="gpt-4o-mini-2024-07-18",
        max_retries=1,
        timeout=60,
    )

    test(llm, "openai")
