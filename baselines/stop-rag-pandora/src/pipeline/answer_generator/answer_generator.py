import os
from typing import List, Dict, Any, Tuple
import asyncio

from ...pandora_repo_defaults import ensure_pandora_repo_root_on_path

ensure_pandora_repo_root_on_path()

from qa_shared.prompts import format_context_from_docs
from .prompts import gen_final_answer_prompt, gen_intermediate_answer_prompt
from ..modules import AsyncOpenAIProcessor


class AnswerGenerator:
    def __init__(self, llm, max_gen_length=400, temperature=0.7, top_p=0.9, provider="vllm"):
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        self.llm = llm

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.provider = provider

        print(f"Answer Generator - {self.provider} initialized successfully.")

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

    def batch_generate_intermediate_answers(self, queries: List[str], batch_docs: List[str]) -> List[str]:
        prompts = [
            gen_intermediate_answer_prompt(
                f"Question: {query}\n" + "\n".join(f"Document: {doc['title']}: {doc['text']}" for doc in docs)
            )
            for query, docs in zip(queries, batch_docs)
        ]

        if self.provider == "vllm":
            answers = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            answers = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return answers

    def batch_generate_final_answers(
        self, questions: List[Dict[str, Any]], batch_histories: List[List[Dict[str, Any]]], fields: Dict[str, str]
    ) -> List[str]:
        prompts = [
            gen_final_answer_prompt(
                f"Question: {question[fields['question']]}\nRetrieved context:\n{format_context_from_docs(history)}"
            )
            for question, history in zip(questions, batch_histories)
        ]

        if self.provider == "vllm":
            answers = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            answers = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return answers

    def generate_intermediate_answer(self, query: Dict[str, Any], doc: str) -> Tuple[str, str]:
        new_traces, answers = self.batch_generate_intermediate_answers([query], [doc])
        return new_traces[0], answers[0]


def test(llm, provider):
    answer_generator = AnswerGenerator(
        llm=llm,
        max_gen_length=200,
        temperature=0.7,
        top_p=0.9,
        provider=provider,
    )

    print("\n=== Testing Intermediate Answer Generation ===")
    queries = ["Where did Peter Kern die?"]
    docs = [
        "Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries."
    ]

    new_traces, answers = answer_generator.batch_generate_intermediate_answers(queries, docs)
    for query, doc, new_trace, answer in zip(queries, docs, new_traces, answers):
        print(f"Query: {query}")
        print("-" * 50)
        print(f"Document: {doc}")
        print("-" * 50)
        print(f"Generated Answer: {answer}")
        print("-" * 50)
        print(f"New Trace: {new_trace}")
        print("=" * 50)

    print("\nTest completed.")


if __name__ == "__main__":
    # Test vLLM
    import torch
    from vllm import LLM

    from ..pandora_repo_defaults import resolve_default_vllm_model_id

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
    from ..modules import AsyncOpenAIConfig

    llm = AsyncOpenAIConfig(
        model_id="gpt-4o-mini-2024-07-18",
        max_retries=1,
        timeout=60,
    )

    test(llm, "openai")
