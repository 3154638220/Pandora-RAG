import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from vllm import LLM, SamplingParams
from ...pipeline.utils import compute_all_answer_metrics
from ...pandora_repo_defaults import resolve_default_vllm_model_id


def _build_docs_question_suffix(trace: dict) -> str:
    return (
        "\n\n"
        + "\n\n".join(
            [f"Wikipedia Title: {doc['title']}\n{doc['text']}" for doc in trace["history"]]
        )
        + "\n\n"
        + "Q: "
        + trace["question"]
        + "\n"
        + "A: "
    )


def _truncate_prompt_shared_icl_prefix(tokenizer, icl_text: str, trace: dict, max_input_tokens: int) -> str:
    """ICL 取头部（与 README 对齐、且便于 vLLM prefix cache）；suffix 为完整 history+Q/A前缀。

    超长时优先裁掉 ICL 尾部；仍超长则裁掉文档串头部，保留末尾的 Q:/A:。
    """
    if max_input_tokens <= 0:
        return icl_text + _build_docs_question_suffix(trace)
    suffix = _build_docs_question_suffix(trace)
    icl_ids = tokenizer.encode(icl_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(suffix_ids) >= max_input_tokens:
        combined = suffix_ids[-max_input_tokens:]
    elif len(icl_ids) + len(suffix_ids) <= max_input_tokens:
        combined = icl_ids + suffix_ids
    else:
        icl_budget = max_input_tokens - len(suffix_ids)
        combined = (icl_ids[:icl_budget] if icl_budget > 0 else []) + suffix_ids
    return tokenizer.decode(combined, skip_special_tokens=True)


def extract_answer(output):
    if "answer is: " in output.lower():
        idx = output.lower().find("answer is: ")
        return output[idx + len("answer is: ") :].split("\n")[0].strip()
    else:
        return output


# fmt: off
# pylint: disable=line-too-long
def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Resume: skip the first N input traces and append metrics to output (output must already contain exactly N lines)",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for processing")
    parser.add_argument("--repeat-size", type=int, default=1, help="Number of times to repeat each trace")
    parser.add_argument("--icl-examples-path", type=str, required=True, help="Path to ICL examples (Required for docs-only mode)")

    vllm_group = parser.add_argument_group("vLLM Options")
    vllm_group.add_argument(
        "--vllm-model-id",
        type=str,
        default=resolve_default_vllm_model_id(),
        help="Model ID or local path for vLLM",
    )
    vllm_group.add_argument("--vllm-tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    vllm_group.add_argument("--vllm-quantization", type=str, help="Quantization method for vLLM")
    vllm_group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    vllm_group.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=16384,
        help="Maximum model length for vLLM (ICL+history can be very long)",
    )
    vllm_group.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture (avoids OOM during engine init on tight VRAM)",
    )

    final_answer_generator_group = parser.add_argument_group("Final Answer Generator Options")
    final_answer_generator_group.add_argument("--fag-max-gen-length", type=int, default=400, help="Maximum generation length for answer generator")
    final_answer_generator_group.add_argument("--fag-temperature", type=float, default=0.0, help="Temperature for answer generator")
    final_answer_generator_group.add_argument("--fag-top-p", type=float, default=1.0, help="Top-p sampling for answer generator")

    return parser.parse_args()
# pylint: enable=line-too-long
# fmt: on


if __name__ == "__main__":
    args = parse_args()

    model = LLM(
        model=args.vllm_model_id,
        tensor_parallel_size=args.vllm_tp_size,
        quantization=args.vllm_quantization,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.fag_max_gen_length,
        temperature=args.fag_temperature,
        top_p=args.fag_top_p,
    )

    print("Model and tokenizer loaded successfully.", flush=True)
    _tokenizer = model.get_tokenizer()
    _max_input_tokens = args.vllm_max_model_len - args.fag_max_gen_length - 32
    if _max_input_tokens < 512:
        _max_input_tokens = 512

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    with open(args.icl_examples_path, "r", encoding="utf-8") as f:
        icl_examples = f.read()
        icl_examples = "\n".join([line for line in icl_examples.split("\n") if not line.startswith("# METADATA")])

    if args.skip_first:
        if args.skip_first < 0 or args.skip_first > len(traces):
            raise ValueError(f"--skip-first {args.skip_first} invalid for input size {len(traces)}")
        if not os.path.isfile(args.output_path):
            raise ValueError("--skip-first requires an existing output_path to append to")
        with open(args.output_path, "r", encoding="utf-8") as f:
            done_lines = sum(1 for _ in f)
        if done_lines != args.skip_first:
            raise ValueError(
                f"resume mismatch: output has {done_lines} non-empty lines, --skip-first {args.skip_first}"
            )
        traces = traces[args.skip_first:]
    else:
        open(args.output_path, "w", encoding="utf-8").close()

    print(f"Number of prompts: {len(traces)}", flush=True)

    assert args.batch_size % args.repeat_size == 0
    effective_batch_size = args.batch_size // args.repeat_size

    for i in tqdm(range(0, len(traces), effective_batch_size)):
        batch_traces = traces[i : i + effective_batch_size]
        batch_traces_repeated = [trace for trace in batch_traces for _ in range(args.repeat_size)]

        batch_answers_repeated = [
            {
                "answer": trace["answers"][0],
                "answer_aliases": trace["answers"][1:],
            }
            for trace in batch_traces_repeated
        ]

        batch_prompts_repeated = [
            _truncate_prompt_shared_icl_prefix(_tokenizer, icl_examples, trace, _max_input_tokens)
            for trace in batch_traces_repeated
        ]
        outputs = model.generate(batch_prompts_repeated, sampling_params, use_tqdm=False)
        batch_predictions_repeated = [extract_answer(output.outputs[0].text.strip()) for output in outputs]

        fields = {
            "answer": "answer",
            "answer_aliases": "answer_aliases",
        }

        em_list, f1_list, acc_list = compute_all_answer_metrics(
            batch_answers_repeated, batch_predictions_repeated, fields
        )

        avg_em_list = np.mean(np.array(em_list).reshape(-1, args.repeat_size), axis=1)
        avg_f1_list = np.mean(np.array(f1_list).reshape(-1, args.repeat_size), axis=1)
        avg_acc_list = np.mean(np.array(acc_list).reshape(-1, args.repeat_size), axis=1)

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, em, f1, acc in zip(batch_traces, avg_em_list, avg_f1_list, avg_acc_list):
                trace["em"] = em
                trace["f1"] = f1
                trace["acc"] = acc
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Processing complete. Output written to {args.output_path}")
