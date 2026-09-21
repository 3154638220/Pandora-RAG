import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, set_seed
from ..train.stop_rag_train import MultiheadModel


# fmt: off
# pylint: disable=line-too-long
def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--max-length", type=int, default=2048, help="Max Length of Inputs")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision for model")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint of the multihead classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()
# pylint: enable=line-too-long
# fmt: on


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    # 勿用 device_map="auto"：可见多 GPU 时 encoder 会被切到多张卡，而 batch 只送到
    # model.device 对应的一张卡，触发 LayerNorm 等参数与激活不在同一设备的错误。
    score_dev = os.environ.get("STOP_RAG_SCORE_DEVICE")
    if score_dev:
        device = torch.device(score_dev)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = MultiheadModel.from_pretrained(
        args.checkpoint_path,
        encoder_kwargs={},
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        inference_mode=True,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model.config.encoder_name_or_path)
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.", flush=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    open(args.output_path, "w", encoding="utf-8").close()

    for i in tqdm(range(0, len(traces), args.batch_size)):
        batch_traces = traces[i : i + args.batch_size]

        batch_texts = [
            sep_token.join([trace["question"]] + [f"{doc['title']}: {doc['text']}" for doc in trace["history"]])
            for trace in batch_traces
        ]

        inputs = tokenizer(
            batch_texts, truncation=True, padding="longest", max_length=args.max_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores1 = outputs["preds_head1"].squeeze(-1).cpu().tolist()
            batch_scores2 = outputs["preds_head2"].squeeze(-1).cpu().tolist()

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, score1, score2 in zip(batch_traces, batch_scores1, batch_scores2):
                trace["score1"] = score1
                trace["score2"] = score2
                f.write(json.dumps(trace) + "\n")

    print(f"Scoring completed and results saved to {args.output_path}", flush=True)
