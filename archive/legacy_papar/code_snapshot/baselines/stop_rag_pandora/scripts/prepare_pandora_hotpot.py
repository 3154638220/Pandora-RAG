import argparse
import json
import re
from pathlib import Path


def parse_doc(raw_doc: str) -> tuple[str, str]:
    match = re.match(r"^\[(.*?)\]\s*(.*)$", raw_doc.strip(), flags=re.S)
    if not match:
        return "", raw_doc.strip()
    return match.group(1).strip(), match.group(2).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_raw_item(item: dict) -> dict:
    sf = item.get("supporting_facts", {})
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    sent_ids = sf.get("sent_id", []) if isinstance(sf, dict) else []
    supporting_facts = [[title, sent_id] for title, sent_id in zip(titles, sent_ids)]
    return {
        "_id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "supporting_facts": supporting_facts,
    }


def build_corpus(all_items: list[dict]) -> list[dict]:
    corpus = {}
    for item in all_items:
        qid = item["id"]
        sf = item.get("supporting_facts", {})
        support_titles = set(sf.get("title", [])) if isinstance(sf, dict) else set()
        for idx, raw_doc in enumerate(item.get("documents", [])):
            title, text = parse_doc(raw_doc)
            if not title and not text:
                continue
            is_supporting = title in support_titles
            new_id = f"{qid}{'-sf' if is_supporting else ''}-{idx:02d}"
            key = (title, text)
            if key not in corpus:
                corpus[key] = {
                    "id": new_id,
                    "title": title,
                    "text": text,
                }
            else:
                corpus[key]["id"] += f"//{new_id}"
    return list(corpus.values())


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Stop-RAG HotpotQA assets from Pandora-RAG processed data.")
    parser.add_argument(
        "--pandora-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Path to the Pandora-RAG repository root.",
    )
    parser.add_argument(
        "--stop-rag-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the Stop-RAG repository root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_dir = args.pandora_root / "data" / "processed" / "hotpotqa"
    raw_dir = args.stop_rag_root / "data" / "raw" / "hotpotqa"
    corpus_path = args.stop_rag_root / "data" / "corpus" / "hotpotqa" / "passages" / "corpus.jsonl"

    split_map = {
        "train": processed_dir / "train.jsonl",
        "calib": processed_dir / "calib.jsonl",
        "dev": processed_dir / "dev.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    split_data = {name: load_jsonl(path) for name, path in split_map.items()}

    train_raw = [to_raw_item(item) for item in split_data["train"]]
    dev_pool_raw = [to_raw_item(item) for name in ("calib", "dev", "test") for item in split_data[name]]
    eval_raw = [to_raw_item(item) for item in split_data["dev"]]
    test_raw = [to_raw_item(item) for item in split_data["test"]]

    write_json(raw_dir / "hotpot_train_v1.1.json", train_raw)
    write_json(raw_dir / "hotpot_dev_fullwiki_v1.json", dev_pool_raw)
    write_json(raw_dir / "hotpot_dev_fullwiki_v1_eval_subsampled.json", eval_raw)
    write_json(raw_dir / "hotpot_dev_fullwiki_v1_test_subsampled.json", test_raw)

    all_items = [item for split in split_data.values() for item in split]
    corpus = build_corpus(all_items)
    write_jsonl(corpus_path, corpus)

    print(
        json.dumps(
            {
                "train_examples": len(train_raw),
                "dev_pool_examples": len(dev_pool_raw),
                "eval_examples": len(eval_raw),
                "test_examples": len(test_raw),
                "corpus_docs": len(corpus),
                "raw_dir": str(raw_dir),
                "corpus_path": str(corpus_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
