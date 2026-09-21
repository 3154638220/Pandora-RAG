import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Stop-RAG raw data and corpora from Pandora-RAG splits using locally cached HF datasets."
    )
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
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,musique,2wiki",
        help="Comma-separated Pandora dataset names to prepare.",
    )
    return parser.parse_args()


def configure_offline_hf(pandora_root: Path) -> None:
    hf_home = pandora_root / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_doc(raw_doc: str) -> tuple[str, str]:
    match = re.match(r"^\[(.*?)\]\s*(.*)$", raw_doc.strip(), flags=re.S)
    if not match:
        return "", raw_doc.strip()
    return match.group(1).strip(), match.group(2).strip()


def build_split_ids(processed_dir: Path) -> dict[str, list[str]]:
    split_ids = {}
    for split in ("train", "calib", "dev", "test"):
        rows = load_jsonl(processed_dir / f"{split}.jsonl")
        split_ids[split] = [row["id"] for row in rows]
    return split_ids


def load_source_dataset(dataset_name: str) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    if dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor")
    elif dataset_name == "musique":
        ds = load_dataset("dgslibisey/MuSiQue")
    elif dataset_name == "2wiki":
        ds = load_dataset("framolfese/2WikiMultihopQA")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loaded: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_ds in ds.items():
        loaded[split_name] = [dict(row) for row in split_ds]
    return loaded


def index_examples(dataset_name: str, source_splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    idx = {}
    id_field = "id"
    for split_rows in source_splits.values():
        for row in split_rows:
            row_id = row[id_field]
            idx[row_id] = row
    return idx


def convert_hotpot_raw(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "type": item.get("type"),
        "level": item.get("level"),
        "supporting_facts": item["supporting_facts"],
        "context": item["context"],
    }


def convert_2wiki_raw(item: dict[str, Any]) -> dict[str, Any]:
    context = []
    titles = item["context"]["title"]
    sentences = item["context"]["sentences"]
    for title, sents in zip(titles, sentences):
        context.append([title, sents])
    sf = item["supporting_facts"]
    supporting_facts = [[title, sent_id] for title, sent_id in zip(sf["title"], sf["sent_id"])]
    return {
        "_id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "type": item.get("type"),
        "evidences": item.get("evidences", []),
        "supporting_facts": supporting_facts,
        "context": context,
    }


def convert_musique_raw(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "paragraphs": item["paragraphs"],
        "question": item["question"],
        "question_decomposition": item["question_decomposition"],
        "answer": item["answer"],
        "answer_aliases": item.get("answer_aliases", []),
        "answerable": item.get("answerable", True),
    }


def build_corpus_hotpot(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        qid = item["id"]
        support_titles = set(item["supporting_facts"]["title"])
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        for idx, (title, sents) in enumerate(zip(titles, sentences)):
            text = " ".join(sents).strip()
            if not title and not text:
                continue
            new_id = f"{qid}{'-sf' if title in support_titles else ''}-{idx:02d}"
            key = (title, text)
            if key not in corpus:
                corpus[key] = {"id": new_id, "title": title, "text": text}
            else:
                corpus[key]["id"] += f"//{new_id}"
    return list(corpus.values())


def build_corpus_2wiki(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        qid = item["id"]
        support_titles = set(item["supporting_facts"]["title"])
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        for idx, (title, sents) in enumerate(zip(titles, sentences)):
            text = " ".join(sents).strip()
            if not title and not text:
                continue
            new_id = f"{qid}{'-sf' if title in support_titles else ''}-{idx:02d}"
            key = (title, text)
            if key not in corpus:
                corpus[key] = {"id": new_id, "title": title, "text": text}
            else:
                corpus[key]["id"] += f"//{new_id}"
    return list(corpus.values())


def build_corpus_musique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        qid = item["id"]
        for paragraph in item["paragraphs"]:
            idx = int(paragraph["idx"])
            title = paragraph["title"]
            text = paragraph["paragraph_text"].strip()
            new_id = f"{qid}{'-sf' if paragraph.get('is_supporting', False) else ''}-{idx:02d}"
            key = (title, text)
            if key not in corpus:
                corpus[key] = {"id": new_id, "title": title, "text": text}
            else:
                corpus[key]["id"] += f"//{new_id}"
    return list(corpus.values())


def ensure_examples(split_ids: dict[str, list[str]], id_to_example: dict[str, dict[str, Any]], dataset_name: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    missing: dict[str, list[str]] = defaultdict(list)
    for split, ids in split_ids.items():
        rows = []
        for item_id in ids:
            example = id_to_example.get(item_id)
            if example is None:
                missing[split].append(item_id)
            else:
                rows.append(example)
        out[split] = rows
    if missing:
        details = {split: values[:5] for split, values in missing.items()}
        raise KeyError(f"Missing source examples for {dataset_name}: {details}")
    return out


def prepare_dataset(dataset_name: str, pandora_root: Path, stop_rag_root: Path) -> dict[str, Any]:
    processed_name = dataset_name
    stop_rag_name = "2wikimultihopqa" if dataset_name == "2wiki" else dataset_name
    processed_dir = pandora_root / "data" / "processed" / processed_name
    split_ids = build_split_ids(processed_dir)
    source_splits = load_source_dataset(dataset_name)
    id_to_example = index_examples(dataset_name, source_splits)
    selected = ensure_examples(split_ids, id_to_example, dataset_name)

    all_selected = selected["train"] + selected["calib"] + selected["dev"] + selected["test"]
    dev_pool = selected["calib"] + selected["dev"] + selected["test"]

    raw_dir = stop_rag_root / "data" / "raw" / stop_rag_name
    corpus_path = stop_rag_root / "data" / "corpus" / stop_rag_name / "passages" / "corpus.jsonl"

    if dataset_name == "hotpotqa":
        write_json(raw_dir / "hotpot_train_v1.1.json", [convert_hotpot_raw(x) for x in selected["train"]])
        write_json(raw_dir / "hotpot_dev_fullwiki_v1.json", [convert_hotpot_raw(x) for x in dev_pool])
        write_json(raw_dir / "hotpot_dev_fullwiki_v1_eval_subsampled.json", [convert_hotpot_raw(x) for x in selected["dev"]])
        write_json(raw_dir / "hotpot_dev_fullwiki_v1_test_subsampled.json", [convert_hotpot_raw(x) for x in selected["test"]])
        corpus = build_corpus_hotpot(all_selected)
    elif dataset_name == "2wiki":
        write_json(raw_dir / "train.json", [convert_2wiki_raw(x) for x in selected["train"]])
        write_json(raw_dir / "dev.json", [convert_2wiki_raw(x) for x in dev_pool])
        write_json(raw_dir / "test.json", [convert_2wiki_raw(x) for x in selected["test"]])
        write_json(raw_dir / "dev_eval_subsampled.json", [convert_2wiki_raw(x) for x in selected["dev"]])
        write_json(raw_dir / "dev_test_subsampled.json", [convert_2wiki_raw(x) for x in selected["test"]])
        corpus = build_corpus_2wiki(all_selected)
    elif dataset_name == "musique":
        write_jsonl(raw_dir / "musique_ans_v1.0_train.jsonl", [convert_musique_raw(x) for x in selected["train"]])
        write_jsonl(raw_dir / "musique_ans_v1.0_dev.jsonl", [convert_musique_raw(x) for x in dev_pool])
        write_jsonl(raw_dir / "musique_ans_v1.0_test.jsonl", [convert_musique_raw(x) for x in selected["test"]])
        write_jsonl(raw_dir / "musique_ans_v1.0_dev_eval_subsampled.jsonl", [convert_musique_raw(x) for x in selected["dev"]])
        write_jsonl(raw_dir / "musique_ans_v1.0_dev_test_subsampled.jsonl", [convert_musique_raw(x) for x in selected["test"]])
        corpus = build_corpus_musique(all_selected)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    write_jsonl(corpus_path, corpus)
    return {
        "dataset": dataset_name,
        "stop_rag_dataset": stop_rag_name,
        "train_examples": len(selected["train"]),
        "dev_pool_examples": len(dev_pool),
        "eval_examples": len(selected["dev"]),
        "test_examples": len(selected["test"]),
        "corpus_docs": len(corpus),
        "raw_dir": str(raw_dir),
        "corpus_path": str(corpus_path),
    }


def main() -> None:
    args = parse_args()
    configure_offline_hf(args.pandora_root)

    summaries = []
    for dataset_name in [part.strip() for part in args.datasets.split(",") if part.strip()]:
        summaries.append(prepare_dataset(dataset_name, args.pandora_root, args.stop_rag_root))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
