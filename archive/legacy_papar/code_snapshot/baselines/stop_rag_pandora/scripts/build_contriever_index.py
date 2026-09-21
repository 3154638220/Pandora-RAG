import argparse
import pickle
import sys
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.contriever.vector_index.faiss_index import FaissIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Faiss index from saved Contriever passage embeddings.")
    parser.add_argument("--embeddings-dir", required=True, help="Directory containing passage_* embedding shards.")
    parser.add_argument("--index-type", default="Flat", help="Faiss index type.")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_dir = Path(args.embeddings_dir)
    shard_paths = sorted(glob(str(embeddings_dir / "passages_*")))
    if not shard_paths:
        raise FileNotFoundError(f"No embedding shards found under {embeddings_dir}")

    with open(shard_paths[0], "rb") as f:
        _, first_embeddings = pickle.load(f)
    dim = int(first_embeddings.shape[1])

    index = FaissIndex(dim=dim, index_type=args.index_type)
    index.load_data(shard_paths)
    index.serialize(str(embeddings_dir))

    print(
        {
            "embeddings_dir": str(embeddings_dir),
            "num_shards": len(shard_paths),
            "dim": dim,
            "index_type": args.index_type,
        }
    )


if __name__ == "__main__":
    main()
