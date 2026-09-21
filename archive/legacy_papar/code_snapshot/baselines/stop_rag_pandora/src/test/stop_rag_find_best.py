import os
import glob
import argparse
import pandas as pd

pd.set_option("display.max_columns", None)


def pick_values(threshold, target_label):
    def wrapper(subdf):
        over = subdf[subdf["score1"] - subdf["score2"] > threshold]
        chosen = over.iloc[0] if not over.empty else subdf.iloc[-1]
        return chosen[[target_label]]

    return wrapper


# fmt: off
# pylint: disable=line-too-long
def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the input JSONL files")
    parser.add_argument("--thresholds", type=float, nargs='+', required=True, help="List of thresholds to apply for score selection")
    parser.add_argument("--target-label", type=str, default="f1", help="Target label to optimize for")

    return parser.parse_args()
# pylint: enable=line-too-long
# fmt: on


if __name__ == "__main__":
    args = parse_args()

    input_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    input_files.sort()
    scores = []

    for input_path in input_files:
        print(f"Reading data from {input_path}", flush=True)

        df = pd.read_json(input_path, lines=True)
        df = df.sort_values(["question_id", "iter_cnt"])

        for threshold in args.thresholds:
            result = (
                df.groupby("question_id", sort=False).apply(pick_values(threshold, args.target_label)).reset_index()
            )
            avg_label = result[args.target_label].mean()
            print(f"|  Average {args.target_label} score for threshold {threshold:.2f}: {avg_label:.4f}", flush=True)
            scores.append((input_path, threshold, avg_label))

    best_score = max(scores, key=lambda x: x[2])
    print(
        f"\n\nBest threshold: '{best_score[0]}' {best_score[1]:.2f} with {args.target_label} score {best_score[2]:.4f}"
    )
