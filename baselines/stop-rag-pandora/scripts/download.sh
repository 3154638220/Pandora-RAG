#!/usr/bin/env bash

set -eu

RETRIEVER="${1:-}"

if [ "$RETRIEVER" != "contriever" ] && [ "$RETRIEVER" != "bm25" ]; then
    echo "Invalid retriever. Use 'contriever' or 'bm25'." >&2
    exit 1
fi

PYTHON_BIN="${STOP_RAG_PYTHON:-python}"
CONTRIEVER_MODEL_PATH="${STOP_RAG_CONTRIEVER_MODEL:-facebook/contriever-msmarco}"

### download datasets
mkdir -p .temp
mkdir -p data/raw

# HotpotQA
if [ ! -f "data/processed/hotpotqa/eval.json" ] || [ ! -f "data/processed/hotpotqa/test.json" ]; then
    echo "Downloading raw HotpotQA data"
    mkdir -p data/raw/hotpotqa
    if [ ! -f "data/raw/hotpotqa/hotpot_train_v1.1.json" ]; then
        wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/raw/hotpotqa/hotpot_train_v1.1.json
    fi
    if [ ! -f "data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json" ]; then
        wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json -O data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json
    fi
    "${PYTHON_BIN}" -m src.download.subsample_data --dataset hotpotqa --num-eval 1000 --num-test 1000
else
    echo "HotpotQA dataset already processed, skipping..."
fi

# 2WikiMultihopQA
if [ ! -f "data/processed/2wikimultihopqa/eval.json" ] || [ ! -f "data/processed/2wikimultihopqa/test.json" ]; then
    echo "Downloading raw 2WikiMultihopQA data"
    mkdir -p data/raw/2wikimultihopqa
    if [ ! -f ".temp/2wikimultihopqa.zip" ]; then
        wget https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0 -O .temp/2wikimultihopqa.zip
    fi
    unzip -jo .temp/2wikimultihopqa.zip -d data/raw/2wikimultihopqa -x "*.DS_Store"
    "${PYTHON_BIN}" -m src.download.subsample_data --dataset 2wikimultihopqa --num-eval 1000 --num-test 1000
else
    echo "2WikiMultihopQA dataset already processed, skipping..."
fi

# MuSiQue
if [ ! -f "data/processed/musique/eval.json" ] || [ ! -f "data/processed/musique/test.json" ]; then
    echo "Downloading raw MuSiQue data"
    mkdir -p data/raw/musique
    if [ ! -f ".temp/musique_data_v1.0.zip" ]; then
        # URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
        "${PYTHON_BIN}" -m gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" -O .temp/musique_data_v1.0.zip
    fi
    unzip -jo .temp/musique_data_v1.0.zip -d data/raw/musique -x "*.DS_Store"
    "${PYTHON_BIN}" -m src.download.subsample_data --dataset musique --num-eval 1000 --num-test 1000
else
    echo "MuSiQue dataset already processed, skipping..."
fi

rm -rf .temp/


### build corpus
mkdir -p .temp

# HotpotQA 
if [ ! -f "data/corpus/hotpotqa/passages/corpus.jsonl" ]; then
    echo "Building HotpotQA corpus"
    mkdir -p data/corpus/hotpotqa/passages
    if [ ! -f ".temp/hotpot_wikipedia.tar.bz2" ]; then
        echo "Downloading HotpotQA Wikipedia corpus (this will take ~5 mins)"
        wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -O .temp/hotpot_wikipedia.tar.bz2
    fi
    "${PYTHON_BIN}" -m src.download.build_corpus --dataset hotpotqa --input-path .temp/hotpot_wikipedia.tar.bz2
else
    echo "HotpotQA corpus already exists, skipping..."
fi

# 2WikiMultiHopQA 
if [ ! -f "data/corpus/2wikimultihopqa/passages/corpus.jsonl" ]; then
    echo "Building 2WikiMultiHopQA corpus"
    mkdir -p data/corpus/2wikimultihopqa/passages
    "${PYTHON_BIN}" -m src.download.build_corpus --dataset 2wikimultihopqa
else
    echo "2WikiMultiHopQA corpus already exists, skipping..."
fi

# MuSiQue
if [ ! -f "data/corpus/musique/passages/corpus.jsonl" ]; then
    echo "Building MuSiQue corpus"
    mkdir -p data/corpus/musique/passages
    "${PYTHON_BIN}" -m src.download.build_corpus --dataset musique
else
    echo "MuSiQue corpus already exists, skipping..."
fi

rm -rf .temp/

### build index

if [ "$RETRIEVER" = "contriever" ]; then
    for ds in hotpotqa 2wikimultihopqa musique; do
        if [ ! -d "data/corpus/$ds/contriever_embeddings" ] || [ ! -f "data/corpus/$ds/contriever_embeddings/index.faiss" ]; then
            echo "Building Contriever index for $ds"
            "${PYTHON_BIN}" src/pipeline/contriever/passage_embedder.py \
                --passages data/corpus/$ds/passages/corpus.jsonl \
                --output_dir data/corpus/$ds/contriever_embeddings \
                --model_name_or_path "$CONTRIEVER_MODEL_PATH"

            "${PYTHON_BIN}" -m src.pipeline.contriever.passage_retriever \
                --passages data/corpus/$ds/passages/corpus.jsonl \
                --embeddings data/corpus/$ds/contriever_embeddings \
                --save_or_load_index
        else
            echo "Contriever index for $ds already exists, skipping..."
        fi
    done
elif [ "$RETRIEVER" = "bm25" ]; then
    for ds in hotpotqa 2wikimultihopqa musique; do
        if [ ! -d "data/corpus/$ds/bm25_index" ] || [ -z "$(ls -A data/corpus/$ds/bm25_index 2>/dev/null)" ]; then
            echo "Building BM25 index for $ds"
            "${PYTHON_BIN}" -m src.pipeline.bm25.bm25_retriever \
                --passages data/corpus/$ds/passages/corpus.jsonl \
                --index_path_dir data/corpus/$ds/bm25_index \
                --save_or_load_index
        else
            echo "BM25 index for $ds already exists, skipping..."
        fi
    done
fi

echo "\nIndexing completed for retriever: $RETRIEVER"
