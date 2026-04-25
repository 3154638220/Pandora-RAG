# Experimental Setup

> Snapshot date: 2026-04-20  
> Repository commit observed in this workspace: `53677b0`  
> Main experimental record: `pdopt_best` Stage-2 probe + Stage-3 `Probe+E-value` with `gamma=0.5`, `alpha=0.1`, predictive betting.

This note records the experimental setup used for Pandora-RAG. It is intended as a paper-writing source of truth: the first sections can be adapted into the paper's Experimental Setup, while the later sections keep reproducibility details and caveats.

## 1. Task and Datasets

We evaluate Pandora-RAG on multi-hop question answering benchmarks that require iterative evidence acquisition:


| Dataset         | Hop type                  | Train | Calib | Dev  | Test | Split seed |
| --------------- | ------------------------- | ----- | ----- | ---- | ---- | ---------- |
| HotpotQA        | 2-hop distractor QA       | 4000  | 1000  | 1000 | 1000 | 42         |
| MuSiQue         | 2-4 hop QA                | 4000  | 1000  | 1000 | 417  | 42         |
| 2WikiMultiHopQA | multi-entity multi-hop QA | 4000  | 1000  | 1000 | 1000 | 42         |


The four-way split is deterministic and sample-id-disjoint. `calib` is reserved for Stage-3 quality-model calibration, E-value/CP calibration, and betting-related thresholds; `dev` is used for Stage-2 stopping-threshold selection; `test` is used only for final evaluation. When no usable gold test split is available, the code slices test examples from the tail of validation after reserving `calib` and `dev`. This is why MuSiQue has only 417 test examples under the default quota.

Split manifests:

- `data/splits/hotpotqa_seed42_manifest.json`
- `data/splits/musique_seed42_manifest.json`
- `data/splits/2wiki_seed42_manifest.json`

## 2. RAG Backbone and Trajectory Collection

Each example is run through a fixed-order iterative RAG process with maximum retrieval depth `K=5`. For every step `k`, the pipeline records:

- retrieved document and retrieval score;
- generated intermediate/current answer;
- answer quality against the gold answer, reported as F1 and Exact Match;
- shallow step features such as retrieval score, answer log-probability, self-evaluation score, context overlap, NLI entailment/contradiction, history deltas, answer-change features, and cost proxies;
- LLM hidden states for the current cumulative state.

The base generator/state model is `meta-llama/Meta-Llama-3.1-8B-Instruct`. Generation is deterministic with `temperature=0.0` and `n=1`. The implementation serves generation through a vLLM OpenAI-compatible endpoint and extracts hidden states with HuggingFace Transformers.

The main hidden-state representation is the last layer `last_token` vector with dimension 4096. Hidden states are stored per example and per retrieval step as:

```text
cache/features/{dataset}/{split}/hidden_states/{id}_step{k}.npz
```

The per-step representation is important: the state after step `k` contains the question, all documents retrieved up to `k`, and the current answer at `k`. A single final-state hidden vector is not considered valid for the main experiments because it is not time-aligned with step-level oracle labels.

### Retriever

Stage 1 supports two retrieval backends:

- `bm25`, the implementation default;
- `contriever_bge`, using `facebook/contriever-msmarco` for dense shortlisting and `BAAI/bge-reranker-v2-m3` for cross-encoder reranking.

When reporting a run, the retriever backend should be stated explicitly. Changing the retriever backend requires clearing `cache/trajectories` and `cache/features` before rerunning Stage 1, otherwise trajectories, hidden states, and oracle labels can mix incompatible retrieval settings.

## 3. Stage 1: Oracle Labels and Structural Baselines

Stage 1 forces each trajectory to run up to `K=5`, then constructs oracle stopping labels from the observed per-step qualities. The default per-step cost is:

```text
cost_per_step = 0.05
oracle_cost_metric = fixed
```

The code also supports `token` and `latency` cost metrics, but the current main result record uses the fixed-cost setting unless otherwise stated.

For an example `i`, let `Q_k^i` be the answer quality after step `k` and `c_k^i` the cost of step `k`. The instance-level DP oracle is:

```text
V_K^i = Q_K^i
V_k^i = max(Q_k^i, V_{k+1}^i - c_{k+1}^i)
```

The oracle stops at the first step where stopping is at least as valuable as continuing:

```text
Q_k^i >= V_{k+1}^i - c_{k+1}^i
```

Stage 1 writes step supervision labels:

- `expected_continue_val = V_{k+1} - c_{k+1}`;
- `margin = expected_continue_val - Q_k`;
- `action_label = 1[margin > 0]`, where 1 means continue and 0 means stop.

We also compute a global Weitzman-style baseline by estimating per-step gain distributions on the training split and applying query-independent reservation values at test time. This is a structured static baseline, not the deployed method.

Stage-1 oracle upper bounds on the test set:


| Dataset         | Oracle F1 | Avg steps |
| --------------- | --------- | --------- |
| HotpotQA        | 0.7810    | 1.58      |
| MuSiQue         | 0.4966    | 2.12      |
| 2WikiMultiHopQA | 0.6954    | 1.59      |


## 4. Stage 2: Learned Stopping Probe

Stage 2 trains a lightweight stopping probe from Stage-1 step labels. The deployed probe predicts a continue probability:

```text
p_theta(continue | s_k)
```

and stops when this probability falls below the learned per-step threshold. The main model is `ProbeMLP_v2` with two branches:

- hidden branch: LayerNorm over the 4096-dimensional `last_token` hidden state, followed by linear compression;
- shallow branch: MLP over the shallow feature vector;
- fusion head: concatenated hidden/shallow representations mapped to a continue logit.

The final `pdopt_best` checkpoint uses binary Continue/Stop supervision rather than F1 regression, because binary supervision performed better on all three datasets.

### Stage-2 Hyperparameters

Common training hyperparameters:


| Hyperparameter          | Value                         |
| ----------------------- | ----------------------------- |
| optimizer learning rate | `3e-4`                        |
| weight decay            | `5e-4`                        |
| dropout                 | `0.30`                        |
| batch size              | `256`                         |
| epochs                  | `60`                          |
| early-stopping patience | `12`                          |
| warmup epochs           | `5`                           |
| focal gamma             | `2.0`                         |
| label smoothing         | `0.05`                        |
| fuse dimension          | `128`                         |
| Stage-2 threshold cap   | `Global-Weitzman(dev) * 1.05` |


Archived `pdopt_best` checkpoint metadata observed in this workspace:


| Dataset         | Hidden compression | Shallow input    | Margin filter | Residual hidden branch | Target          |
| --------------- | ------------------ | ---------------- | ------------- | ---------------------- | --------------- |
| HotpotQA        | 64                 | 20 main features | 0.00          | no                     | binary Continue |
| MuSiQue         | 64                 | 20 main features | 0.00          | no                     | binary Continue |
| 2WikiMultiHopQA | 64                 | 20 main features | 0.02          | no                     | binary Continue |


The code can construct a 31-dimensional shallow feature vector, but the `p2_full31` retraining did not improve the main operating point and is retained as a diagnostic/negative result rather than the main setup.

Implementation note: `stage2/run_stage2.py` currently defines a `PER_DATASET_OPTIMAL` retraining table with larger compression for HotpotQA/MuSiQue and a residual branch for 2Wiki. The archived `pdopt_best` `.pt` files in this workspace report the metadata in the table above. For paper numbers tied to `docs/stage2_report_pdopt_best.md` and `results/stage2_probe_table_*_pdopt_best.csv`, cite the archived `pdopt_best` artifact metadata rather than assuming a fresh rerun's code defaults.

### Threshold Selection

Thresholds are selected on the `dev` split. Candidate operating points must satisfy the Global-Weitzman dev-step budget cap. Among feasible points, the code maximizes a cost-aware utility of the form:

```text
F1 - lambda * normalized_cost
```

with conservative per-step threshold refinement accepted only when it improves utility, matched-budget F1, or the Pareto frontier.

Final per-step thresholds:


| Dataset         | Per-step thresholds              |
| --------------- | -------------------------------- |
| HotpotQA        | `[0.73, 0.69, 0.81, 0.73, 0.73]` |
| MuSiQue         | `[0.61, 0.63, 0.63, 0.69, 0.63]` |
| 2WikiMultiHopQA | `[0.61, 0.79, 0.73, 0.77, 0.67]` |


### Stage-2 Test Results


| Strategy             | HotpotQA F1 / steps | MuSiQue F1 / steps | 2Wiki F1 / steps |
| -------------------- | ------------------- | ------------------ | ---------------- |
| Fixed-K=1            | 0.4340 / 1.00       | 0.0797 / 1.00      | 0.2846 / 1.00    |
| Fixed-K=2            | 0.6773 / 2.00       | 0.1830 / 2.00      | 0.5908 / 2.00    |
| Fixed-K=3            | 0.6775 / 3.00       | 0.3210 / 3.00      | 0.5583 / 3.00    |
| Fixed-K=4            | 0.6747 / 4.00       | 0.3816 / 4.00      | 0.5382 / 4.00    |
| Fixed-K=5            | 0.6668 / 4.99       | 0.4022 / 5.00      | 0.5288 / 5.00    |
| Global-Weitzman      | 0.7092 / 1.64       | 0.4229 / 3.29      | 0.6449 / 1.77    |
| Oracle               | 0.7810 / 1.58       | 0.4966 / 2.12      | 0.6954 / 1.59    |
| Probe (`pdopt_best`) | 0.6544 / 1.73       | 0.3969 / 3.31      | 0.5941 / 1.82    |


Recommended wording: the probe recovers most of the oracle stopping benefit at a low retrieval budget, but it does not uniformly beat the best fixed depth on every dataset.

## 5. Stage 3: E-value Risk Monitoring

Stage 3 adds an online risk-monitoring layer to the Stage-2 probe. The main setting is:


| Parameter                  | Value                                                        |
| -------------------------- | ------------------------------------------------------------ |
| quality threshold `gamma`  | `0.5`                                                        |
| alpha grid                 | `{0.1, 0.2}`                                                 |
| main-table alpha           | `0.1`                                                        |
| betting strategy           | `predictive`                                                 |
| outcome-aware update       | `true`                                                       |
| quality-model calibration  | `quantile`                                                   |
| quality model              | `StandardScaler + LogisticRegression(class_weight=balanced)` |
| quality model random state | `42`                                                         |
| quality model max iter     | `300`                                                        |


The error event is:

```text
e_n = 1[F1(s_tau_n) < gamma]
```

For outcome-aware E-values, wealth is updated by:

```text
E_n = E_{n-1} * (1 - lambda_n + lambda_n * e_n / alpha)
lambda_n = clip(1 - p_hat_n, eps, 1 - eps)
```

where `p_hat_n` is the quality model's predicted probability that the stopped answer has `F1 >= gamma`. The monitoring boundary is `1 / alpha`.

Important paper caveat: E-value provides anytime-valid evidence for detecting elevated risk under adaptive stopping. It is not a mechanism that guarantees the empirical error rate is always below `alpha`.

### Stage-3 Main Results (`gamma=0.5`, `alpha=0.1`)


| Dataset         | Method        | F1     | EM     | Error rate | Avg steps |
| --------------- | ------------- | ------ | ------ | ---------- | --------- |
| HotpotQA        | Probe         | 0.6569 | 0.5190 | 0.3030     | 1.70      |
| HotpotQA        | Probe+E-value | 0.6654 | 0.5290 | 0.2950     | 1.81      |
| HotpotQA        | Probe+CP      | 0.6716 | 0.5310 | 0.2960     | 4.11      |
| MuSiQue         | Probe         | 0.4152 | 0.3189 | 0.5803     | 3.39      |
| MuSiQue         | Probe+E-value | 0.4127 | 0.3141 | 0.5827     | 3.41      |
| MuSiQue         | Probe+CP      | 0.4015 | 0.3046 | 0.5971     | 4.85      |
| 2WikiMultiHopQA | Probe         | 0.5639 | 0.4740 | 0.4090     | 1.82      |
| 2WikiMultiHopQA | Probe+E-value | 0.5728 | 0.4810 | 0.4010     | 1.91      |
| 2WikiMultiHopQA | Probe+CP      | 0.5383 | 0.4340 | 0.4370     | 4.32      |


Final E-wealth under no-shift main evaluation:


| Dataset         | Final E-wealth at alpha=0.1 | Cap | Interpretation                                  |
| --------------- | --------------------------- | --- | ----------------------------------------------- |
| HotpotQA        | 0.014                       | 10  | no alarm; gate mainly improves quality slightly |
| MuSiQue         | 4.193                       | 10  | risk evidence accumulates but does not hit cap  |
| 2WikiMultiHopQA | 10.000                      | 10  | cap is reached, rejecting the low-error null    |


## 6. Distribution Shift and Selective Prediction

Stage 3 evaluates three test-stream shifts:

- `sudden`: the stream switches from easier to harder examples after `shift_fraction=0.5`;
- `gradual`: difficulty increases gradually after the shift point;
- `periodic`: easy and hard examples alternate periodically.

The stable conclusion is that E-value wealth is sensitive to clearly degraded test streams, especially on MuSiQue and 2Wiki. HotpotQA's response depends more on `alpha` and shift type. This should not be overstated as "all datasets and all alpha values always hit the cap."

Selective prediction is implemented as a post-detection intervention: if the wealth before a sample is already at `1/alpha`, the system abstains on that sample. For `gamma=0.5`, `alpha=0.1`:


| Dataset         | Coverage | Selective accuracy |
| --------------- | -------- | ------------------ |
| HotpotQA        | 0.931    | 0.705              |
| MuSiQue         | 0.444    | 0.373              |
| 2WikiMultiHopQA | 0.706    | 0.581              |


## 7. Baselines

### Internal Baselines

We compare Pandora-RAG against:

- Fixed-K retrieval depths for `K in {1,2,3,4,5}`;
- Global-Weitzman static stopping;
- instance-level DP Oracle;
- Probe without E-value;
- Probe with CP quantile gating.

Primary metrics are answer F1, EM, and average retrieval steps. Stage-3 risk-monitoring metrics include error rate at `gamma`, E-wealth trace, cap-touch behavior, coverage, and selective accuracy.

### Stop-RAG Alignment

Stop-RAG is evaluated as the main external stopping baseline. The aligned comparison uses the same Pandora test split and sample ids and only accepts true online early-stopping results from `stop_rag_test.sh`. Offline replay files and upstream Stop-RAG re-splits are not used as final head-to-head numbers.

Aligned Stop-RAG checkpoints/thresholds:


| Dataset         | Checkpoint / threshold | N    |
| --------------- | ---------------------- | ---- |
| HotpotQA        | `ckpt1000 / -0.09`     | 1000 |
| MuSiQue         | `ckpt1200 / -0.02`     | 417  |
| 2WikiMultiHopQA | `ckpt2400 / 0.03`      | 1000 |


Pandora `Probe+E-value` vs Stop-RAG:


| Dataset         | Method                | N    | F1     | EM     | Avg steps |
| --------------- | --------------------- | ---- | ------ | ------ | --------- |
| HotpotQA        | Stop-RAG              | 1000 | 0.5963 | 0.4640 | 5.000     |
| HotpotQA        | Pandora Probe+E-value | 1000 | 0.6654 | 0.5290 | 1.807     |
| MuSiQue         | Stop-RAG              | 417  | 0.2654 | 0.1942 | 4.643     |
| MuSiQue         | Pandora Probe+E-value | 417  | 0.4127 | 0.3141 | 3.410     |
| 2WikiMultiHopQA | Stop-RAG              | 1000 | 0.5076 | 0.4150 | 4.477     |
| 2WikiMultiHopQA | Pandora Probe+E-value | 1000 | 0.5728 | 0.4810 | 1.905     |


Macro averages:


| Method                | Macro F1 | Macro EM | Macro avg steps |
| --------------------- | -------- | -------- | --------------- |
| Stop-RAG              | 0.4564   | 0.3577   | 4.707           |
| Pandora Probe+E-value | 0.5503   | 0.4414   | 2.374           |


Recommended caveat: the current Stop-RAG result is a strong aligned single-point comparison, not a complete online threshold Pareto sweep.

## 8. Reproduction Commands

Stage 1, using existing prepared splits:

```bash
python -m stage1.run_stage1 \
  --datasets hotpotqa,musique,2wiki \
  --max-k 5 \
  --skip-prepare
```

Stage 1 with Contriever+BGE retrieval:

```bash
python -m stage1.run_stage1 \
  --datasets hotpotqa,musique,2wiki \
  --max-k 5 \
  --skip-prepare \
  --retriever-backend contriever_bge
```

Stage 2 main probe:

```bash
python -m stage2.run_stage2 \
  --per-dataset-optimal \
  --artifact-suffix pdopt_best
```

Stage 3 main evaluation:

```bash
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --artifact-suffix pdopt_best \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive
```

Stage 3 distribution-shift examples:

```bash
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --artifact-suffix pdopt_best \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --shift-type sudden
```

Stop-RAG aligned evaluation uses:

```bash
baselines/stop-rag-pandora/scripts/prepare_pandora_datasets.py
baselines/stop-rag-pandora/scripts/dataset.sh
baselines/stop-rag-pandora/scripts/train.sh
baselines/stop-rag-pandora/scripts/stop_rag_find_best.sh
baselines/stop-rag-pandora/scripts/stop_rag_test.sh
```

Do not use upstream Stop-RAG `download.sh` for the aligned comparison, because it creates a different split.

## 9. Main Artifacts

Stage-2 artifacts:

- `docs/stage2_report_pdopt_best.md`
- `results/stage2_probe_table_hotpotqa_pdopt_best.csv`
- `results/stage2_probe_table_musique_pdopt_best.csv`
- `results/stage2_probe_table_2wiki_pdopt_best.csv`
- `artifacts/probe/{dataset}/probe_mlp_pdopt_best.pt`

Stage-3 artifacts:

- `docs/stage3_results_summary.md`
- `docs/stage3_closeout.md`
- `results/stage3_evalue_hotpotqa.json`
- `results/stage3_evalue_musique.json`
- `results/stage3_evalue_2wiki.json`
- `results/stage3_final_pareto_all.png`

Stop-RAG artifacts:

- `docs/stop_rag_alignment_closeout.md`
- `baselines/stop-rag-pandora/results/**/online_test/*.metrics.json`

## 10. Software and Runtime Environment

Observed environment:


| Item                          | Value                                         |
| ----------------------------- | --------------------------------------------- |
| OS                            | Ubuntu 22.04 kernel `6.8.0-87-generic`        |
| Python                        | `3.12.2`                                      |
| Repository commit             | `53677b0`                                     |
| GPU driver                    | NVIDIA Driver `550.144.03`                    |
| CUDA reported by `nvidia-smi` | `12.4`                                        |
| GPUs                          | 4 x NVIDIA GeForce RTX 4090 D, 24564 MiB each |


GPU snapshot provided on 2026-04-20 15:14:19:


| GPU | Name                      | Memory used / total   | GPU util | Power draw / cap |
| --- | ------------------------- | --------------------- | -------- | ---------------- |
| 0   | NVIDIA GeForce RTX 4090 D | 22472 MiB / 24564 MiB | 100%     | 293 W / 425 W    |
| 1   | NVIDIA GeForce RTX 4090 D | 22122 MiB / 24564 MiB | 100%     | 291 W / 425 W    |
| 2   | NVIDIA GeForce RTX 4090 D | 8609 MiB / 24564 MiB  | 36%      | 162 W / 425 W    |
| 3   | NVIDIA GeForce RTX 4090 D | 1746 MiB / 24564 MiB  | 0%       | 62 W / 425 W     |


Python dependencies are recorded in `requirements.txt`, including:

- `torch>=2.2.0`
- `transformers>=4.44.0`
- `datasets>=2.20.0`
- `openai>=1.0.0`
- `scikit-learn>=1.3.0`
- `xgboost>=2.0.0`
- `rank_bm25>=0.2.2`
- `matplotlib>=3.7.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`

The GPU table above is a utilization snapshot, not a claim that all experiments used those exact instantaneous memory/utilization levels.

## 11. Paper-Writing Caveats

- Do not state that the probe uniformly beats the best fixed-K baseline. The safer claim is that it recovers most of the oracle benefit at substantially lower retrieval budget; dataset-specific fixed-depth comparisons vary.
- Do not claim E-value enforces empirical error rate below `alpha`. Its guarantee is an anytime-valid detection/monitoring guarantee under the low-error null.
- CP should be described as a high-cost static gate in this setting, not as a universally stronger safety baseline.
- Stop-RAG numbers are aligned online single-point results. A full Stop-RAG online threshold sweep would be needed for a complete Pareto frontier claim.
- Always state whether a run used `bm25` or `contriever_bge`, because changing the retriever changes trajectories, hidden states, oracle labels, and probe training data.

