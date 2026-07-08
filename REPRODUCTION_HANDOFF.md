# HotpotQA RAG Paper Reproduction Handoff

This document is the handoff note for a new Codex/chat session. Read it first before changing or running anything.

## Current Goal

The user wants the repository to reproduce the paper flow as strictly as possible, using the local prepared HotpotQA data, Neo4j graph, local embedding/reranker models, and DeepSeek for generation/judging. The user does not want OpenAI API usage and does not want `.env` committed.

The important practical goal now is to understand and resolve why the implemented `Proposed` method has paper-like absolute metrics but does not clearly beat several strong baselines under the current fair/source-compatible evaluation.

## Latest Final State: 2026-07-09

The current working code has been re-stabilized after the graph-expansion over-fetch experiment. The final full run used:

```powershell
D:\cuda\venvs\vector-gpu\Scripts\python.exe -m new_experiments.all.run_all --sample-size 300 --max-workers 3
```

Result directory:

```text
D:\ew\test02\new_experiments\results\paper_hotpotqa_20260709_060513
```

This was the third and final allowed `300 x 15` full-flow run under the user's cap. It completed successfully against Neo4j database `paper7405`, with local CUDA embedding/reranker models and DeepSeek generation/judging.

Key code state:

| Area | Final decision |
|---|---|
| Neo4j source | Required by default; online evidence is read from Neo4j, not JSON documents |
| Graph expansion | Uses paper-style bounded semantic graph expansion; it no longer over-fetches extra neighbors for lexical re-ranking |
| Router | Keeps the heuristic/paper-feature router as default because DeepSeek LLM routing was worse in retrieval diagnostics |
| DeepSeek client | Uses `extra_body={"thinking":{"type":"disabled"}}`, fixing the prior empty-response compatibility issue |
| LLM router | Available behind `--llm-router`, but not used in the final run |

Latest verification:

```powershell
D:\cuda\venvs\vector-gpu\Scripts\python.exe -m pytest tests -q
```

Latest result before this handoff:

```text
67 passed, 3 warnings
```

### Final 300-Sample Main Comparison

| Method | Recall | Avg Len | ACC | Faith |
|---|---:|---:|---:|---:|
| Semantic RAG | 0.55 | 258.13 | 0.39 | 0.65 |
| +Rerank | 0.56 | 264.02 | 0.42 | 0.64 |
| GraphRAG | 0.58 | 256.47 | 0.45 | 0.70 |
| KG-RAG | 0.65 | 390.31 | 0.48 | 0.67 |
| MacRAG | 0.77 | 498.98 | 0.54 | 0.72 |
| Proposed | 0.78 | 1150.82 | 0.51 | 0.74 |

### Final 300-Sample Fixed-Scale Control

| Variant | Recall | Avg Len | ACC | Faith |
|---|---:|---:|---:|---:|
| Fine only | 0.60 | 329.43 | 0.41 | 0.69 |
| Uniform parent | 0.77 | 779.37 | 0.49 | 0.72 |
| Fixed 1-hop | 0.60 | 2164.46 | 0.41 | 0.64 |
| Fixed 2-hop | 0.60 | 2246.56 | 0.40 | 0.64 |
| Proposed | 0.78 | 1150.82 | 0.53 | 0.73 |

### Final 300-Sample Ablation

| Setting | Recall | Avg Len | ACC | Faith |
|---|---:|---:|---:|---:|
| Full model | 0.78 | 1150.82 | 0.52 | 0.71 |
| No graph expansion | 0.77 | 1029.36 | 0.52 | 0.71 |
| No selective parent | 0.66 | 1570.84 | 0.48 | 0.65 |
| No summary evidence | 0.78 | 1150.33 | 0.52 | 0.73 |

Interpretation:

- The final code follows the paper pipeline structure: initial sentence retrieval, reranking, adaptive routing, optional parent mapping, bounded graph expansion, summary evidence, budget selection, generation, and judge.
- The most reproducible gain is still parent/context organization. Graph expansion gives a small retrieval gain, matching the paper's qualitative implication, but it does not materially improve ACC in this sample.
- DeepSeek LLM routing was tested and rejected as default because it underperformed the heuristic router in retrieval diagnostics.
- The paper's absolute `Proposed` Recall/AvgToken are close, but the paper's claim that `Proposed` clearly dominates all baselines is still not reproduced: MacRAG is much stronger under this fair/local implementation.

## Locations

| Item | Path |
|---|---|
| Current working repo | `D:\ew\test02` |
| Original code copy provided by user | `D:\ew\最初版本代码\test02` |
| Paper PDF | `D:\ew\面向复杂多跳问答的自适应多尺度检索增强生成方法-0617-最终稿.pdf` |
| Python env used for tests/runs | `D:\cuda\venvs\vector-gpu\Scripts\python.exe` |
| Latest 200-sample result dir | `D:\ew\test02\new_experiments\results\paper_hotpotqa_20260708_221546` |
| Latest 200-sample run log | `D:\ew\test02\new_experiments\run_logs\run_200_workers6_20260708_221453.*.log` |

## Git State

Remote repository:

```text
origin = https://github.com/ratherbemyself/hotpotQARag.git
upstream = https://github.com/YuanCXC/Muti-scale-RAG.git
```

Current branch:

```text
codex/paper-neo4j-rag-flow
```

Important commits already pushed to `origin/codex/paper-neo4j-rag-flow`:

| Commit | Meaning |
|---|---|
| `6940180` | Paper-aligned Neo4j RAG flow |
| `0c313ed` | Source-compatible baseline alignment |

`gh` CLI is not installed in this environment, so pushes were done with plain `git push`; no PR was opened from this machine.

## Local Data / Runtime Facts

- `.env` exists locally and must not be committed.
- DeepSeek is used for generation/judge. OpenAI should not be used.
- Embedding and reranking use local models, not API calls.
- The run logs showed `bge-m3` and `bge-reranker-base` using `cuda:0`.
- Neo4j is required by the current runner. In the 200-sample run it reported:

```text
Neo4j ready: nodes=1024706, edges=2714190
```

The evidence source for online retrieval is Neo4j, not JSON, when `require_neo4j=True`.

## Previous Verification Note

The code was previously verified with:

```powershell
D:\cuda\venvs\vector-gpu\Scripts\python.exe -m pytest tests -q
```

Older result before the latest 2026-07-09 changes:

```text
59 passed, 3 warnings
```

## What Was Changed In Current Code

The latest code intentionally aligned the five main comparison baselines with the user's original source files under `D:\ew\最初版本代码\test02\new_experiments\all\exp1_*`.

Current implementation location:

```text
new_experiments/core.py
```

Source-compatible baseline behavior now is:

| Method | Current behavior | Source match |
|---|---|---|
| Semantic RAG | `vector_retrieve(sentence, k3)` | Matches `exp1_1_semantic_rag.py` |
| +Rerank | `vector_retrieve(sentence, k1)` then rerank to `k3` | Matches `exp1_2_rerank_rag.py` |
| GraphRAG | vector `k1`, use first 3 seeds, graph expand limit 5, final `(initial + expanded)[:k3]` | Strategy matches `exp1_3_graphrag.py`; Cypher/schema differs because current graph is Neo4j Section/Entity semantic graph |
| KG-RAG | vector `k1` + keyword `k2`, rerank to `k3`, no graph expansion | Matches `exp1_4_kg_rag.py` |
| MacRAG | sentence vector `k1` + paragraph vector `5`, rerank to `k3` | Matches `exp1_5_macrag.py` |

Tests added/updated in:

```text
tests/test_paper_graph_flow.py
```

These tests prevent the main baselines from silently using Proposed-only features such as parent context, budget selection, complexity routing, or graph expansion where the source baseline did not.

## Important Caveat: Fixed-Scale Controls

The fixed-scale controls are not fully source-compatible with `exp2_*` in the original source tree.

Current fixed-scale controls are more paper-flow oriented:

| Control | Current runner behavior | Original `exp2_*` issue |
|---|---|---|
| Fine only | shared initial candidates, then fine-grained selection | Original is mostly sentence vector + rerank |
| Uniform parent | true parent mapping via current parent context logic | Original file called it parent, but behaved more like sentence+paragraph rerank |
| Fixed 1-hop / 2-hop | fixed graph expansion on current Neo4j semantic graph | Original used old `Section - SEMANTIC_LINKS - Section` style queries |

This matters because the current `Uniform parent` result is very strong and can exceed `Proposed`.

## 200-Sample Run

Command used:

```powershell
D:\cuda\venvs\vector-gpu\Scripts\python.exe -m new_experiments.all.run_all --sample-size 200 --max-workers 6
```

The full runner was stopped after `table8_9_fixed_scale.csv` was produced because the user asked to look at comparison results first and not wait for ablation. The runner did start the ablation retrieval phase briefly before being stopped, but no ablation table was produced or used.

Generated files:

```text
new_experiments/results/paper_hotpotqa_20260708_221546/table5_6_method_retrieval.csv
new_experiments/results/paper_hotpotqa_20260708_221546/table7_semantic_records.csv
new_experiments/results/paper_hotpotqa_20260708_221546/table8_9_fixed_scale.csv
```

No result/log files were committed.

## 200-Sample Results: Main Methods

| Method | Recall | MRR | NDCG | MAP | Avg Len | ACC | Faith | AR | CR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic RAG | 0.56 | 0.69 | 0.54 | 0.45 | 258.08 | 0.46 | 0.85 | 0.77 | 0.75 |
| +Rerank | 0.58 | 0.69 | 0.55 | 0.46 | 266.02 | 0.49 | 0.82 | 0.75 | 0.76 |
| GraphRAG | 0.59 | 0.72 | 0.57 | 0.48 | 257.17 | 0.52 | 0.86 | 0.78 | 0.76 |
| KG-RAG | 0.67 | 0.81 | 0.66 | 0.58 | 383.46 | 0.54 | 0.82 | 0.77 | 0.81 |
| MacRAG | 0.78 | 0.92 | 0.79 | 0.75 | 497.75 | 0.62 | 0.87 | 0.82 | 0.84 |
| Proposed | 0.76 | 0.75 | 0.67 | 0.59 | 1429.86 | 0.64 | 0.88 | 0.83 | 0.81 |

Interpretation:

- `Proposed` absolute metrics are close to the paper on Recall, Avg Len, and ACC.
- `Proposed` has the highest ACC among main methods in this 200-sample run.
- But `MacRAG` is stronger than `Proposed` on Recall, MRR, NDCG, and MAP.
- This means the paper claim that `Ours` clearly beats all baselines is not reproduced under the current source-compatible baseline evaluation.

## 200-Sample Results: Fixed-Scale Controls

| Variant | Recall | MRR | NDCG | MAP | Avg Len | ACC | Faith | AR | CR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fine only | 0.60 | 0.80 | 0.63 | 0.57 | 324.64 | 0.51 | 0.82 | 0.76 | 0.78 |
| Uniform parent | 0.77 | 0.86 | 0.77 | 0.75 | 772.89 | 0.65 | 0.88 | 0.83 | 0.84 |
| Fixed 1-hop | 0.68 | 0.71 | 0.60 | 0.50 | 2885.24 | 0.59 | 0.89 | 0.84 | 0.78 |
| Fixed 2-hop | 0.68 | 0.71 | 0.60 | 0.51 | 2914.14 | 0.59 | 0.88 | 0.82 | 0.74 |
| Proposed | 0.76 | 0.75 | 0.67 | 0.59 | 1429.86 | 0.61 | 0.86 | 0.84 | 0.82 |

Interpretation:

- `Proposed` beats Fine only and fixed graph expansion on most practical tradeoffs.
- `Proposed` uses much less context than fixed 1-hop/2-hop.
- But current `Uniform parent` is too strong and beats or matches `Proposed` on many metrics.
- This suggests the strongest contribution may be parent/context organization, not graph expansion or summary.

## Paper Values To Compare Against

From the paper PDF:

### Table 3 Retrieval Quality

| Method | Recall | MRR | NDCG | MAP | AvgToken |
|---|---:|---:|---:|---:|---:|
| SemanticRAG | 0.470 | 0.80 | 0.67 | 0.60 | 222.73 |
| +Rerank | 0.480 | 0.84 | 0.70 | 0.63 | 206.64 |
| GraphRAG | 0.514 | 0.74 | 0.64 | 0.54 | 2333.15 |
| KG-RAG | 0.611 | 0.78 | 0.68 | 0.59 | 1362.38 |
| MacRAG | 0.469 | 0.75 | 0.63 | 0.55 | 505.74 |
| Ours | 0.747 | 0.84 | 0.72 | 0.65 | 1511.35 |

### Table 4 Generation Quality

| Method | ACC | Faith | AR | CR |
|---|---:|---:|---:|---:|
| SemanticRAG | 0.29 | 0.84 | 0.74 | 0.61 |
| +Rerank | 0.31 | 0.88 | 0.76 | 0.62 |
| GraphRAG | 0.33 | 0.87 | 0.80 | 0.58 |
| KG-RAG | 0.46 | 0.89 | 0.83 | 0.67 |
| MacRAG | 0.30 | 0.88 | 0.77 | 0.58 |
| Ours | 0.59 | 0.89 | 0.85 | 0.78 |

### Table 5 Fixed-Scale Variant Control

| Variant | Recall | AvgToken | Time/ms | ACC | Faith | CR |
|---|---:|---:|---:|---:|---:|---:|
| Fine only | 0.328 | 357.77 | 3592.45 | 0.19 | 0.67 | 0.19 |
| Parent rise | 0.596 | 760.65 | 3334.37 | 0.50 | 0.76 | 0.74 |
| 1-hop expansion | 0.436 | 1912.62 | 6341.08 | 0.53 | 0.53 | 0.44 |
| 2-hop expansion | 0.615 | 2722.41 | 6503.54 | 0.51 | 0.74 | 0.59 |
| Ours | 0.747 | 1511.35 | 4871.52 | 0.59 | 0.89 | 0.78 |

### Table 6 Component Ablation

| Setting | Recall | ACC | Faith |
|---|---:|---:|---:|
| Full | 0.747 | 0.59 | 0.89 |
| w/o Graph | 0.722 | 0.56 | 0.91 |
| w/o Parent | 0.620 | 0.46 | 0.88 |
| w/o Summary | 0.734 | 0.58 | 0.88 |

Important paper implication:

- Graph expansion adds only a small gain in the paper.
- Summary evidence adds an even smaller gain.
- Parent rise is the largest component contribution.

## Current Diagnosis

The awkward but important finding:

```text
Proposed itself is close to the paper in absolute terms, but several baselines are stronger than the paper reports.
```

Most important mismatches:

1. `MacRAG` is much stronger than paper:
   - Paper MacRAG Recall: `0.469`
   - Current MacRAG Recall: `0.78`
   - This is a retrieval metric, so it is not caused by DeepSeek judge noise.

2. `Uniform parent` is much stronger than paper:
   - Paper parent-rise Recall: `0.596`
   - Current Uniform parent Recall: `0.77`
   - Current implementation is a true parent mapping; the original source `exp2_2_uniform_parent.py` was closer to sentence+paragraph rerank, not a clean parent-map baseline.

3. GraphRAG/KG-RAG Avg Len does not match the paper:
   - Paper GraphRAG AvgToken: `2333.15`; current: `257.17`
   - Paper KG-RAG AvgToken: `1362.38`; current: `383.46`
   - Source-compatible GraphRAG slices `(initial + expanded)[:k3]`, so many expansions do not enter final context.
   - Source-compatible KG-RAG does not use graph expansion at all.

4. DeepSeek warnings are common:
   - Logs contain many `empty response` and `unparsable JSON` warnings.
   - This affects ACC/Faith/AR/CR stability.
   - It does not explain retrieval metric mismatches.

## Recommended Next Decisions

Do not mix these two experiment definitions:

### Option A: Source-Compatible Reproduction

Use the current committed behavior for main baselines. This is faithful to the user's original source tree under `new_experiments/all/exp1_*`.

Use this when the question is:

```text
What does the provided source code actually do?
```

Expected outcome:

- Fair same-data comparison.
- `MacRAG` likely remains strong.
- The result may not reproduce the paper's "Ours beats every baseline" claim.

### Option B: Paper-Strict Reproduction

Modify/add a separate mode or separate runner that follows paper descriptions rather than the original source files.

Use this when the question is:

```text
What is the most defensible implementation of the paper text?
```

Likely changes:

- KG-RAG should use actual entity/knowledge graph expansion if the paper says KG/entity expansion.
- GraphRAG should actually include expanded graph evidence in final context rather than slicing it away behind the initial candidates.
- Uniform parent should be defined clearly as a fixed parent-map control.
- Fixed 1-hop/2-hop should be clearly separated from Proposed budgeted selection.
- Report if results still do not match the paper rather than weakening baselines to force a win.

Recommended implementation style:

- Add an explicit config/mode such as `baseline_profile="source_compat"` vs `baseline_profile="paper_strict"` instead of overwriting behavior silently.
- Keep tests for both profiles.

## Suggested Next Commands

Check repo state:

```powershell
cd D:\ew\test02
git status -sb
git log --oneline -5
```

Run tests:

```powershell
D:\cuda\venvs\vector-gpu\Scripts\python.exe -m pytest tests -q
```

Run a 200-sample full flow if needed:

```powershell
D:\cuda\venvs\vector-gpu\Scripts\python.exe -m new_experiments.all.run_all --sample-size 200 --max-workers 6
```

If only retrieval metrics are needed and DeepSeek noise should be avoided, consider using `--no-generation --no-judge`, but note that the user previously preferred real/full flows and not smoke tests.

## What Not To Do

- Do not commit `.env`.
- Do not use OpenAI API.
- Do not rely on JSON evidence when `require_neo4j=True`; current intended flow reads online evidence from Neo4j.
- Do not silently change baselines to make `Proposed` win.
- Do not claim "strict paper reproduction" unless the baseline/profile definition is stated clearly.
- Do not treat ACC/Faith/AR/CR as fully stable without noting DeepSeek empty/unparsable judge responses.

## Current Best Summary

The code now runs the Neo4j-based paper flow, with the main comparison baselines source-compatible with the user's original code and the adaptive flow aligned to the paper's described modules. On the final 300-sample run, `Proposed` has paper-like absolute Recall and lower-than-paper Avg Len, but it does not clearly dominate the strongest local baselines. The strongest verified contribution is selective parent/context organization; graph expansion is useful but small, and DeepSeek-as-router did not outperform the heuristic router.
