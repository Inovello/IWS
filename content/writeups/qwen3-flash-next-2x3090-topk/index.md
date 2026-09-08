---
title: "A top-k fallback that was sorting far more than it needed to"
series: flash-next
slug: qwen3-flash-next-2x3090-topk
date: 2026-09-08
kind: writeup
part: 3
summary: "Long-context decode was 9 to 12% faster once the sparse-attention indexer stopped fully sorting each row, and a 480-request quality screen found no consistent regression."
stats:
  - label: Decode at 119k
    value: "+9-12%"
    accent: true
  - label: Quality requests
    value: "480"
  - label: Matched pairs
    value: "240"
  - label: Seeds
    value: "3"
config:
  - k: Model
    v: Qwen3.8-Flash-Next
  - k: Quant
    v: UD-Q4_K_XL
  - k: Cache
    v: 150 slots
  - k: Draft
    v: shared Q8_0 MTP, n-max 3
  - k: Context
    v: "261,888"
  - k: Depths tested
    v: 37k and 119k
  - k: KV
    v: f16
  - k: ubatch
    v: "512"
  - k: RAM
    v: 128 GB, four DIMMs
  - k: CUDA
    v: "12.0"
---

The next thing worth writing down on this box came from a small operation doing far more work than the answer required. In my CUDA build, the sparse-attention indexer's top-k fell back to sorting an entire row before taking the handful of entries it wanted. At long context those rows are wide, and that sort was showing up in the profile.

Switching that fallback to llama.cpp's existing radix selection made median decode 9 to 12% faster at about 119k of context, on all three sampling seeds I tried. Same model, same quant, same MTP setup. One binary with an environment switch choosing the old or new path, so nothing else moved between arms.

Then I ran 480 requests over real documents to check the answers had not quietly got worse. They had not, with one recall question I cannot call either way.

## Setup

- Box: [dual Xeon E5-2696 v4, 2x RTX 3090 on PCIe 3.0](/hardware/), running on four 32 GB DIMMs for these runs, so 128 GB rather than the usual 192
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL, expert cache at 150 slots, shared Q8_0 MTP head at draft depth 3 on the second GPU
- Allocated context 261,888, f16 KV, ubatch 512, CUDA toolkit 12.0
- Corpus: a frozen set of project documents, 38 questions at about 37k tokens and 42 at about 119k
- Sampling matched production: temperature 0.7, top-p 0.8, top-k 20, min-p 0, thinking on

## Where this sits after part 2

[Part 2](/writeups/qwen3-flash-next-2x3090-q4-mtp/) covered the move to UD-Q4_K_XL, MTP on top of the expert cache, and the loader fix. Its headline of 37 to 41 t/s came from a coding prompt with thinking on.

This post is a different workload: questions over long project documents. A control sitting near 30 t/s here does not mean the earlier numbers evaporated. What matters is the control and the candidate answering the same questions with the same sampling on the same hardware, and that is what these runs are.

There is one hardware difference worth stating plainly. Two DIMMs were out for replacement, so every arm here ran on four DIMMs and 128 GB, which is two memory channels per socket instead of three. All six arms had it, so the comparison is fair, but do not line these absolute numbers up against part 2's. A six-DIMM rerun is its own comparison for later.

## The expensive fallback

Flash-Next's QSA indexer scores positions and then uses top-k to decide which ones actually feed attention. That is a different operation from the top-k sampler that picks an output token, and it runs constantly during decode.

The CUDA implementation takes different paths depending on what the build has available. When the newer CUB `DeviceTopK` is not there, the older fallback sorts the whole row and takes what it needs off the front. My CUDA 12.0 build lands on that fallback.

At long context the rows are wide, so that is a full sort of thousands of entries, repeatedly, to find sixteen of them. An Nsight Systems capture at around 131k pointed straight at it.

llama.cpp already had a radix selector that finds the top entries without sorting the rest. Swapping it in took top-k kernel time from roughly 5.1 ms to 0.25 ms per committed token in the first patched captures.

Those are kernel timings, averaged over the captured tokens. They are not request latency, and a 20x on one kernel does not become a 20x on anything you can feel. The model still has all its other work to do. What it does mean is that the operation stopped being worth looking at, which is the point.

## The change, and the PR I did not open

My revision turns on radix selection for rows of 8,192 columns or more inside the CUB fallback branch. Narrower rows keep the old path, and builds that have `DeviceTopK` are untouched.

The threshold came from measuring the operators directly. At the k=16 shapes I tested, argsort was still ahead at 4,096 columns and radix selection was ahead at 8,192. Those measurements were taken without CUDA graphs on a GPU that was not otherwise idle, so 8,192 is a conservative pick for this build rather than a number anyone should treat as universal.

Before writing anything up for upstream I went looking, and found [Rhonstin's PR #28366](https://github.com/ggml-org/llama.cpp/pull/28366) already proposing to enable the radix fallback when `DeviceTopK` is missing, building on the [radix top-k work in #27466](https://github.com/ggml-org/llama.cpp/pull/27466). So there is no PR from me here. What this exercise produced is the profiling, the local qualification and the model-level evidence, which is more useful to that thread than a second implementation of the same idea. The numbers below come from my local revision, with its own threshold and control switch. They are not measurements of that PR's head.

Operator correctness passed 525 of 525 cases on each path, and 268 of those cases were rows at or above the threshold, 88 of them with ties. That says the operator selects the right entries. It does not say the model gives the same answers, which is why the quality screen exists. Ties are exactly where top-k ordering gets subtle.

## Comparing the build against itself

Both arms are the same binary. A local environment variable picks the path:

| Arm | Setting | Top-k behaviour |
|---|---|---|
| Control | `GGML_CUDA_TOPK_ARGSORT=1` | Old argsort fallback |
| Candidate | Unset | Radix selection on eligible wide rows |

That switch is part of my patch. It does nothing in a stock llama.cpp build.

The question list, answer key, grader and runner were all fixed before the first launch, and the answer key never went near the model. Each arm ran all 80 question and depth combinations on seeds 1, 2 and 3, which is 480 requests and 240 matched comparisons. Those are 80 combinations repeated three times, not 240 independent tasks.

Every arm started in a fresh server process. Within an arm the harness holds one conversation per document depth, so later questions reuse the document prefix rather than re-prefilling it, which means the stated depths describe where each conversation starts rather than an identical token count on every request. Control went first on seeds 1 and 3, candidate first on seed 2. The loaded build and the control flag were recorded at every launch.

## Decode results

Throughput is the server's own decode timing, and each figure is the median across that arm's requests at that depth.

| Seed | 37k control | 37k candidate | 119k control | 119k candidate |
|---|---|---|---|---|
| 1 | 39.5 t/s | 41.3 t/s | 30.2 t/s | 33.7 t/s |
| 2 | 37.6 t/s | 40.6 t/s | 30.2 t/s | 33.3 t/s |
| 3 | 39.5 t/s | 40.2 t/s | 30.4 t/s | 33.1 t/s |

At the deeper end that is 9 to 12% on the median, with the candidate ahead on every seed. Each of those medians covers 42 requests.

The spread behind them, at 119k:

| Seed | Control, median [min, max] | Candidate, median [min, max] |
|---|---|---|
| 1 | 30.2 [27.3, 34.6] | 33.7 [30.3, 37.8] |
| 2 | 30.2 [27.1, 34.1] | 33.3 [30.3, 36.9] |
| 3 | 30.4 [27.3, 34.4] | 33.1 [29.0, 37.5] |

Those ranges are variation across different questions, not confidence intervals. The percentages compare the two arm medians on each seed, not the median of per-question speedups, because under sampling the two arms write different answers of different lengths.

An earlier greedy screen at about 131k found a benefit too. I have kept those numbers separate rather than pooling them with these. Nothing here says anything about prefill.

## Whether the answers got worse

The grader scored against the frozen key, and every result stayed in the totals:

| Seed | Control | Candidate |
|---|---|---|
| 1 | 80/80 | 79/80 |
| 2 | 77/80 | 79/80 |
| 3 | 78/80 | 80/80 |
| Total | 235/240 | 238/240 |

Matched question by question within each seed, the candidate was better on four, worse on one, and identical on 235. Three points in the totals is not evidence that the patch improves quality, and I am not claiming it does.

The one disputed item was question 33 at the deeper depth. It asks for all twelve distinct repository paths in the document, and the grading rule is exact: miss one or invent one and the answer is wrong.

| Seed | Control | Candidate |
|---|---|---|
| 1 | All 12 | Missed one |
| 2 | Missed one | Missed the same one |
| 3 | Missed one | All 12 |

Both arms failed it twice. Every failure dropped exactly one real path and none of them made anything up. I had the seed-1 answers reviewed with the arm labels shuffled and hidden, and the reviewer confirmed the candidate's answer was genuinely incomplete, so it stayed marked wrong.

The other two seeds are what stop me pinning that on the patch. On seed 2 both arms dropped the same path, and on seed 3 the candidate got the full list and the control did not. So that question is inconclusive, and I would rather say so than round it off in either direction.

What the screen supports is narrower than "quality is fine": no consistent regression showed up in this suite, with one recall item unresolved. The control's own inconsistency is not an argument that a real difference would not matter. It is a reason the test cannot resolve one this small.

## What this changes and what it leaves open

Part 2 reported poor MTP acceptance on a deep-context workload. This document workload behaved differently, so context length on its own does not tell you whether MTP will pay. The prompt and what the model writes back matter as much. Both arms here ran MTP, so none of this ranks MTP against no draft at all.

The quality evidence covers this document suite at roughly 37k and 119k, on three seeds, on four DIMMs. It does not cover other task types, the full 261k, or the six-DIMM machine. The run-to-run output variation is present with the old path too, and chasing that down is its own job.

For this workload the speedup repeated on every seed and survived a model-level check, so it stays in the build I run, and the measurements go to the existing upstream thread rather than into a competing PR. The branch with the source change, build instructions and an A/B client is [on my fork](https://github.com/Inovello/llama.cpp/blob/dd64a3db0c453b0e03de15574ff874c2dc77cb28/examples/flashnext-topk/README.md) if you want to try it on your own documents. The quality corpus is private, so what you get is a way to test the effect on your workload, not a way to reproduce my scores.

## Next

Prefill. A profile of a single 8k-token document request has about 74 seconds of host-to-device transfer on the first GPU against roughly 5 seconds of kernel time, which says the time is going into moving expert weights over PCIe rather than into computing anything. That needs its own writeup once I have something measured rather than something suspected, and a faster top-k during decode says nothing about it.
