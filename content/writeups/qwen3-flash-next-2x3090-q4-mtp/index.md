---
title: "25-29 to 37-41 t/s with UD-Q4_K_XL and MTP"
series: flash-next
slug: qwen3-flash-next-2x3090-q4-mtp
date: 2026-09-04
kind: writeup
part: 2
summary: "A quant swap, MTP on top of the cache, a 4x faster load, a bug in the cache PR, and the discovery that my RAM had been thermal throttling the whole time."
discussion: https://www.reddit.com/r/LocalLLaMA/comments/1w6ozbj/update_qwen38flashnext_on_2x3090_ddr4_part_2_2529/
stats:
  - label: Decode, thinking on
    value: "37-41 t/s"
    accent: true
  - label: Code emission
    value: "49 t/s"
  - label: Load to ready
    value: "2 min"
  - label: Host RAM
    value: "105 GB"
config:
  - k: Model
    v: Qwen3.8-Flash-Next
  - k: Quant
    v: UD-Q4_K_XL
  - k: Experts
    v: all 48 layers, pinned host RAM
  - k: Cache
    v: 150 slots (188 without MTP)
  - k: Draft
    v: MTP head, n-max 3, on CUDA1
  - k: Context
    v: "261,888"
  - k: KV
    v: f16
  - k: ubatch
    v: "512"
  - k: Branch
    v: flashnext-2x3090
---

This follows [Part 1](/writeups/qwen3-flash-next-2x3090-expert-cache/), which got this box from 17 to 25-29 t/s with the GPU expert cache. Same machine: two RTX 3090s on PCIe 3.0, dual Xeon E5-2696 v4, 188 GiB of DDR4-2133, llama.cpp, the full 261k context, f16 KV, all 48 expert layers in host RAM and everything else on the GPUs. Since Part 1 I switched quants, stacked MTP on top of the cache, fixed the load time, found a bug in the cache PR, and found out my RAM had been thermal throttling. So I now own a fan I did not plan to buy.

Unless stated otherwise, numbers are from the same 4,000-token Python coding prompt with thinking on, at temperature 0.7, and single runs on that prompt spread about 8%.

## Setup

- Box: [dual Xeon E5-2696 v4, 188 GiB usable DDR4-2133, 2x RTX 3090 on PCIe 3.0](/hardware/)
- Branch [`flashnext-2x3090`](https://github.com/Inovello/llama.cpp/tree/flashnext-2x3090): master `b96806d` + PR #27861 (expert cache) + PR #28223 (pinned host experts under mmap, load fix) + PR #28243 (MTP) + the duplicate-id fix and the 8-token cache gate described below
- Model: unsloth Qwen3.8-Flash-Next UD-Q4_K_XL, 111.3 GB on disk, plus the 2.79 GB shared MTP head from the same repo
- Full 261,888-token context, f16 KV, one slot, ubatch 512
- Decode is llama-server's own eval timing; hit rate is the cache's own counter; draft acceptance is the server's per-request line at `-lv 4`

## Where it is now

| | Start: UD-Q6_K_XL, 4+4 resident layers | Part 1: Q6 + cache, 135 slots | Now: UD-Q4_K_XL + cache 188 slots + n-gram draft | Now: Q4 + cache 150 slots + MTP |
|---|---|---|---|---|
| Decode, coding prompt with thinking | 17 | 25-29 | 32-35 | **37-41** |
| Decode, code emission, thinking off | - | 24 | 37 | **49** |
| Decode at 131k depth | 12 | 17 | 18-20 | 14-16 |
| Prefill, 26k prompt | ~350 at ub 2048 | 138 | 180-195 | 180-195 |
| Load to ready | ~13 min | 8.5 min | 2 min | 2 min |
| Host RAM for the experts | 104 GB pinned + 51 GB PLE | same | 73 GB pinned + 28 GB PLE | same |
| Cache hit rate | - | 84-85% | 90-92% | 84-85% (fewer slots) |

Two profiles come out of this, and the choice is per workload. MTP is the default for chat and code at short to mid context. The n-gram profile with 188 slots is what I switch to for long documents, where MTP loses.

## What changed, in order of payoff

### 1. UD-Q4_K_XL instead of UD-Q6_K_XL

The interesting finding is that the hit rate did not change at equal slot count. Q4 at 135 slots hits 84.7%; Q6 at 135 hits 84-85%. The LRU only cares how many slots it has. What Q4 buys is 1.44x the slots per GB of VRAM, because a Q4 expert-layer is 2.93 MiB against 4.22 for Q6. So 188 to 193 slots fit where 135 did, which takes the hit rate to 90-92% and decode from 27 to 32-35 t/s. Cheaper misses add a few percent on top.

Prefill went up 32 to 39% at the same ubatch, 181 to 196 t/s on the 26k prompt against 138, and 165 against 119 on a 131k prompt. A prefill micro-batch streams every expert weight once and Q4 has 30% fewer expert bytes.

Host RAM drops from about 155 GB to about 105: 73.5 GiB pinned experts plus a 28 GiB mapped embedding table. That answers the question from Part 1's thread about whether this needs 188 GB. It does not. A 128 GB box does it.

Quality cost, measured here against Q6 on 32k tokens of wikitext-2: KL divergence 0.0506 mean and 0.0108 median, top-1 agreement 92.59%, perplexity ratio 1.0008. That lands on unsloth's published table (0.047 and 92.3% against BF16). The quant disagrees with Q6 on the top token about 7% of the time, mostly on near-ties. On HumanEval with greedy sampling and thinking off, Q4 scored 156 of 164, the same as Qwen3.8-27B at Q8_0 on the same harness. On a 250-question MMLU subset through the chat API it scored 87.2% against the 27B's 81.2%. Standard error on 250 questions is about 2.3 points, so the gap is real but not large.

### 2. MTP on top of the cache

In Part 1 I concluded MTP did not pay at temperature 0.7. That conclusion was from the unsloth fork with the cache disabled during the verify batch: every verify re-read its tokens' experts from host RAM, and that cost about what the accepted drafts saved.

On mainline, with PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) (danielhanchen's port of the unsloth MTP head) and with the cache accepting verify batches (item 4), it is a different story. The MTP head drafts every step. Acceptance is 50-58% on reasoning text and 94% on code emission. Decode went from 32-35 to 37-41 t/s on the thinking prompt and from 37 to 49 on code emission.

The draft head sits on the second GPU and costs about 4.5 GB, which is why the slot count in the table drops from 188 to 150. Even with 38 fewer slots it is a clear win at short context: +15% on the coding run against 188-slot production, and about +32% on the refactor prompt.

Where MTP loses is depth. At 131k the acceptance collapses to about 20%, every rejected draft is a wasted verify step, and decode falls from 18-20 to 14-16 t/s. Long prefill is about 7% slower from the draft context's checkpoints. The n-gram draft (`--spec-type ngram-map-k`, size-m 7) is the better choice for long-document sessions: on a continuation of the document's own text it accepted 140 of 140 drafted tokens and decoded at 32 t/s at 131k depth, which no MTP head can match.

I also tried chaining both drafts (`--spec-type ngram-map-k,draft-mtp`). At short context it equals MTP alone, and at depth it is worse than either, because MTP's rejected tokens dominate and the n-gram's verbatim hits are lost. Dropped.

### 3. Load 8.5 minutes to 2

The load log already had the answer. Metadata took 2 seconds, one `cudaMallocHost` of 104 GiB took 60 seconds, and the copy of the expert tensors into it took 450. During that copy `/proc/<pid>/io` showed 106 GiB of block reads against 22 MB of `read()` traffic: the loader was pulling the experts through the mmap one page fault at a time, 26.7 million of them, at 236 MB/s. The reason is that `--numa distribute` applies `POSIX_MADV_RANDOM` to the whole mapping, which turns off readahead, and the loader's host-destination branch was a plain memcpy from that mapping.

Reading the host-destination tensors straight from the file with `pread` instead fixed it. Nine lines in the loader. The copy phase went from 450 seconds to 104, block reads from 236 MB/s to 1.2 GB/s in 128 KiB requests, load to ready from 512 seconds to 168 on the same box the same hour, greedy output byte-identical, the buffer still pinned, prefill unchanged. The plain mmap path without a host override is not touched by the change and still loads in 24 seconds. That fix is the second commit on my PR #28223. The quant swap then took it from 168 to 125 seconds, because there is less to pin.

### 4. A bug in the cache PR when n_tokens > 1

A 5090 user hit an illegal memory access running the cache with a draft model. The cause is in the cache PR, not their setup. #27861 maps every uncached expert of a token to the same dummy slot, and the CUDA batched `mul_mat_id` paths assume a token's expert ids are distinct. When they are not, the id helper collapses the duplicate lanes into one compact row, leaves the inverse map uninitialised for the losers, and the scatter quantizer writes to garbage rows. That is the crash. The rows for the losing lanes are then never written and the graph multiplies stale pool memory by real router weights, which is the part the reporter's memset workaround does not fix.

I reproduced it without the model in `test-backend-ops` with a `dup_ids` option on the `MUL_MAT_ID` cases: the unfixed tree aborts with a CUDA illegal memory access at the first duplicate case that reaches the quantized matmul kernel. Fixing the id helper to give every matching lane its own compact row removes the abort and passes every pre-existing case, but three other sites still assume distinct ids (the F16 path's slot map and two launch grids sized from the token count), so batched results through the cache are still wrong outside one window.

The window is `mmvq`, the per-output-row kernel that quantized experts use for batches of up to 8 tokens on this GPU and on Blackwell. It gathers every lane and is duplicate-safe. Boundary data: with duplicate ids, Q4_K, Q5_1 and Q6_K all pass at 8 tokens (6 of 6) and all fail at 9, 12 and 16 (18 of 18). So the branch gates the cache at 8 tokens with quantized experts, and MTP's `--spec-draft-n-max 3` keeps the verify batch at 4. A server run at `-ub 8 -b 8`, which pushes the prompt itself through the cache in 8-token batches, produced byte-identical greedy output to the batch-1 path. F16 experts never take that kernel on NVIDIA, so the cache cannot be used with batches at all on F16 experts.

Details and the reproduction are in [my comment on #27861](https://github.com/ggml-org/llama.cpp/pull/27861). The clean fix is for the cache to emit distinct dummy slots per uncached lane. The PR's author has not picked it up yet.

### 5. My RAM was thermal throttling

This is the slowdown from Part 1, the one where decode fell from 17 to 8-12 t/s after some minutes of sustained load and a restart fixed it. It was the memory controller throttling the DIMMs.

The tell was the shape of the slow phase. Same instruction stream, same cache miss rate, clocks unchanged at 3.1 to 3.2 GHz on both sockets, GPUs idle, no swap, no page-fault difference, and the CPU running at 0.28 instructions per cycle instead of 0.43. That is memory latency going up. A pointer-chasing probe confirmed it: 105 ns local and 165 remote when healthy under load, 160 to 230 during the slow phase, 340 to 440 at the worst. And the uncore counters named it:

```bash
perf stat -a --per-socket -e unc_m_dram_refresh.high,unc_m_power_critical_throttle_cycles -- sleep 3
```

During a slow phase `unc_m_power_critical_throttle_cycles` reads 0.7 to 2 billion per 3-second window, on every rank, alternating between the sockets minute by minute. Healthy, it reads zero. You do not need a BMC to see this, and this box has none.

The DIMM temperatures themselves are readable from the memory controller's thermal registers with `setpci` (on Broadwell-EP the low byte of offset 0x150 on the channel's thermal-control function). Under sustained decode the hottest of the six quad-rank LRDIMMs reads 78 C in still air, and the throttling starts within a minute of it getting there. The BIOS runs the DIMMs in open-loop thermal throttling mode, where the controller estimates temperature from activity, which is why onset came earlier with more traffic and why an 8-minute disk-bound load "fixed" it: the DIMMs cooled.

The fix is airflow, not software. One fan on the DIMM banks: 44 to 57 C under load, zero throttle cycles for the whole run, and the 16,000-token coding generation went from 10 to 15 t/s average to 24.6 in 10.8 minutes, with the DIMMs peaking at 55 C. Opening the side panel without the fan bought nothing; the DIMMs climbed 3 C a minute and throttled at 78 again.

If your decode falls off after a few minutes on a DDR4 Xeon box with quad-rank DIMMs, check this before touching software. It does not affect the numbers in the table above, which were all taken with the fan on.

## Things that did nothing or hurt on this box

So you can skip them:

- q8_0 KV: fits 26 more slots (219 against 193) and raises the hit rate 1.4 points, but the q8 attention path costs more than the saved misses. -4.8% at short context, -18% at 131k depth on a clean 128-token measurement. Production stays on f16.
- Mirror-NUMA, PR #27986: cannot replicate a pinned buffer, and it says so in its own log. No effect.
- QSA gather, PR #28213: identical numbers with and without at 131k. The attention scan is not the limiter with the cache on.
- `--load-mode none`: same cold-prefill effect as on-direct, 25 seconds longer load, and it eats the page cache.
- The ik_llama small-GEMV port for the hyper-connection matmuls: mainline already routes those through `mul_mat_vec_f` on Ampere, so there was nothing to gain. 27.2 t/s before, 26.0 after.
- Chained n-gram and MTP drafts, as above.
- More than 2 cache uploads per step, thread, poll and priority flags: same as Part 1.
- `--lazy-mode on-direct`, PR #28136: +7 to 12% on the first long prompt after a restart or page-cache drop, nothing after, nothing on decode. Nice, but it is a draft PR and not in my build.

## To replicate

```bash
git clone -b flashnext-2x3090 https://github.com/Inovello/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j -t llama-server

LLAMA_ATTN_ROT_DISABLE=1 numactl --interleave=all build/bin/llama-server \
  -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp -devd CUDA1 --spec-draft-n-max 3 \
  -ngl 99 -c 261888 --parallel 1 -fa on \
  -ot "ffn_(gate|up|down)_exps\.weight=CUDA_Host,per_layer_token_embd\.weight=CPU" \
  -lzm off --numa distribute -t 16 -tb 44 -b 4096 -ub 512 -ctk f16 -ctv f16 \
  --moe-expert-cache 150 -lv 4
```

- The MTP head is `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.79 GB) from the unsloth/Qwen3.8-Flash-Next-GGUF repo. The shared file has no embeddings of its own; the PR borrows them from the target, so it only loads as `-md` of the main model. On Windows the shared head can fail with a null `model_shared` because the fitter pokes the draft before the target is registered. Try `-fit off`, or use the non-shared head.
- Slot sizing on Q4: about 75 MB per slot per GPU at full context and ub 512. Without MTP I fit 188 slots with about 1 GB free per GPU; with the draft head on CUDA1, 150. The CUDA pool grows about 350 MB during a 131k prefill, so leave that much.
- `-lv 4` prints the cache hit rate every 512 steps (`moe-cache: ... hit-rate=`) and the draft acceptance per request.
- `-lzm off` keeps the 27 GiB embedding table resident. That is a 188 GB decision. On less RAM use `-lzm on` and let it page from disk.
- `-ub 512` is a cache-specific compromise: the compute buffer is 1.9 GiB per GPU at 512 and 7.2 at 2048, and that VRAM is about 50 slots. If your experts are resident in VRAM there is nothing to trade and you want 2048.
- The two NUMA arguments are not multi-GPU flags. They place the pinned expert buffer and the compute threads. Someone dropped both on a two-socket EPYC box because the post implied they were, and lost a few t/s to it.
- For long-document sessions, swap the three MTP flags for `--spec-type ngram-map-k --spec-ngram-map-k-size-m 7` and raise the cache to 188.
- On one GPU: same idea, put `-devd` on it, or skip MTP and take the slots.

Both PRs are drafts. #28243 has open review comments and #27861 has the bug above. The branch is what runs here today; it is not something I would call finished.

## What other people measured

Five boxes have run the branch since the post went up, and the results are consistent with the mechanism:

- RTX 5090 + RTX 3090 Ti, DDR5-6400, Q4: 41 t/s on stock llama.cpp with 21 of 48 expert layers resident, 58 t/s average over a 32k-token generation with the branch, windows up to 88, hit rate 90.7%. Same cache-versus-resident gain as here; faster RAM raises the whole curve.
- Threadripper 3970X, 3090 + 3090 Ti, DDR4-3200: 40 to 44 t/s. The first box to land on my number exactly. Thread count irrelevant, as here.
- One A40 48 GB, dual EPYC 7343, 768 GB: 25 t/s, then 27 with the NUMA arguments restored and the context cut to buy slots. About 90% of what bandwidth scaling predicts from the 3090 pair. Binding memory to the GPU's own node instead of interleaving made no difference.
- 2x 3090 with NVLink: 25 to 35 t/s, prefill 100 t/s at ub 512. Useful as a control: under a layer split only a few KB of hidden state cross between the cards per token, so NVLink cannot help here, and it did not.
- RTX 5090, 9800X3D, 192 GB DDR5, Windows, on PR #28223 alone: prefill 227 to 627 t/s at 47k with pinned experts, decode flat, no load-time penalty on Windows.

The cache itself has no backend guard, and two people on the PR thread have it working on Vulkan for single-token decode. What is CUDA-only in my branch is the batched-request fix, so on Vulkan the cache without MTP is worth a try and the cache with MTP is not.

## Next

A proper comparison against Qwen3.8-27B at Q8 on this box, speed and quality, since that is what most people actually want to know. The harness exists and the first passes are in; the writeup will follow when the third pass is.
