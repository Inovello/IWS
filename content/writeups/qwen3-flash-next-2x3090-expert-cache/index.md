---
title: "17 to 25-29 t/s decode with the expert cache PR"
series: flash-next
slug: qwen3-flash-next-2x3090-expert-cache
date: 2026-09-03
kind: writeup
part: 1
summary: "All 48 expert layers in host RAM, an LRU cache of hot experts in VRAM, and the VRAM budget that made it pay."
discussion: https://www.reddit.com/r/LocalLLaMA/comments/1w5vjp6/qwen38flashnext_on_2x3090_ddr4_17_2529_ts_decode/
stats:
  - label: Decode
    value: "25-29 t/s"
    accent: true
  - label: Decode at 131k
    value: "17 t/s"
  - label: Cache hit rate
    value: "81-85%"
  - label: Prefill, 26k
    value: "138 t/s"
config:
  - k: Model
    v: Qwen3.8-Flash-Next
  - k: Quant
    v: UD-Q6_K_XL
  - k: Experts
    v: all 48 layers, pinned host RAM
  - k: Cache
    v: 135 slots per layer
  - k: Context
    v: "261,888"
  - k: KV
    v: f16
  - k: ubatch
    v: "512"
  - k: llama.cpp
    v: master b96806d + PR #27861
---

Most posts about this model are from people running one 3090 or a unified-memory box. My setup is two 3090s on PCIe 3.0 with a lot of slow DDR4 behind them, and I could not find numbers for it, so here are mine. This is the first of two parts. The [second part](/writeups/qwen3-flash-next-2x3090-q4-mtp/) takes the same box from 25-29 t/s to 37-41 with a different quant and speculative decoding, and explains a slowdown I could not account for when I wrote this.

## Setup

- Box: [dual Xeon E5-2696 v4, 188 GiB usable DDR4-2133, 2x RTX 3090 on PCIe 3.0](/hardware/)
- llama.cpp master `b96806d` plus PR [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) (the GPU-resident expert cache) and my own [#28223](https://github.com/ggml-org/llama.cpp/pull/28223) (pinned host experts under mmap)
- Model: unsloth Qwen3.8-Flash-Next UD-Q6_K_XL, 157.5 GiB on disk
- Full 261,888-token context, f16 KV cache, one slot
- All 48 expert layers pinned in host RAM, everything else on the GPUs
- Decode numbers are llama-server's own eval timing on a 4,000-token Python coding prompt with thinking on, temperature 0.7. Hit rate is the cache's own counter.

## The model, in the numbers that matter

Flash-Next is a 177B mixture-of-experts model with 48 layers. Each layer has 512 routed experts of which 10 fire per token, plus one shared expert. At Q6_K_XL the routed experts come to about 102 GiB, which is the part that does not fit in 48 GB of VRAM and never will. Everything else on the GPU side, attention, the Gated DeltaNet layers, hyper-connections, embeddings, output head, is about 5 GiB. There is also a 50.7 GiB per-layer token embedding table that is memory-mapped and read lazily on the CPU.

Two things about the architecture help here. Only 12 of the 48 layers are full attention layers, so the KV cache is small: about 3.1 GiB per GPU at the full context in f16. And the routed experts a token picks are mostly the same experts the last few dozen tokens picked, which is the whole reason the cache in this post works.

The practical context ceiling is 261,888 rather than the advertised 262,144. A CUDA `rms_norm` grid dimension limit breaks at exactly 262,144 on this build. I have not retested that since.

## Where I started

The starting point was 17 t/s decode at short context and 12 t/s at 131k depth, with four expert layers resident on each GPU and the other 40 in host RAM. Getting to that number had already taken a few evenings, and two of the lessons carry into everything below.

The first is that where a page lands in memory matters more than anything you can set with a flag. With mmap, expert pages are placed on the NUMA node of whichever thread touches them first. If the decode threads touch them first under `--numa distribute`, the pages interleave across both sockets and I measured 16 to 19 t/s. If the loader or a single-threaded prefill weight stream touches them first, they pile onto one node and the same configuration decodes at 12. I ran every flag variant on a cold page cache and they all landed at 11.4 to 12.4, because flags were not the variable. First touch was.

The second is that prefill from pageable memory runs at about half of PCIe speed. During prefill, llama.cpp streams every CPU-resident expert layer, about 2.1 GiB each, to the GPUs once per micro-batch. From ordinary mmapped memory the CUDA driver has to bounce those copies through a staging buffer. From pinned memory it is one DMA. Pinning the experts took a 26k-token prefill from 166 t/s to 392 at ub 2048. Master could not do that under mmap at the time, which is what my PR #28223 is about, and there is a [separate note](/writeups/pinned-host-experts-under-mmap/) on it.

So the baseline for this post is that: 17.6 t/s short context, 15.8 at 26k depth, 12.1 at 131k, 392 t/s prefill on a 26k prompt.

## Why decode was stuck at 17

One token at 17 t/s is about 59 ms. The 40 CPU-resident expert layers cost about 1.4 ms each, so roughly 55 ms of that is the expert path. Per token the CPU reads 40 layers times 10 experts times 4.2 MiB, about 1.7 GB, and at the 63 GB/s this box measures that is a 27 ms floor if the reads were perfectly bandwidth-bound. They are not. All experts on CPU gives 13.5 t/s at 44 threads and 12.6 at 22, so the thread count barely matters, and the effective memory traffic is around 22 GB/s against 63 available. The CPU-side MoE is a scattered top-10-of-512 gather, and the other 30 ms is thread wake-ups, per-layer host-to-device syncs, and remote NUMA reads. Under `perf`, 34% of compute thread time in a healthy decode is the OpenMP barrier spin.

Putting more expert layers on the GPU does not help much either. Each resident layer removes 1/48 of the gathers, which measured as about 3% decode per layer, and at full context only four or five layers fit next to the compute buffer anyway. Twelve layers on the GPUs at small context gave 18.5 t/s. That is the ceiling of the resident-layer approach on this box.

So the levers were: read fewer bytes from RAM per token, or amortise the overhead across more tokens per step. This post is the first one. Part 2 is the second.

## The expert cache

PR #27861 by csantiago78 adds a GPU-resident LRU cache of experts. Instead of parking whole expert layers in VRAM, it keeps a pool of slots per layer and fills them with the experts that were recently routed to. On a cache hit the expert runs on the GPU from the cached copy; on a miss the CPU chain computes it from host RAM as before, and the two partial outputs are summed. The math is exact by construction: cached experts are computed from the same weights, just on the other side of the bus. It only engages for single-token decode. Prefill and any batch larger than one token use the stock graph.

The reason it works is temporal locality in the routing. The long-run distribution of experts is flat, but the experts one token picks overlap heavily with the experts the previous few dozen tokens picked. The trace in the PR thread for this model showed 64 slots at about 81% hit rate, 96 at 87%, 128 at 90%.

The first run I did was 80 slots per layer, all 48 expert layers on the host, otherwise the production profile at ub 2048:

| Decode, 261,888 ctx | Baseline, 4+4 resident layers | Cache, 80 slots, all 48 layers on host |
|---|---|---|
| Short context | 17.6 | 20.3 climbing to 21.6 while the LRU fills, 23-24 steady |
| At 26k depth | 15.8 | 20.7 cold, 23.7 warm |
| At 131k depth | 12.1 | 16.3 (128 tokens), 17.2 on a cached prefix |
| Prefill, 26k prompt, warm | 392 | 356 |
| Prefill, 131k prompt | 303 | 282 |

The control matters here. The same all-host layout with the cache turned off decodes at 14.7 t/s, because it lost the eight resident layers. So 8.5 GB per GPU of cache, 80 slots times 24 layers times 4.4 MB, beat 8.5 GB of resident layers by about 45%. Prefill lost about 6%, which is the extra eight layers streamed per micro-batch.

Decode at 26k depth being faster than the first short essays looks odd until you remember the LRU is still filling during the first thousand or so tokens. Steady state at 80 slots is 23 to 24 t/s.

## Giving the cache VRAM

Eighty slots was the most that fit at ub 2048 with f16 KV, and 88 slots (the absolute VRAM ceiling) changed nothing. The thing that did was noticing where the VRAM was going. The load log at `-lv 4` gives the budget per GPU at full context: non-expert weights 2.1 to 2.5 GiB, attention KV 3.0 GiB, indexer KV 1.1 GiB, and then the compute buffers, which are 7.2 to 7.4 GiB per GPU at ub 2048 and 1.9 GiB at ub 512. One cache slot costs 24 layers times 4.22 MiB, about 101 MiB per GPU on Q6.

Dropping ubatch from 2048 to 512 frees about 5 GB per GPU, and 5 GB is 50 slots. That took the cache from 80 to 135 slots at the full 261,888 context, the hit rate from 70-77% to 81-85%, and decode to 25-29 t/s.

The cost is prefill on long prompts. A 26k prompt goes from about 75 seconds at ub 2048 to 100 to 120 seconds at ub 512, and I measured 138 t/s on a 26k prompt at ub 512 with a cold embedding table. Short prompts are unaffected, because one micro-batch streams the entire expert set over PCIe once regardless of size, about 5.6 seconds, so a 200-token prompt reads as tens of tokens per second either way.

This is the trade the whole config sits on: VRAM goes to cache slots instead of compute buffers, and long-prompt prefill pays for decode. If your experts are fully resident in VRAM you have nothing to trade against, and copying my `-ub 512` will cost you about 3x prefill for nothing. Use 2048.

## The cache was too small for code, and I did not know it

Every decode number in the first night of testing came from 512-token prose essays. The first real coding prompt after switching production to the 80-slot profile generated 25,000 tokens at 13.6 t/s average: 21 to 23 t/s for the first 7,800 tokens, then 8 to 16 for the remaining 17,000. The hit rate on code at 80 slots was 70-77%, lower than on prose, and something else was happening on top that I did not understand yet.

Reproduced under instrumentation, sampling decode, the cache's hit counter, PCIe receive, and a DRAM latency probe every minute:

| Run | Profile | Healthy decode | Hit rate | Slow phase began | Average over 16,000 tokens |
|---|---|---|---|---|---|
| A | 80 slots, 261,888 ctx, ub 2048 | 21-24 t/s | 70-77% | ~7,000 tokens, 5.5 min in | 15.6 t/s |
| B | 160 slots, 65,536 ctx, ub 512 | 25-29 t/s | 80-89% | ~11,800 tokens, 8.5 min in | ~22.5 t/s |
| C | old build, 4+4 resident, no cache | 17.1 t/s | n/a | ~4,100 tokens, 4 min in | 10.3 t/s |

Run B is what led to the 135-slot full-context profile above. Run C is the interesting one: the old build without any cache slowed down too, and earlier. The onset moved with the amount of DRAM traffic per token, 2.1 GB on the old build, 1.4 GB at 80 slots, less again at 160. Whatever was going on, it was not the cache. At the time I wrote it down as an open problem and a reason to restart the server after long sessions. The answer is in Part 2 and it is not software.

The lesson I did take away immediately: benchmark on the workload you actually run. Five hundred tokens of prose told me nothing about 25,000 tokens of code.

## Uploads per step

The cache fills at a configurable number of slot uploads per layer per step, default 2. I tried 4 to see if the LRU would settle faster. It did the opposite: hit rate 1 to 4%, the essays collapsed from 13.5 to 6 t/s, `nvidia-smi dmon` showed 4.2 GB/s of PCIe receive on GPU0 and 74-96% GPU utilisation from the copies alone. Four uploads times 48 layers times 4.4 MB is about 850 MB of host-to-device copies per token, the single upload worker cannot keep up over PCIe 3.0, slots stay in flight, and the LRU never settles. On PCIe 3.0, `--moe-expert-cache-inserts` stays at 2 or lower.

## Things that did nothing or hurt

All measured, so you can skip them:

- Thread count: 8, 16, 36 and 44 decode threads all give the same 21-22 t/s with the cache on. Decode is no longer CPU-bound. Production runs `-t 16` for the idle cores and the lower heat.
- `--poll`, `--prio`, CPU affinity masks: within noise, and strict pinning to physical cores is a disaster at 7 t/s. A one-socket mask halves throughput because the pages are interleaved across both.
- q8_0 KV: slower, again. The q8 attention path is about 3x slower per layer than f16 at head size 256, and on this model the KV cache is small enough that the saving buys almost nothing.
- Lazy loading of the embedding table (`-lzm auto`): -3% decode, no change to prefill.
- n-gram self-speculation (`ngram-mod`, `ngram-map-k`): produces no drafts on novel prose. It needs a 12 to 24 token repeat to fire. Revisited on code in Part 2.
- MTP at temperature 0.7: the verify batch re-reads the union of its tokens' experts from host RAM, which costs about as much as it saves, and the batch bypassed the cache entirely on the fork I tested. It only won greedy or at deep context. Also revisited in Part 2, where it turns out to be the biggest win once the cache serves the verify batch.
- `expert_used_count=8` via `--override-kv`: +10% decode, +6% prefill, and an unmeasured quality cost. The router renormalises the remaining top-k weights. Not used.
- More than 6 expert layers per GPU at full context: out of memory at ub 1024 and above.
- ik_llama.cpp: 353 t/s prefill with its fused MoE, but 10 t/s decode on this architecture. Mainline with pinned experts beats it on both.

## To replicate

You do not need my fork for this part. Master plus the PR:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/27861/head:pr-27861 && git merge pr-27861
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

LLAMA_ATTN_ROT_DISABLE=1 numactl --interleave=all build/bin/llama-server \
  -m Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
  -ngl 99 -c 261888 --parallel 1 -fa on \
  -ot "ffn_(gate|up|down)_exps\.weight=CUDA_Host,per_layer_token_embd\.weight=CPU" \
  --numa distribute -t 16 -tb 44 -b 4096 -ub 512 -ctk f16 -ctv f16 \
  --moe-expert-cache 135 -lv 4
```

Notes on the command:

- `-ot ...=CUDA_Host` pins the experts in page-locked host memory. On master that target is rejected at argument parsing, and even if it got through, the loader downgrades host buffer types to plain CPU under mmap. PR #28223 fixes both. Without it, use `=CPU` and accept pageable prefill.
- The two NUMA arguments are not multi-GPU flags. `numactl --interleave=all` decides where the pinned expert buffer lands; `--numa distribute` covers the compute threads. On a two-socket box without them the whole buffer first-touches one node.
- Size the cache to whatever VRAM is left after KV and compute buffers, about 100 MB per slot per GPU on Q6. Watch `nvidia-smi` after a long prompt: the CUDA pool grows about 350 MB during a 131k prefill and does not shrink. Leave headroom.
- `-lv 4` prints the cache hit rate every 512 steps as `moe-cache: ... hit-rate=`. That line is the first thing to look at if your number does not match mine.
- Load takes about 8.5 minutes on this profile. Part 2 has the fix that brings it to 2.

## Next

UD-Q4_K_XL is the obvious next step: about 30% fewer bytes per cache miss and 1.44x the slots per GB of VRAM. And MTP deserves another look once the cache can serve the verify batch. Both are in [Part 2](/writeups/qwen3-flash-next-2x3090-q4-mtp/).
