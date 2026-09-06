---
title: "A silent -ot downgrade under mmap, and a 9-line loader fix"
slug: pinned-host-experts-under-mmap
date: 2026-09-05
kind: note
summary: "Why -ot ...=CUDA_Host did nothing on master, what my PR #28223 changes, and where the 8 minutes of load time went."
---

This is the note behind llama.cpp PR [#28223](https://github.com/ggml-org/llama.cpp/pull/28223). It is small: two files, nine lines added, two changed, plus a second commit of nine more. It is also worth 2 to 2.3x prefill for anyone running a MoE model with the experts offloaded to CPU RAM, which is why I wrote it down.

## The problem

When MoE experts live in host RAM, prompt processing copies each expert layer to the GPU for every micro-batch. From ordinary pageable memory the CUDA driver cannot DMA directly; it bounces the copy through its own staging buffer at roughly half of PCIe speed. From pinned, page-locked memory it is a single DMA at full speed. On my box that is the difference between 166 t/s and 392 t/s on a 26k-token prompt.

The way to get pinned experts should be `-ot "ffn_(gate|up|down)_exps\.weight=CUDA_Host"`. Two things stopped that from working on master:

1. The argument parser does not offer host buffer types as `-ot` targets at all. The flag is rejected at startup with "Available buffer types: CPU, CUDA0, CUDA1".
2. Even when a host buffer type reaches the loader, there is a rule in `create_tensor()` that replaces it with a plain CPU buffer under mmap. The rule exists for a good reason: with mmap the bytes are already in memory through the mapping, so a host buffer would be a second copy. For the loader's automatic choice that is waste, and it downgrades.

The only way around it was `--load-mode none`, which reads the entire file into RAM. For Flash-Next that includes a 50.7 GiB per-layer embedding table that is otherwise read lazily from the page cache. On a 188 GiB box that left almost nothing free, and in one run decode collapsed to 8 to 10 t/s under swap pressure.

## The change

The first commit registers each device's host buffer type as an `-ot` target, and adds a flag in the loader that records whether a tensor's buffer type came from an explicit override. The downgrade still applies to the automatic choice, which is unchanged. It is skipped when the user named the host buffer type themselves. No new flag, no API change, default behaviour identical. If `cudaHostAlloc` fails, ggml's host buffer type already falls back to a plain CPU buffer.

The result is that mmap and pinned memory coexist per tensor: experts pinned, everything else mapped, the embedding table still evictable page cache.

| Experts in | Prefill 26k, cold | Prefill 26k, warm | Prefill 131k | Decode short | Decode at 131k | RAM |
|---|---|---|---|---|---|---|
| `CPU` (mmap, pageable) | 166 t/s | not measured (power cut) | 193 t/s | ~16.3 t/s | 9.1 t/s | page cache |
| `CUDA_Host` (this PR) | 330 t/s | **379 t/s** | 300 t/s | 15.9 t/s | 11.3 t/s | 89.6 GB pinned |

Same binary for both rows, master `b96806d` plus the change, Qwen3.8-Flash-Next UD-Q6_K_XL, 261,888 context, ub 2048, f16 KV, 4 expert layers per GPU and the other 40 in host RAM. "Cold" and "warm" refer to the lazily-read embedding table.

A Reddit user ran the first commit on hardware I do not have: RTX 5090, Ryzen 9800X3D, 192 GB DDR5, Windows, a 4.27 bpw Q4_K_M, expert cache on. Prefill went from 227 / 237 / 226 t/s to 627 / 575 / 515 at 47k / 101k / 154k of context, decode stayed flat, and there was no load-time penalty on Windows. Medians of three, their numbers, unchanged.

## Where the load time went

With the experts pinned, load went from 96 seconds to 488. I assumed that was the price of copying 85 GiB into page-locked memory. It was not. The server log's own timestamps split it: metadata 2 seconds, one `cudaMallocHost` of 104 GiB 60 seconds, the copy 450. And `/proc/<pid>/io` showed 106 GiB of block reads against 22 MB through `read()`: the copy was faulting every page of the mapping at queue depth 1, 26.7 million faults at 236 MB/s.

The cause is `--numa distribute`, which applies `POSIX_MADV_RANDOM` to the whole mapping and disables readahead, combined with the loader's host-destination branch being a plain memcpy from that mapping. A single-threaded memcpy of 102 GiB from a warm page cache would be about 20 seconds, so parallelising the copy would have targeted the wrong term.

The second commit reads host-destination tensors from the file with `seek` and `read_raw` instead. Nine lines in `load_all_data`, and the `MADV_RANDOM` in the mmap code stays, because the plain-mmap decode mode depends on it for NUMA page placement.

| | Before | After |
|---|---|---|
| Load to `/health` ok, cold page cache | 490-512 s | 168-183 s |
| Copy phase | 450 s | 104-110 s |
| Block read rate during the copy | 236 MB/s, 4 KiB at queue depth 1 | 1,222 MB/s, ~128 KiB requests |
| Buffer still pinned after load | 107 GB | 107 GB |
| Greedy 256-token reference | | identical |
| 26k prefill, same hour, same profile | 137.1 t/s | 137.9 t/s |
| Plain mmap load, no host override | 24 s | 24 s, path not touched |

The remaining copy phase is one thread doing 128 KiB reads at about half the device's utilisation; a two-to-four thread read would probably halve it again. Not done.

## Status

The PR is open and ready for review. The code was written by an AI coding agent working under my direction on my hardware, which the PR says, because llama.cpp requires that disclosure and because it is true. The measurements, the design decision to honour the explicit override rather than add a flag, and the PR text are mine, and I have walked through the change closely enough to defend it in review. Related items I found while checking for prior work: #26659 (pins the entire mapping with `cudaHostRegister`, a different scope), #26110, #25859, #26448, and #28136 (direct reads for the embedding table, which is the complementary half of this problem).
