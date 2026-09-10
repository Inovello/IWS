---
title: "The box"
slug: hardware
date: 2026-09-06
updated: 2026-09-10
summary: "Every number on this site comes from this machine unless the writeup says otherwise."
---

Every number on this site comes from this machine unless the writeup says otherwise. It is a used dual-socket Xeon workstation with two consumer GPUs bolted on, bought for the memory capacity rather than the compute.

## Specification

| Component | Value |
|---|---|
| CPU | 2x Intel Xeon E5-2696 v4 (Broadwell), 22 cores each, 44 cores / 88 threads total |
| Host memory | 192 GB DDR4-2133 ECC, 6x 32 GB (3 per socket), 188 GiB usable. The quad-rank LRDIMMs were replaced in September 2026; part 3 ran on four DIMMs while two were out |
| Measured memory bandwidth | ~63 GB/s aggregate interleaved, ~31.6 GB/s per NUMA node, measured on the LRDIMMs |
| GPU | 2x NVIDIA RTX 3090, 24 GB each, 48 GB total, PCIe 3.0 x16 |
| GPU topology | One card per socket: GPU0 on NUMA node 0, GPU1 on NUMA node 1, no NVLink |
| OS | Ubuntu Server 24.04, kernel 6.8 |
| CUDA | Toolkit 12.0, driver 595.84 |
| Inference | llama.cpp (production), vLLM for the dense models, ik_llama.cpp built for comparison |
| Storage | NVMe for the model files (~1.2 GB/s single-threaded read measured during loads) |

## How the offload works

The models I care about are mixture-of-experts models much larger than 48 GB. The routed expert weights, which are most of the bytes, sit in host RAM. Attention, the recurrent layers, the shared expert, embeddings, and the output head sit on the GPUs, split by layer across the two cards.

During prefill, llama.cpp streams each CPU-resident expert layer to the GPUs once per micro-batch, so prefill speed is bounded by PCIe 3.0 and by whether the host memory is pinned. During decode, each token gathers its ten active experts per layer from host RAM, so decode speed is bounded by memory latency and by how many of those gathers a GPU-side cache can absorb. The writeups are mostly about moving those two bounds.

Two things about this box shape every result:

- The two sockets are separate NUMA nodes and each GPU hangs off a different one. Where a page lands in memory can be worth 35% of decode. Everything runs under `numactl --interleave=all` for the pinned buffers and `--numa distribute` for the compute threads.
- The quad-rank LRDIMMs that were in the box until September 2026 ran hot. Without forced air over the DIMM banks the memory controller started throttling after a few minutes of sustained load, at a sensor reading of about 78 C. A fan zip-tied over each bank fixed it, and the fans stayed when the DIMMs were replaced. Part 2 of the Flash-Next writeups has the measurements.
