---
title: "The expert cache moves out of the way while the prompt runs"
series: flash-next
slug: qwen3-flash-next-2x3090-phase-memory
date: 2026-09-10
kind: writeup
part: 4
summary: "Releasing the expert cache and the decode buffers for the length of a prompt lets prefill run at a 2048-token micro-batch: 2.2x faster at 8k and 2.4 to 2.5x at 37k to 119k, with decode unchanged and a 240-pair quality screen finding no consistent regression."
stats:
  - label: Prefill at 8k
    value: "2.24x"
    accent: true
  - label: Prefill at 119k
    value: "2.54x"
  - label: Decode
    value: "-1 to +2%"
  - label: Matched pairs
    value: "240"
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
  - k: Prefill ubatch
    v: "2048, transaction mode"
  - k: Server ubatch
    v: "512"
  - k: KV
    v: f16
  - k: RAM
    v: 192 GB, six DIMMs (quality screen on four)
  - k: CUDA
    v: "12.0"
---

Part 3 ended with a profile that said prefill on this box was mostly waiting for expert weights to cross PCIe, and that a faster top-k had nothing to say about it. This part is about that wait, and about a compromise I had been carrying since part 2 without labelling it.

The short version: the expert cache that makes decode fast sits idle for the whole of every prompt, holding the VRAM that a larger prefill micro-batch needs. So the server now gives that memory up when a prompt starts and takes it back before the first generated token. Prefill went from 100 to 224 t/s on an 8k document, and from about 85 to about 210 t/s at 37k and 119k. Decode did not move.

Then the same 480-request quality screen as part 3, because a change that touches memory management under the model is exactly the kind that can be wrong quietly.

## Setup

- Box: [dual Xeon E5-2696 v4, 2x RTX 3090 on PCIe 3.0](/hardware/), back on all six 32 GB DIMMs. The quality screen predates their return and ran on four; every table says which
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL, expert cache at 150 slots, shared Q8_0 MTP head at draft depth 3 on the second GPU
- Allocated context 261,888, f16 KV, server micro-batch 512, prefill micro-batch 2048 on the candidate, CUDA toolkit 12.0
- Throughput at 8k: a fresh 8,192-token document followed by a 2,048-token continuation, greedy, two fresh processes per arm
- Throughput and quality at depth: the part 3 corpus, 38 questions at about 37k tokens and 42 at about 119k, production sampling (temperature 0.7, top-p 0.8, top-k 20, min-p 0, thinking on)

## Where this sits after part 3

[Part 3](/writeups/qwen3-flash-next-2x3090-topk/) was a decode change measured on the long-document corpus. This part is a prefill change, measured on the same corpus and on a fixed 8k document, against the build from part 3 as the control. The control here is the configuration I was serving myself until this went in.

The DIMMs came back in the middle of this work. The quality screen and the first throughput numbers are four-DIMM; the re-baseline at 8k and the depth throughput are six-DIMM, with both arms on the same memory each time. There is a section on what the two extra channels did, because it was not what I expected.

## Why prefill was paying for decode

In part 2 the launch line had `-ub 512` in it and I did not say why. The reason was the cache. A 2048-token micro-batch needs about 7.3 GiB of compute buffer on each GPU; 512 needs 1.9 GiB. On a 24 GB card the difference is roughly fifty cache slots, and I wanted those slots for decode.

What that costs is close to linear. With the experts in host RAM, every prefill micro-batch streams each expert layer over PCIe once, and that upload costs about the same whether the micro-batch carries 512 tokens or 2048. An 8k prompt at 512 uploads the expert set sixteen times; at 2048, four times. So the choice was three times the prefill or a third of the cache, and I took the cache.

The cache only ever serves batches of eight tokens or fewer, which is decode and the MTP verify batches. During a prompt it does nothing except hold memory.

## What changes during a prompt

When a prompt arrives, the server now runs a transaction around it. It quiesces the target and draft contexts, invalidates the CUDA graphs, releases the cache's device slots, the compute workspaces and the CUDA pool allocations, then reserves compute buffers sized for a 2048-token micro-batch and runs the whole prompt at that size. When the prompt is done and the draft has caught up, it frees those buffers, restores the 150 slots and the decode workspaces, and rebuilds the graphs. Decode then runs the code it always ran. Nothing about the logical batch or the sequence state changes; only the physical compute capacity does.

The transaction is all-or-nothing. It will only begin with the cache in its ready phase at exactly 150 slots and only end with the generation counters matching what it saw at the start. Any mismatch fails the call and the server reports it rather than continuing half restored. The server enables the whole thing from one environment variable, `LLAMA_PHASE_PREFILL_UBATCH=2048`, and refuses to start unless the configuration is the one this was qualified on: server micro-batch 512, batch 4096, one sequence, and the MTP head as the only speculative decoder.

The fixed cost is the transition itself, about 1.5 s to release and 1.1 s to restore per prompt in an untraced run. That is why the gain is smaller on short prompts than on long ones.

## Comparing against the previous production

Two arms, each a fresh server process, production stopped for the arm and restored after it:

| Arm | Build | Prefill micro-batch |
|---|---|---|
| Control | part 3 build, `-ub 512`, cache 150 | 512 |
| Candidate | this change, `-ub 512`, cache 150, transaction mode | 2048 |

At 8k the order was control, candidate, candidate, control, two processes per arm. At depth it was candidate then control, one process each, seed 1. The quality screen ran candidate on seeds 1, 2, 3 and then a fresh control on the same seeds, not rotated, because the candidate was the required evidence that night and a same-day control was the comparison I wanted. Every candidate launch was checked from the process's own environment for the two phase variables, and every candidate server log was scanned after its run for the transaction's failure strings and counted for begin and end markers: 80 begins and 80 ends per arm, no failures, in all four candidate arms.

## Prefill and first token

Six DIMMs, same day for each row pair. The 8k row is the median of two fresh processes; the depth rows are the one fresh prefill of the document at that depth, since later questions in a conversation reuse the prefix.

| | Control | Candidate | Ratio |
|---|---|---|---|
| 8k document, prefill | 99.9 t/s | 223.7 t/s | 2.24x |
| 8k, time to first token | 82.2 s | 36.8 s | 0.45 |
| 37k document, prefill | 88.1 t/s | 212.6 t/s | 2.41x |
| 37k, time to first token | 424 s | 176 s | 0.41 |
| 119k document, prefill | 81.3 t/s | 206.5 t/s | 2.54x |
| 119k, time to first token | 1461 s | 575 s | 0.39 |

The 8k control ranged 96.8 to 103.1 t/s across its two processes; the two candidate processes agreed to 0.01 t/s. Time to first token includes the transitions.

The four-DIMM quality screen measured prefill at depth on all three seeds, and it told the same story: 83.5 to between 202 and 211 t/s at 37k, 79.2 to between 199 and 207 t/s at 119k, 2.4 to 2.6x on every seed.

## Decode

The number this change had to protect. At 8k, the median over the 2,048-token continuation, two processes per arm:

| | Control | Candidate | Ratio |
|---|---|---|---|
| Overall decode | 33.4 [33.3, 33.6] t/s | 34.3 [34.0, 34.5] t/s | 1.024 |

At depth on six DIMMs, the median over every request of that depth on seed 1:

| Depth | Control, median [min, max] | Candidate, median [min, max] | Ratio |
|---|---|---|---|
| 37k, 38 requests | 41.7 [30.5, 47.5] t/s | 41.2 [29.8, 48.0] t/s | 0.988 |
| 119k, 42 requests | 33.9 [30.5, 38.1] t/s | 33.9 [30.0, 38.2] t/s | 1.001 |

The four-DIMM screen had a predeclared decode gate: every seed-and-depth cell at or above 0.97 of its control. Five of six cells passed, at 0.98 to 1.02. One did not, seed 3 at 37k, at 0.964. That arm was the rerun launched a minute after a power cut, its whole sequence sits uniformly about 1.2 t/s below the other two candidate arms, and its paired control happened to be the fastest of the three controls. The spread between one arm's own seeds is 2.5 to 3.3% at that depth, which is the size of the miss. I report it as a failed gate because that is what it was; I do not think it is a decode regression, and neither the six-DIMM re-baseline nor the seed-1 rerun reproduced it. Pooled over all 240 candidate requests against all 240 control requests, decode was level at 1.007.

MTP acceptance was 0.79 to 0.83 in every arm, both configurations.

## Whether the answers got worse

Same corpus, grader and frozen key as part 3, four DIMMs, three seeds, 240 matched pairs.

| Seed | Control | Candidate |
|---|---|---|
| 1 | 80/80 | 79/80 |
| 2 | 78/80 | 79/80 |
| 3 | 78/80 | 79/80 |
| Total | 236/240 | 237/240 |

Matched within seed: the candidate was worse on two, better on three, identical on 235. No question regressed on more than one seed. The two candidate-only misses were question 34 at 119k on seed 1 and question 33 at 119k on seed 2. Question 33 is the twelve-paths recall item that part 3 already could not resolve, and both arms miss it on some seed here too. Question 34 was failed by part 3's control on its seed 3. I had the seed-1 answers reviewed with the arm labels hidden, and the reviewer agreed the candidate's answer was incomplete, so it stays wrong.

The seed-1 rerun on six DIMMs, same corpus, came out one worse, 78 identical, one better, with the one miss being question 33 again.

There is a sharper check than answers. The first-token logits of a request on the candidate path differ from a control's by at most 1.51, mean 0.22, across the 248k vocabulary, and the argmax is the same. For scale, changing the micro-batch from 512 to 2048 with nothing released moves the same request's logits by at most 1.81, mean 0.27. So the release and restore adds less than the batch-shape change that any micro-batch change already carries, and the model is not bit-identical between micro-batch sizes in the first place.

What the screen supports is the same narrow statement as part 3: no consistent regression on this suite, with one recall item unresolved. It is a bounded screen on one corpus, not equivalence.

## What the extra DIMMs did

The six-DIMM re-baseline against my four-DIMM numbers, same builds:

| | Four DIMMs | Six DIMMs | Change |
|---|---|---|---|
| Control, 8k prefill | 96.7 t/s | 99.9 t/s | +3% |
| Candidate, 8k prefill | 187.5 t/s | 223.7 t/s | +19% |
| Control, 37k / 119k prefill | 83.5 / 79.2 t/s | 88.1 / 81.3 t/s | +5.5% / +2.7% |
| Candidate, 37k / 119k prefill | 211.3 / 206.6 t/s | 212.6 / 206.5 t/s | +0.6% / 0% |

Two more memory channels helped the configuration that streams the expert set four times per 8k prompt far more than the one that streams it sixteen times, which fits the sixteen-upload arm being bound by the PCIe transfer itself. At depth the channels did nearly nothing for either. Attention grows with context and runs on the GPU, so the share of prefill that host bandwidth can touch shrinks as the document gets longer. The depth rows are one run each on different days, so treat the small deltas as direction, not measurement. The 8k rows are two processes each.

## What this changes and what it leaves open

For this box the compromise is gone: the cache stays at 150 slots and prefill runs at 2048. It has been the configuration I serve since it passed the screen.

It is narrow on purpose. The code accepts a prefill micro-batch of exactly 512 or 2048, a cache of exactly 150 slots, one sequence, batch 4096 and the MTP head as the only draft, and refuses anything else at startup. Those are the values it was qualified on, not the limits of the idea. It is CUDA only, one machine, one model, one quantization, 8k to 119k. I have not measured above 119k, other models, other slot counts, or the no-MTP configuration.

The profile that motivated this is also what says where the time still goes. On the new configuration an 8k prompt still uploads 278 GB to the first GPU, which is 3.6 times the size of the expert set, and that upload is 72% of the 37 seconds to first token. Kernels are under 4 seconds and nothing overlaps them. So the micro-batch change bought the easy factor of four, and the expert set is still crossing PCIe more than three times per prompt.

The source is on [my fork as `flashnext-e06`](https://github.com/Inovello/llama.cpp/tree/flashnext-e06): the part 2 branch plus this change and a few inert diagnostic switches. The notes at the root of the branch have the launch line the numbers were measured with and the values the code insists on. The quality corpus is private, as before.

## Next

Reuse. If the expert set can cross PCIe once per prompt instead of 3.6 times, the same profile projects time to first token at 8k somewhere near 18 seconds, which is a hypothesis, not a number. The patch for that exists and has been reviewed; it has not been built or measured, and the bar for it is the same as here: at least 1.5x on prefill, decode within 3%, and the screen.
