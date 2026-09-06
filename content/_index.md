---
title: "Inovello"
tagline: "Running models that don't fit. Writing down what it costs."
---

I'm Evans. I run large mixture-of-experts models on a dual-Xeon box with two RTX 3090s and 192 GB of DDR4. The expert weights live in host memory, everything else on the GPUs, and the interesting work is in the gap between the two.

What I publish is measurements: what a change did to decode and prefill, what broke, and what it took to find out. The code I've sent upstream to llama.cpp came out of the same runs.
