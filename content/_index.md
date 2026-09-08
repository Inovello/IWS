---
title: "Inovello"
tagline: "Running models on hardware that shouldn't run them. Writing down what it costs."
---

I'm Evans. I run language models locally on a dual-Xeon box with two RTX 3090s and 192 GB of DDR4. Sometimes that means a mixture-of-experts model far too big for 48 GB of VRAM, with its expert weights out in host memory and the interesting work happening in the gap. Sometimes it means a dense model that fits on the cards and the question is how fast it can go.

What I publish is measurements: what a change did to decode and prefill, what broke, and what it took to find out. The code I've sent upstream to llama.cpp came out of the same runs.
