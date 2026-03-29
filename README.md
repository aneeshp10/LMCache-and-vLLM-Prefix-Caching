# LMCache-and-vLLM-Prefix-Caching

## 1. Motivation
This experiment evaluates whether LMCache improves inference performance when used alongside vLLM prefix caching. The central question is not whether LMCache works syntactically, but under what workload conditions it produces measurable value.

The motivation comes from a common serving pattern in LLM inference: many requests share a large common prefix, such as a system prompt, a retrieved document, or a long conversation history. In these cases, prefix caching can reduce repeated prefill work if the KV cache for that prefix remains available. However, under memory pressure, GPU-resident KV may be evicted. LMCache is intended to address this by preserving KV outside GPU memory so that it can be reused later rather than recomputed from scratch.

This leads to two practical questions:

Does LMCache improve latency and throughput in a single-node setup with repeated prefixes?
How do these results relate to the larger value proposition of LMCache in distributed inference systems?

## 2. Experimental Setup
### Model and serving stack
Model: Qwen/Qwen2.5-1.5B-Instruct
Serving engine: vLLM
Offload backend: LMCache
Hardware: Google Colab A100 HighRAM

### Conditions compared
Two serving conditions were evaluated:

Condition 1: Prefix caching enabled, LMCache disabled
Condition 2: Prefix caching enabled, LMCache enabled
The goal was to isolate the effect of LMCache while holding the rest of the serving configuration constant.

### LMCache configuration
For LMCache-enabled runs, the server was configured with:

--kv-offloading-backend lmcache
--kv-offloading-size
--disable-hybrid-kv-cache-manager
Environment variables enabled local CPU-backed LMCache storage:

LMCACHE_LOCAL_CPU=True
LMCACHE_MAX_LOCAL_CPU_SIZE=<configured size>
LMCACHE_CHUNK_SIZE=256

### Datasets used
Two workloads were tested:

prefix_repetition

Synthetic benchmark designed to create repeated shared prefixes
Best suited for testing prefix reuse and KV offload behavior
sharegpt

More realistic conversational workload
Prompts are shorter and more diverse
Less favorable to exact prefix reuse
### Memory-pressure tuning
The experiment progressively increased memory pressure by changing:

prefix length
number of prefixes
output length
concurrency
GPU memory utilization
This was necessary because early runs showed that with a small model on an A100, GPU KV capacity was large enough that LMCache mostly added overhead.


## 3. Conceptual Background
What prefix caching does
Prefix caching in vLLM reuses the KV cache for repeated prompt prefixes, but only if that KV remains available in the serving process.

What LMCache adds
LMCache acts as an external KV storage layer. In the local Colab setup, it primarily stores KV in CPU memory. Its value is not that it makes decode inherently faster. Its purpose is to preserve reusable KV when GPU memory is insufficient to keep all useful prefixes resident.

A simple example:

Request 1 and Request 2 both begin with the same 10k-token prefix.
Without LMCache:
if the prefix KV is still on GPU, Request 2 benefits from reuse
if it has been evicted, Request 2 must recompute the full prefix
With LMCache:
the prefix KV can be stored outside GPU
if GPU memory evicts it, the KV can be retrieved rather than recomputed
This means the main expected benefit of LMCache is reduced prefill cost, which is usually reflected in lower TTFT under memory pressure. It is not primarily expected to reduce ITL, because retrieving or transferring KV still incurs overhead.

## 4. Results
### Prefix repetition benchmark
| Condition | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix ON, LMCache OFF | 2.05 | 383.56 | 34416.64 | 41346.97 | 43524.47 | 52700.64 | 93.45 | 100.47 | 152.02 |
| Prefix ON, LMCache ON | 2.03 | 381.94 | 34175.45 | 38473.64 | 39095.18 | 60104.68 | 116.58 | 126.93 | 220.88 |

Interpretation:
Under this extreme repeated-prefix workload, LMCache improved mean and median TTFT:

mean TTFT decreased from 41.35 s to 38.47 s
median TTFT decreased from 43.52 s to 39.10 s
This is the first point in the experiment where LMCache began to show its intended value: preserving reusable prefix KV under high pressure so that repeated requests could avoid some recomputation.

However, this benefit did not translate into better overall serving performance:

ITL worsened materially
request throughput and token throughput were slightly lower
tail latency at p99 TTFT was worse
This suggests that under very high pressure, LMCache can reduce prefill cost, but the added offload/retrieval overhead still degrades decode and end-to-end throughput.


### ShareGPT benchmark

| Condition | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix ON, LMCache OFF | 38.47 | 4923.93 | 44315.41 | 1066.85 | 924.29 | 2625.90 | 20.03 | 11.47 | 45.01 |
| Prefix ON, LMCache ON | 29.05 | 3718.89 | 33470.04 | 1131.26 | 901.48 | 3444.79 | 28.76 | 12.20 | 62.04 |

Interpretation
On ShareGPT, LMCache was consistently worse:

lower throughput
worse mean TTFT
worse p99 TTFT
worse ITL
This is expected. ShareGPT is a more realistic conversational workload, but it has less exact repeated-prefix structure than prefix_repetition. As a result:

prefix reuse is weaker
there is less reusable KV to preserve
LMCache has fewer opportunities to offset its own overhead
In effect, ShareGPT behaved as a realism check rather than a favorable benchmark for LMCache.

## 5. Key Findings
### Finding 1: LMCache does not help by default
In low-to-moderate pressure regimes, LMCache consistently made performance worse. This was observed in early prefix_repetition runs and clearly on sharegpt.

### Finding 2: LMCache begins to help only under extreme repeated-prefix memory pressure
The heavy prefix_repetition configuration was the first scenario where LMCache improved TTFT. This indicates that the system had finally entered a regime where preserving KV externally was preferable to full recomputation.

### Finding 3: TTFT is the main metric where LMCache should be expected to help
This experiment supports the idea that LMCache is mainly a prefill-side optimization. Its primary value is:

preserving large reusable prefixes
reducing recomputation after GPU eviction
improving TTFT under memory pressure
Lower ITL should not be assumed. In practice, ITL may worsen because the offload path introduces retrieval and transfer overhead.

### Finding 4: End-to-end wins are harder to achieve than TTFT wins
Even when LMCache improved TTFT on prefix_repetition, overall throughput still did not improve. This highlights an important distinction:

LMCache can help one phase of inference
but that does not guarantee better overall serving performance

## 6. Limitations
This experiment has several limitations.

Single-node setup
The system under test was a single A100 HighRAM Colab instance. This captures the local CPU-backed offload case, but it does not fully reflect the distributed serving scenarios where LMCache may provide stronger architectural value.

Small model size
Qwen2.5-1.5B-Instruct is relatively small for an A100 HighRAM GPU. As a result, it took substantial tuning to create meaningful KV pressure. A larger model may have revealed clearer crossover points.

Synthetic benchmark dependence
The clearest LMCache benefit appeared only on prefix_repetition, a synthetic benchmark explicitly designed to create repeated prefixes. This is useful for isolating mechanism, but it is not representative of every production workload.

ShareGPT is not strongly reuse-oriented
ShareGPT is more realistic, but it does not strongly stress exact prefix reuse. Therefore, it is better interpreted as a realism check than as an ideal benchmark for LMCache.

## 7. Relevance to Large-Scale Distributed Inference
The Colab experiment mainly shows LMCache in its simplest form: CPU-backed KV preservation on a single node. That is only part of the story.

In a large-scale distributed inference system, LMCache can offer broader value because KV reuse is no longer constrained to a single GPU worker.

This matters in architectures where:

requests with the same prefix may hit different replicas
GPU-local KV is fragmented across workers
prefill and decode are disaggregated
useful KV must survive beyond the lifetime of a single worker’s local cache
In those systems, LMCache can serve as an external KV layer that enables:

reuse across workers
transfer of prefix state between stages
reduced recomputation in multi-instance deployments
better utilization when identical or similar contexts recur across the fleet
This is an important distinction:

In a single-node setup, LMCache competes directly with fast local GPU reuse and often loses unless pressure is severe.
In a distributed system, LMCache may enable reuse that local prefix caching alone cannot provide at all.
So while the Colab results show only modest or negative gains, they should not be interpreted as evidence that LMCache lacks value in production. Rather, they show that on a single strong GPU with a small model, its benefits are narrow and workload-dependent.
