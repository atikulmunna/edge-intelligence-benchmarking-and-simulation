# Edge-LLM-Control-Benchmark: Analytical Framework

![Platform](https://img.shields.io/badge/Platform-Windows%20CUDA%20vs%20Pi5-red) ![Analysis](https://img.shields.io/badge/Analysis-Wilson%20Score-green)


## Analytical Methodology

### 1. The "Specialists vs. Champions" Strategy

A key challenge in Edge AI is that "One Size Does Not Fit All." A Raspberry Pi cannot run massive generalist models effectively due to RAM constraints. To create a fair but rigorous benchmark, we adopted an asymmetric testing strategy:

#### Phase A: The Desktop Baseline (High-Performance Reference)
* **Environment:** Desktop PC (Windows, CUDA-accelerated).
* **Models:** `CodeLlama-7b`, `DeepSeek-Coder-6.7B`, `GPT-2`, `Mathstral-7B`, `Mixtral-8x7B`.
* **Method:** We benchmarked a suite of recognized specialized models in a high-performance CUDA environment. Each model was evaluated against the full cross-domain prompt set to verify its capabilities.
* **Goal:** To establish a "performance ceiling" for standard 7B/8B architectures. Rather than searching for a new champion, this phase provided the ground-truth latency and reliability metrics needed to calculate the exact hardware penalty (slowdown factor) incurred when moving these specific model classes to the edge.

#### Phase B: The Edge Challenger (The Specialists)
* **Environment:** Raspberry Pi 5 (8GB RAM, CPU-only).
* **Models:** `SmolLM2-1.7B`, `DeepSeek-R1-7B`, `Llama-3.1-8B`, `Gemma-3-4b`.
* **Method:** Due to hardware limitations, we could not run a cross-domain sweep. Instead, we selected **one specialized model** per domain (e.g., DeepSeek-R1 for Code) that fit within the 8GB memory budget (using swap).
* **Goal:** To measure if a "Specialist" on the Pi could achieve the same *correctness* as the "Champion" on the Desktop, and at what cost to latency.

---

### 2. The Comparative Framework

We established a strict comparison protocol to measure the trade-offs of edge computing. We compare the Pi's **Specialist** directly against the Desktop's **Champion** for that specific task.

| Domain | Pi Model (Specialist) | Pi Time | WSL Model (Champion) | WSL Time | Slowdown Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **General** | **SmolLM2-1.7B** | 7.7s | **GPT-2** | 0.4s | **19x** (Acceptable) |
| **Code** | **DeepSeek-R1-7B** | 2,245s | **DeepSeek-6.7B** | 1.0s | **2,200x** (Extreme) |
| **Writing** | **Llama-3.1-8B** | 2,378s | **Mathstral-7B*** | 2.8s | **850x** (Extreme) |
| **Math** | **Gemma-3-4b** | 3,592s | **Mathstral-7B** | 2.8s | **1,280x** (Extreme) |

*> **Note:** For the Writing domain, **Mathstral-7B** was chosen as the desktop baseline over Mixtral-8x7B. While Mixtral is a larger "generalist," Mathstral provided a comparable 7B-to-8B comparison point that ran efficiently on the desktop, offering a clearer view of the hardware-induced slowdown without the noise of desktop memory bottlenecks.*

---

### 3. Statistical Normalization: The Wilson Score Interval

A critical challenge in benchmarking LLMs is the "Sample Size Illusion." A model scoring 6/6 (100%) on a small test set is not necessarily perfect.

To address this, we rejected raw accuracy percentages in favor of the **Wilson Score Interval (Lower Bound, 95% Confidence)**.

#### Why we used it:
* **Penalizes Small Samples:** It mathematically accounts for the uncertainty inherent in small datasets (N=6).
* **Reliability Metric:** Instead of asking "What *was* the accuracy?", it asks "What is the *lowest* accuracy we can guarantee for the future?"
* **Differentiation:** It successfully differentiated between models that were "lucky" and models that were robust, especially in strict formatting tasks.

$$w^- = \frac{\hat{p} + \frac{z^2}{2n} - z\sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

---

### 4. Hardware Telemetry Analysis

We went beyond simple "Tokens Per Second" metrics by correlating performance with hardware telemetry logs (`telemetry.json`):

* **The "Swap Wall" Detection:** By tracking `swap_percent` over time, we identified the exact parameter threshold (approx. 2B) where inference shifts from "Compute Bound" to "I/O Bound."
* **Thermal Throttling:** We monitored `cpu_temp` to ensure latency spikes were due to architectural limits (RAM/Swap) rather than thermal throttling.

---

### 5. Architectural Validation: "BigBoss" Pipeline

The analysis validated that our custom "BigBoss" pipeline successfully implements core concepts from **vLLM** (Virtual Large Language Model) on edge hardware:

* **Memory Paging:** Using NVMe-backed swap to handle model weights larger than physical RAM.
* **Checkpoint Streaming:** Dynamically loading model shards to prevent OOM kills during initialization.
* **Graceful Failure:** Custom handlers to recover from freeze states common in edge deployments.


