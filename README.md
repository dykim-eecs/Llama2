
## **Verification Environment**

* Vitis, Vitis HLS 2022.2
* Xilinx U250

Test on  Vitis 2025 & kv260

### **Minimum Board Requirements for Running llama2.c**

| **BRAM** | **DSP** | **FF**  | **LUT** | **URAM** |
| -------- | ------- | ------- | ------- | -------- |
| 300      | 200     | 100,000 | 100,000 | 0        |

## **Introduction**

This project is inspired by the [HLSTransform](https://arxiv.org/abs/2405.00738) paper, which provided the foundational ideas and framework.

Our objective is to explore an end-to-end FPGA implementation of a large language model (LLM), targeting the **Llama2-7B model**.

## **Purpose**

Modern LLMs consist of a vast number of parameters. For example, Xai’s Grok4 model has **1.7 trillion parameters**, requiring **6.8 TB** of weight storage alone.

While implementing matrix multiplication accelerators (like NPUs) is useful, our goal is to **realize the full LLM pipeline** on FPGA hardware. To make this feasible, we focus on **Llama2-7B**, a smaller yet representative model.

## **Llama2 vs. Llama2.c**

**Llama 2** is Meta’s official large-scale model, built with **PyTorch** and **Transformers**, designed for GPU/TPU training and inference. It is distributed via Hugging Face in multiple scales (7B, 13B, 70B) with checkpoints and configs.

In contrast, **llama2.c** is a lightweight C implementation created by Karpathy and the community. It supports **inference only**, running directly on CPUs. Models are exported to `.bin` format with options for FP32 or INT8 quantization, making it suitable for **embedded systems and FPGA experimentation**.

| Category    | **Llama 2 (Meta)**                   | **llama2.c (C Implementation)**              |
| ----------- | ------------------------------------ | -------------------------------------------- |
| Developer   | Meta (official)                      | Karpathy / community (unofficial)            |
| Language    | Python (PyTorch, Transformers)       | C (single-file, CPU-centric)                 |
| Purpose     | Large-scale training & inference     | Lightweight inference, education & research  |
| Model Sizes | 7B, 13B, 70B                         | Small/quantized versions                     |
| Format      | PyTorch checkpoint, Hugging Face Hub | `.bin` (custom header, quantization support) |
| Environment | GPU/TPU (high-performance)           | CPU, FPGA, embedded systems                  |


**Dataset Reference:**

* [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)

| **Model** | **dim** | **n\_layers** | **n\_heads** | **n\_kv\_heads** | **Max Context** | **Parameters** | **Val Loss** | **Download**                                                                               |
| --------- | ------- | ------------- | ------------ | ---------------- | --------------- | -------------- | ------------ | ------------------------------------------------------------------------------------------ |
| 260K      | 64      | 5             | 8            | 4                | 512             | 260K           | 1.297        | [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)            |
| OG        | 288     | 6             | 6            | 6                | 256             | 15M            | 1.072        | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin)   |
| 42M       | 512     | 8             | 8            | 8                | 1024            | 42M            | 0.847        | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin)   |
| 110M      | 768     | 12            | 12           | 12               | 1024            | 110M           | 0.760        | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

**Note**: Instead of directly downloading Hugging Face models, you must adapt them to your hardware. In `/llama2.c/export.py`, the function `version2_export(model, filepath, group_size=64)` allows you to configure **group size** and **quantization bits**.

## **HLS Code Structure**

📑 **Source Files**

```
├─ config.h
├─ forward.cpp
├─ forward.h
└─ typedefs.h
```

📑 **Test Bench**

```
├─ model.bin
├─ testbench.cpp
└─ tokenizer.bin
```

Repository: [https://github.com/dykim-eecs/Llama2](https://github.com/dykim-eecs/Llama2)
Pre-trained Weights: [Google Drive Link](https://drive.google.com/drive/folders/1L1EirYH7ygG8Vnf5wjBya-lNNzCfKDmX?usp=drive_link)

**Performance & Resource Estimates**

| Module / Loop         | Violation Type   | Slack | Latency (cycles) | Latency (ns) | Iteration Latency | Interval | Trip Count | Pipelined | BRAM | DSP  | FF      | LUT     | URAM |
| --------------------- | ---------------- | ----- | ---------------- | ------------ | ----------------- | -------- | ---------- | --------- | ---- | ---- | ------- | ------- | ---- |
| **forward**           |                  | -0.10 | 4,562,698        | 4.563E7      | -                 | 45,62699 | -          | no        | 41   | 3136 | 617,389 | 684,944 | 0    |
| forward\_Pipeline\_1  |                  | -     | 771              | 7.710E3      | -                 | 771      | -          | no        | 0    | 0    | 519     | 112     | 0    |
| matmul\_768\_32000\_s | **II Violation** | -     | 864,259          | 8.643E6      | -                 | 864,259  | -          | no        | 0    | 384  | 67,336  | 59,671  | 0    |
| main\_forward\_loop   | **II Violation** | -     | 3,691,968        | 3.692E7      | 307,664           | -        | 12         | no        | -    | -    | -       | -       | -    |

**≈22 tokens/s**

## **Analyze**

[Rmsnorm Analyze](https://www.notion.so/Rmsnorm-Analyze-25584215ac3680e6b59ae6e603022f6a?pvs=21)

## **Optimization Tips**

### **For Low-Resource Boards**

1. **Remove all pragmas**
   Pragmas usually reduce latency by increasing resource usage. On low-resource boards, it is often better to disable them.
2. **Adjust Group Size (GS)**

   * **Definition**: Quantization group size.
   * Smaller GS → Higher accuracy, but increased latency and resource usage.
   * Larger GS → Faster and lighter, but reduced accuracy.

**Trade-off Summary**

| Item              | GS=64      | GS=24      |
| ----------------- | ---------- | ---------- |
| Latency (cycles)  | 62,560,000 | 76,640,000 |
| Latency (ns)      | 6.26e8     | 7.66e8     |
| Iteration Latency | 1,955      | 2,395      |
| BRAM              | 308        | 338        |
| DSP               | 142        | 150        |
| FF                | 101,649    | 67,243     |
| LUT               | 171,924    | 98,567     |
| URAM              | 0          | 0          |

* **GS=64** → Lower latency, higher FF/LUT usage, reduced accuracy.
* **GS=24** → Higher latency, lower FF/LUT usage, improved accuracy.

**Note:** If you modify the GS value, be sure to update `GS=64` in *config.h* and also adjust the line `constexpr int num_groups = S / 64;` in *forward.h* (within the quantize function) to use the new GS value instead of 64.

### **For High-Resource Boards**

Here, the goal is **low latency**.

* Check pipeline intervals (II). If II ≠ 1, identify bottlenecks.
* Possible causes:

  1. **Memory port limitations** → Try memory partitioning, LUTRAM mapping, or switching from RAM\_1P to RAM\_2P.
  2. **DSP conflicts** → Adjust DSP latency or rebalance arithmetic units.

Logs will highlight whether II violations are caused by **load operations** (memory-bound) or **DSP allocation** (compute-bound).

(Examples of logs are included in the detailed section above.)

## **Final Goal**

The ultimate objective is to **optimize RTL (Verilog) generated by HLS** and implement it on real silicon.

This process involves multiple toolchains, but by iteratively refining designs, we aim to tape out a working chip by **early next year**.
