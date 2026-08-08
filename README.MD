# 🚀 GPU Microarchitecture & Performance Profiler

An automated data-processing and visualization pipeline designed to parse, analyze, and diagnose hardware performance bottlenecks from NVIDIA Nsight Compute (`NCU`) telemetry data. 

This tool helps systems and hardware architects quickly transition from raw microarchitectural metrics to actionable performance insights, specifically targeting **Speed of Light (SOL)** limits, **Occupancy Limiters**, and **Warp Stall distributions**.

---

## 📊 Designed Features

*   **Automated NCU CSV Ingestion:** Automatically skips application stdout and unit-metadata rows, parsing complex multi-metric NVIDIA Nsight Compute exports into structured Pandas DataFrames.
*   **Roofline & SOL Analysis:** Evaluates Compute (SM) versus Memory (DRAM) throughput to classify whether a kernel is Compute-Bound or Memory-Bound.
*   **Warp Stall Diagnostics:** Parses and ranks microarchitectural stall reasons (e.g., *Long Scoreboard*, *MIO Throttle*, *Barriers*) to pinpoint pipeline starvation causes.
*   **Occupancy Limiters Breakdown:** Maps resource utilization constraints (Registers, Shared Memory, and Block/Warp limits per Streaming Multiprocessor).
*   **Automated Hardware Diagnosis Engine:** Generates heuristic-based strategic recommendations directly inside the visual report based on real-time hardware performance counters.
*   **Data Science Presentation Dashboard:** Generates a modern, Seaborn-styled 2x2 multi-plot analytics report saved automatically as a high-resolution `.png`.

---

## 🛠️ Tech Stack & Dependencies

*   **Python 3.x**
*   **Pandas** (Data cleaning, transformation, and statistical aggregation)
*   **Seaborn & Matplotlib** (Professional-grade statistical visualization and reporting dashboards)

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/gpu-hardware-profiler.git](https://github.com/YOUR_USERNAME/gpu-hardware-profiler.git)
cd gpu-hardware-profiler

2. Install Dependencies

pip install pandas matplotlib seaborn

3. Generate NCU Metrics CSV

Run your compiled CUDA executable through the Nsight Compute CLI and export the raw telemetry data:
Bash

ncu --csv --page raw --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,regex:smsp__warp_issue_stalled ./your_cuda_executable > ncu_metrics.csv

4. Run the Profiler Pipeline

Ensure ncu_metrics.csv is in your working directory, then execute:
Bash

python gpu_bottleneck_analyzer.py

📈 Sample Visual Output

The script generates a comprehensive analytical report (gpu_architecture_report.png) featuring:

    Speed of Light (SOL) Utilization (Compute vs. Memory bandwidth capacity).

    Warp Stall Distribution (Sorted categorical breakdown of active warp stalls).

    Occupancy Limiters (Hardware bottlenecks restricting block/warp scaling).

    Automated Hardware Diagnosis Box (Targeted optimization strategies).# 🚀 GPU Microarchitecture & Performance Profiler

An automated pipeline that parses, analyzes, and visualises NVIDIA Nsight Compute (**NCU**) telemetry data.  
It turns raw micro‑architectural metrics into actionable insights, focusing on:

* **Speed‑of‑Light (SOL) limits** – compute vs. memory throughput  
* **Occupancy limiters** – registers, shared memory, block/warp counts  
* **Warp‑stall distributions** – pipeline starvation causes  

---

## 📊 Core Features

| Feature                     | Description |
|-----------------------------|-------------|
| **NCU CSV ingestion**       | Skips stdout rows, parses multi‑metric exports into clean `pandas` DataFrames |
| **Roofline & SOL analysis** | Classifies kernels as *Compute‑Bound* or *Memory‑Bound* |
| **Warp‑stall diagnostics** | Ranks stall reasons (Long Scoreboard, MIO Throttle, Barriers, …) |
| **Occupancy breakdown**     | Shows which resource (registers, shared memory, block/warp limits) caps SM occupancy |
| **Heuristic diagnosis**     | Generates optimisation recommendations directly in the report |
| **2×2 Seaborn dashboard**   | Saves a high‑resolution PNG with all plots |

---

## 🛠️ Tech Stack & Dependencies

| Component   | Version |
|-------------|---------|
| Python      | 3.8+    |
| pandas      | latest  |
| matplotlib  | latest  |
| seaborn     | latest  |

Install with:

```bash
pip install pandas matplotlib seaborn
```

---

## ⚙️ Quick‑Start Guide

1. **Clone the repo**

   ```bash
   git clone https://github.com/YOUR_USERNAME/gpu-hardware-profiler.git
   cd gpu-hardware-profiler
   ```

2. **Collect NCU metrics**

   ```bash
   ncu --csv --page raw \
       --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, \
                dram__throughput.avg.pct_of_peak_sustained_elapsed, \
                regex:smsp__warp_issue_stalled \
       ./your_cuda_executable > ncu_metrics.csv
   ```

3. **Run the analysis**

   ```bash
   python gpu_bottleneck_analyzer.py
   ```

   The script expects `ncu_metrics.csv` in the current directory and produces `gpu_architecture_report.png`.

---

## 📂 Repository Layout

```
gpu-hardware-profiler/
├─ gpu_bottleneck_analyzer.py   # main pipeline
├─ utils/                       # helper modules (CSV parsing, plotting)
├─ data/                        # example CSVs (optional)
├─ .gitignore                   # ignored files (see below)
└─ README.md
```

---

## 📜 License

MIT License – feel free to fork, modify, and use in your projects.

--- 

```
