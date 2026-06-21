# 🏆 Hybrid Edge AI OS: Real-Time Resource Protection & App Preloading

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Torch-CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, localized deployment framework utilizing a dual-brain Deep Learning pipeline directly on-device via Android Debug Bridge (ADB). The system targets low-latency resource management by proactively mitigating memory thrashing using Reinforcement Learning while parallelly boosting application load times via an Attention-based LSTM predictor.

---

## 🧠 System Architecture Overview

The framework orchestrates a closed-loop environment sensing cycle dividing decisions between two deep networks:

                                                                              ```text
                                             ┌────────────────────────────────────── ───┐
                                             │        Android Device (Via ADB)          │
                                             └────┬─────────────────────────────── ▲────┘
                                                  │ State:                         │ Action:
                                                  │ RAM Headroom, Apps Run         │ Kill / Preload
                                                  ▼                                │
                                    ┌───────────────────────────┐    ┌─────────────┴─────────────┐
                                    │   🛡️ RL Protector (DQN)   │    │    🔮 Oracle Predictor    │
                                    │  Threshold Warning Track  │    │  Deep Context Attention   │
                                    └───────────────────────────┘    └───────────────────────────┘         


1. **The Protector (Deep Q-Network):** Constantly audits real-time system RAM consumption via telemetry loops (`dumpsys meminfo`). If headroom scales past a dynamically calculated hazard barrier ($\le 15\%$ available buffer), the DQN dictates high-priority termination vectors to stabilize system performance.
2. **The Predictor (LSTM with Attention):** Examines historical execution sequence parameters provided in `full_ml_ready_app_usage.csv` to map upcoming app usage expectations. If predictability parameters align, hidden system background activities launch application frameworks before user invocation.

---

## 📊 Performance Benchmarks

The hybrid engine successfully clears all structural optimization targets set forth in edge computing evaluations. 

* **App Launch Acceleration:** Boosted execution latency by **20.5%** (shattering the original target milestone of 10.0%).
* **Memory Thrashing Mitigation:** Decreased terminal RAM depletion crashes by **100.0%** (surpassing the stability requirement threshold of 50.0%).

The verified execution breakdown metrics are plotted below:

![Final KPI Dashboard](Final_KPI_Dashboard.png)

---

## 📁 Repository Contents

* `benchmark.py`: Main thread executor script containing system telemetry listeners and localized ADB manipulation routines.
* `full_ml_ready_app_usage.csv`: Normalized mobile interaction sequences containing user contextual telemetry indicators.
* `Final_KPI_Dashboard.png`: Visual statistical evaluation outlining performance against expected metrics.
* `models/`: Production network checkpoints housing standard `label_encoder.pkl` state objects alongside model weights.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Verify your Android Device has **Developer Options** enabled and **USB Debugging** active over a physical or TCP connection.

Ensure your path pointers accurately register your Android SDK setup (edit the `ADB_PATH` global string in `benchmark.py` if necessary).

```bash
# Clone the repository
git clone [https://github.com/yourusername/hybrid-edge-ai-os.git](https://github.com/yourusername/hybrid-edge-ai-os.git)
cd hybrid-edge-ai-os

# Setup dependencies
pip install -r requirements.txt
