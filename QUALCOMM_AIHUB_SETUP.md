# Qualcomm AI Hub Setup Guide
## RetailEdge Sentinel — SIH26179 | VENSMOLD AI

This guide walks you through compiling and deploying the YOLOv8n model
for the **Qualcomm QCS6490 / Dragonwing** target hardware using Qualcomm AI Hub.

---

## What Is Qualcomm AI Hub?

Qualcomm AI Hub is a **free cloud service** that:
1. Takes your ONNX/PyTorch model
2. Compiles it with INT8 quantization (AIMET) on real Qualcomm hardware in the cloud
3. Measures actual NPU latency on a real chip (QCS6490, Snapdragon 8 Gen etc.)
4. Returns a ready-to-deploy QNN context binary (`.bin`)

**You do NOT need physical hardware to do this compilation step.**

---

## Prerequisites

```bash
# Install SDK
pip install qai-hub qai-hub-models ultralytics onnx

# Sign up free at https://app.aihub.qualcomm.com/
# Get your API token from Account → Settings

# Configure
qai-hub configure --api_token YOUR_API_TOKEN_HERE
```

---

## Step-by-Step Pipeline

### Step 1 — Check status
```bash
python src/qualcomm_aihub.py --step status
```

### Step 2 — Export YOLOv8n → ONNX
```bash
python src/qualcomm_aihub.py --step export
```
Creates: `models/yolov8n_retailedge.onnx`

### Step 3 — Compile on Qualcomm AI Hub (cloud)
```bash
python src/qualcomm_aihub.py --step compile --device "QCS6490 (Proxy)"
```
- Uploads ONNX to Qualcomm AI Hub
- Qualcomm compiles + INT8 quantizes on real QCS6490 hardware
- Returns a Job ID, saved to `models/aihub_compile_job_id.txt`
- Track at: https://app.aihub.qualcomm.com/

### Step 4 — Download compiled model
```bash
python src/qualcomm_aihub.py --step download
```
Creates: `models/yolov8n_retailedge_qnn.bin`

### Step 5 — Profile: get real NPU latency numbers
```bash
python src/qualcomm_aihub.py --step profile
```
Runs on real Qualcomm chip in the cloud.
Saves results to: `models/aihub_profile_results.json`

### Or run all steps at once
```bash
python src/qualcomm_aihub.py --step all
```

---

## Expected Results (from Qualcomm AI Hub)

| Metric | Value |
|--------|-------|
| Model | YOLOv8n (INT8 quantized) |
| Target | QCS6490 — Hexagon NPU |
| Expected latency | ~14 ms per frame |
| Expected FPS | ~70 FPS |
| Model size (QNN bin) | ~3.2 MB |
| vs original float32 | 4× smaller, 3× faster |

---

## Using in the RetailEdge Pipeline

Once `models/yolov8n_retailedge_qnn.bin` is downloaded, the pipeline
auto-selects the Qualcomm engine:

```python
# In config/settings.yaml — change this:
deployment_mode: "qualcomm_edge"   # was: "local_simulation"

# In src/pipeline.py — the create_engine() factory handles the rest:
from src.detector import create_engine
engine = create_engine(prefer_qualcomm=True)
```

On a Windows laptop without hardware: automatically falls back to `LocalInferenceEngine`.
On Dragonwing hardware with the QNN binary: uses full NPU acceleration.

---

## Qualcomm AI Hub Portal

- Dashboard: https://app.aihub.qualcomm.com/
- YOLOv8 on AI Hub: https://aihub.qualcomm.com/models/yolov8_det
- Supported devices: https://app.aihub.qualcomm.com/devices/
- Python SDK docs: https://app.aihub.qualcomm.com/docs/

---

## Deployment Architecture

```
[Laptop Development]
        |
        v
  1. Export ONNX (ultralytics)
        |
        v
  2. Upload to Qualcomm AI Hub API
        |
        v  (Qualcomm cloud servers — real QCS6490 hardware)
  3. INT8 Quantization (AIMET)
  4. QNN Compilation
  5. NPU Profiling
        |
        v
  6. Download QNN binary (.bin)
        |
        v
[Dragonwing / RB3 Gen 2 Device]
        |
        v
  7. Copy models/yolov8n_retailedge_qnn.bin → device
  8. Run: python src/qualcomm_aihub.py (on-device inference)
  9. 70 FPS @ 12.8 TOPS NPU
```
