# edge-crop-disease-ai

TensorFlow/Keras plant disease classification project for training, evaluation, TFLite export, and edge inference on PlantVillage-style datasets.

## Current Status

This repository is implemented and runnable. The following workflow has been validated locally:

- Train a MobileNetV2-based classifier
- Evaluate on a held-out test split
- Export FP32 / FP16 / INT8 TFLite models
- Run single-image inference with Keras or TFLite
- Benchmark TFLite inference latency
- Launch a Streamlit demo

## Project Layout

```text
config.yaml
checkpoints/
data/
results/
scripts/
src/edge_crop_disease_ai/
```

## Environment

Recommended Python environment:

- Python `3.11`
- Conda env used during validation: `edgecrop311`

Important notes:

- Do not run this project from the Conda `base` environment if it uses Python `3.13`; TensorFlow may crash during import.
- The training/export/evaluation scripts work without `pip install -e .` because the entry scripts already add `src/` to `sys.path`.

## Install Dependencies

If you are creating a fresh environment:

```bash
conda create -n edgecrop311 python=3.11 -y
conda activate edgecrop311
pip install -r requirements.txt
```

If you already have the validated environment:

```bash
conda activate edgecrop311
```

Optional but recommended on macOS to avoid Matplotlib cache warnings:

```bash
mkdir -p .mplconfig
export MPLCONFIGDIR=$PWD/.mplconfig
```

## Dataset

Default dataset path is configured in [config.yaml](/Users/qijiacheng/Desktop/edge-crop-disease-compression/config.yaml:1):

- `data/plantvillage`

The loader expects class folders directly under that directory, for example:

```text
data/plantvillage/
  Potato___healthy/
  Potato___Late_blight/
  Tomato_healthy/
  ...
```

This repository already contains dataset content under `data/plantvillage`.

## Configuration

Main runtime settings live in [config.yaml](/Users/qijiacheng/Desktop/edge-crop-disease-compression/config.yaml:1), including:

- dataset path and output directories
- image size and batch size
- train/val/test split ratios
- MobileNetV2 model settings
- checkpoint and export filenames
- benchmark and inference options

## End-to-End Workflow

Run from the repository root:

```bash
cd /Users/qijiacheng/Desktop/edge-crop-disease-compression
conda activate edgecrop311
mkdir -p .mplconfig
export MPLCONFIGDIR=$PWD/.mplconfig
```

### 1. Train

```bash
python scripts/train.py --config config.yaml
```

Expected outputs:

- `checkpoints/mobilenetv2_best.keras`
- `results/training_history.json`
- `results/labels.txt`
- `results/splits/dataset_splits.json`

Note:

- If `model.weights: imagenet`, the first training run downloads MobileNetV2 pretrained weights.
- If you need fully offline training, set `model.weights: null` in `config.yaml`.

### 2. Evaluate

```bash
python scripts/evaluate.py --config config.yaml
```

Expected outputs:

- `results/metrics/evaluation_summary.json`
- `results/metrics/classification_report.json`
- `results/metrics/confusion_matrix.png`

### 3. Export TFLite

```bash
python scripts/export_tflite.py --config config.yaml
```

Expected outputs:

- `results/export/plant_disease_fp32.tflite`
- `results/export/plant_disease_fp16.tflite`
- `results/export/plant_disease_int8.tflite`
- `results/export/export_summary.json`

The TensorFlow converter prints verbose logs during export. The success signal is the final summary:

```text
TFLite export completed.
fp32: results/export/plant_disease_fp32.tflite
fp16: results/export/plant_disease_fp16.tflite
int8: results/export/plant_disease_int8.tflite
```

### 4. Single-Image Inference

Keras inference:

```bash
python scripts/infer_keras.py \
  --config config.yaml \
  --image "data/plantvillage/Potato___healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG"
```

TFLite inference:

```bash
python scripts/infer_tflite.py \
  --config config.yaml \
  --image "data/plantvillage/Potato___healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG" \
  --model results/export/plant_disease_fp16.tflite
```

Example validated output:

```text
TFLite inference completed.
Predicted class: Potato___healthy
- Potato___healthy: 0.9992
- Potato___Late_blight: 0.0007
- Pepper__bell___healthy: 0.0001
```

Note:

- TensorFlow may print a deprecation warning for `tf.lite.Interpreter`. This does not block current execution.

### 5. Benchmark

```bash
python scripts/benchmark.py --config config.yaml
```

Expected outputs:

- `results/edge_benchmark/edge_metrics_raw.json`
- `results/edge_benchmark/edge_metrics_summary.csv`

### 6. EdgeAI Evaluation

This project evaluates edge deployment tradeoffs across:

- Model Quality: accuracy, macro-F1, classification report, confusion matrix
- Inference Efficiency: latency, p50/p95 latency, throughput, FPS
- Memory Usage: RAM before/after inference, RAM delta, optional peak memory, model size
- Energy Consumption: estimated energy and emissions when CodeCarbon is supported
- Communication Cost: input/output tensor bytes and model file size
- System Stability: CPU utilization, runtime status, failed runs, optional profiling hooks

Notes:

- `psutil`, `memory-profiler`, and `codecarbon` are installed through `requirements.txt`.
- `pyRAPL` and `pynvml` are optional hardware-specific integrations. They are not required and unsupported systems report `not_available`.

Run the full edge comparison workflow after training, evaluation, and TFLite export:

```bash
python scripts/benchmark.py --config config.yaml
python scripts/compare_edge_models.py --config config.yaml
python scripts/plot_edge_tradeoffs.py --config config.yaml
```

Expected outputs:

- `results/edge_benchmark/model_comparison.csv`
- `results/edge_benchmark/model_comparison.md`
- `results/edge_benchmark/figures/accuracy_vs_latency.png`
- `results/edge_benchmark/figures/model_size_vs_accuracy.png`
- `results/edge_benchmark/figures/latency_vs_model_size.png`
- `results/edge_benchmark/figures/fps_comparison.png`
- `results/edge_benchmark/figures/memory_comparison.png`

Expected comparison table format:

| Model Variant | Backend | Accuracy | Macro-F1 | Size MB | Latency ms | FPS | RAM Delta MB | Deployment Suitability |
|---|---|---:|---:|---:|---:|---:|---:|---|

### 7. Streamlit Demo

```bash
streamlit run src/edge_crop_disease_ai/app/streamlit_app.py
```

The app lets you:

- upload a leaf image
- choose Keras original, TFLite FP32, TFLite FP16, or TFLite INT8
- see edge metrics for the selected model when benchmark results exist
- inspect top-k predictions

## Main Scripts

- `scripts/train.py`: train the classifier
- `scripts/evaluate.py`: evaluate a trained checkpoint
- `scripts/export_tflite.py`: export TFLite variants
- `scripts/infer_keras.py`: single-image inference with Keras
- `scripts/infer_tflite.py`: single-image inference with TFLite
- `scripts/benchmark.py`: edge deployment benchmarking
- `scripts/compare_edge_models.py`: combined quality and edge metrics report
- `scripts/plot_edge_tradeoffs.py`: edge tradeoff visualizations
- `src/edge_crop_disease_ai/app/streamlit_app.py`: Streamlit UI

## Troubleshooting

### TensorFlow crashes on import

If `import tensorflow` segfaults:

- check that you are not using Python `3.13`
- switch to the validated environment:

```bash
conda activate edgecrop311
```

### Training tries to download MobileNetV2 weights

If you are offline and training fails while downloading pretrained weights:

- open [config.yaml](/Users/qijiacheng/Desktop/edge-crop-disease-compression/config.yaml:1)
- change:

```yaml
model:
  weights: imagenet
```

- to:

```yaml
model:
  weights: null
```

### Matplotlib cache warnings on macOS

If you see `.matplotlib is not a writable directory`:

```bash
mkdir -p .mplconfig
export MPLCONFIGDIR=$PWD/.mplconfig
```

## Verified Artifacts

The following outputs have been produced and verified in this repository:

- `checkpoints/mobilenetv2_best.keras`
- `results/export/plant_disease_fp32.tflite`
- `results/export/plant_disease_fp16.tflite`
- `results/export/plant_disease_int8.tflite`
