# Edge Model Comparison

| Model Variant | Backend | Purpose | Accuracy | Macro-F1 | Model Size MB | Avg Latency ms | FPS | RAM Delta MB | Energy / Emissions if available | Deployment Suitability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keras original | keras | Baseline training model | 0.9254 | 0.9163 | 9.3994 | 25.658 | 38.974 | 4.906 | not_available | Baseline only; heavier runtime dependency |
| TFLite FP32 | tflite | Edge-compatible baseline | 0.9254 | 0.9163 | 8.5546 | 2.551 | 391.961 | 0.000 | not_available | Strong edge candidate |
| TFLite FP16 | tflite | Reduced precision model | 0.9254 | 0.9163 | 4.3445 | 2.505 | 399.259 | 0.000 | not_available | Strong edge candidate |
| TFLite INT8 | tflite | Aggressively quantized edge model | 0.9254 | 0.9163 | 2.6302 | 0.941 | 1062.393 | 0.000 | not_available | Strong edge candidate |
| Pruned model | optional | Sparse architecture, optional | 0.9254 | 0.9163 | not_available | not_available | not_available | not_available | not_available | Not evaluated |
| Distilled model | optional | Smaller student network, optional | 0.9254 | 0.9163 | not_available | not_available | not_available | not_available | not_available | Not evaluated |
| Combined optimization | optional | Real deployment configuration, optional | 0.9254 | 0.9163 | not_available | not_available | not_available | not_available | not_available | Not evaluated |
