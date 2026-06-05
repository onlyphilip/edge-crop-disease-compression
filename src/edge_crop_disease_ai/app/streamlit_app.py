"""Simple Streamlit app placeholder for plant disease inference."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _project_root() -> Path:
    """Return the repository root from the Streamlit app module location."""
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    """Return the default config.yaml path."""
    return _project_root() / "config.yaml"


def _resolve_available_tflite_models(config: dict) -> list[Path]:
    """Collect exported TFLite models that exist on disk."""
    export_dir = Path(config["export"]["export_dir"]).expanduser()
    names = [
        config["export"]["tflite_name_fp16"],
        config["export"]["tflite_name_fp32"],
        config["export"]["tflite_name_int8"],
    ]
    return [export_dir / name for name in names if (export_dir / name).exists()]


def _resolve_model_choices(config: dict) -> dict[str, dict]:
    """Return selectable model variants for the demo."""
    export_dir = Path(config["export"]["export_dir"]).expanduser()
    keras_path = Path(config["export"]["keras_model_path"]).expanduser()
    if not keras_path.exists():
        keras_path = Path(config["paths"]["checkpoints_dir"]).expanduser() / config["train"]["checkpoint_name"]
    return {
        "Keras original": {
            "backend": "keras",
            "format": "Keras Saved Model",
            "path": keras_path,
            "description": "Original training model",
        },
        "TFLite FP32": {
            "backend": "tflite",
            "format": "TFLite FP32",
            "path": export_dir / config["export"]["tflite_name_fp32"],
            "description": "Edge deployment baseline",
        },
        "TFLite FP16": {
            "backend": "tflite",
            "format": "TFLite FP16",
            "path": export_dir / config["export"]["tflite_name_fp16"],
            "description": "Reduced precision edge model",
        },
        "TFLite INT8": {
            "backend": "tflite",
            "format": "TFLite INT8",
            "path": export_dir / config["export"]["tflite_name_int8"],
            "description": "Aggressively quantized edge model",
        },
    }


def _load_edge_metrics(config: dict) -> dict[str, dict]:
    """Load benchmark summary rows keyed by model name."""
    metrics_path = Path(config["paths"]["results_dir"]).expanduser() / "edge_benchmark" / "edge_metrics_summary.csv"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["model_name"]: row for row in csv.DictReader(handle)}


def _file_size_mb(path: Path) -> float | None:
    """Return file size in MiB when available."""
    if not path.exists():
        return None
    return path.stat().st_size / (1024.0 * 1024.0)


def _format_metric(value: object, suffix: str = "", digits: int = 3) -> str:
    """Format Streamlit metric values with not_available fallback."""
    if value in (None, "", "not_available"):
        return "not available"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _edge_suitability(selected_name: str, metrics: dict) -> str:
    """Return a short deployment suitability label."""
    if not metrics or metrics.get("status") != "ok":
        return "Benchmark not available"
    if selected_name == "Keras original":
        return "Baseline only; TFLite is preferred for edge deployment"
    latency = metrics.get("avg_latency_ms")
    size = metrics.get("model_size_mb")
    try:
        if float(latency) <= 50 and float(size) <= 10:
            return "Strong edge candidate"
        if float(latency) <= 100:
            return "Suitable for CPU edge inference"
    except (TypeError, ValueError):
        pass
    return "Requires device-specific validation"


def _render_predictions(predictions: list[dict]) -> None:
    """Render top-k predictions in Streamlit."""
    import pandas as pd
    import streamlit as st

    rows = [
        {
            "rank": index + 1,
            "class_name": prediction["class_name"],
            "confidence": round(prediction["confidence"], 4),
        }
        for index, prediction in enumerate(predictions)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    """Launch the Streamlit demo workflow.

    """
    import streamlit as st

    from edge_crop_disease_ai.config import load_config
    from edge_crop_disease_ai.inference.keras_infer import predict_with_keras
    from edge_crop_disease_ai.inference.tflite_infer import predict_with_tflite

    st.set_page_config(page_title="Edge Crop Disease AI", layout="wide")

    config_path = _default_config_path()
    config = load_config(config_path)

    st.title(config["demo"]["title"])
    st.caption(config["demo"]["description"])

    with st.sidebar:
        st.header("Settings")
        model_choices = _resolve_model_choices(config)
        selected_model_name = st.radio(
            "Model variant",
            options=list(model_choices.keys()),
            index=2 if model_choices["TFLite FP16"]["path"].exists() else 0,
        )
        selected_model = model_choices[selected_model_name]
        top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=int(config["inference"].get("top_k", 3)))
        config["inference"]["top_k"] = int(top_k)

        if selected_model["backend"] == "keras":
            st.caption("Keras is the original training model backend.")
        else:
            st.caption("TFLite is the edge deployment backend.")

    edge_metrics = _load_edge_metrics(config)
    selected_metrics = edge_metrics.get(selected_model_name, {})

    st.subheader("Edge deployment metrics")
    metric_cols = st.columns(4)
    size_value = selected_metrics.get("model_size_mb") or _file_size_mb(selected_model["path"])
    metric_cols[0].metric("Backend", selected_model["backend"])
    metric_cols[1].metric("Format", selected_model["format"])
    metric_cols[2].metric("Model size", _format_metric(size_value, " MB"))
    metric_cols[3].metric("Edge suitability", _edge_suitability(selected_model_name, selected_metrics))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Avg latency", _format_metric(selected_metrics.get("avg_latency_ms"), " ms"))
    metric_cols[1].metric("FPS", _format_metric(selected_metrics.get("fps")))
    metric_cols[2].metric("RAM delta", _format_metric(selected_metrics.get("ram_delta_mb"), " MB"))
    metric_cols[3].metric("Status", selected_metrics.get("status", "not benchmarked"))

    if not selected_metrics:
        st.info("Run `python scripts/benchmark.py --config config.yaml` to populate edge metrics.")

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)

    run_inference = st.button("Run inference", type="primary")
    if not run_inference:
        return

    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_image_path = Path(temp_file.name)

    try:
        with st.spinner("Running inference..."):
            if selected_model["backend"] == "keras":
                result = predict_with_keras(config, str(temp_image_path))
            else:
                if not selected_model["path"].exists():
                    st.error(f"Selected TFLite model is missing: {selected_model['path']}")
                    return
                result = predict_with_tflite(config, str(temp_image_path), str(selected_model["path"]))

        st.success(f"Predicted class: {result['predicted_class']}")
        st.write(f"Backend: `{result['backend']}`")
        st.write(f"Selected variant: `{selected_model_name}`")
        st.write(f"Model: `{Path(result['model_path']).name}`")
        _render_predictions(result["predictions"])
    except Exception as exc:
        st.exception(exc)
    finally:
        if temp_image_path.exists():
            temp_image_path.unlink()


if __name__ == "__main__":
    main()
