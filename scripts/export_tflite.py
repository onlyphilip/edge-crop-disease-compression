"""Command-line entry point for exporting Keras models to TFLite."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_crop_disease_ai.export.tflite_exporter import main


if __name__ == "__main__":
    main()
