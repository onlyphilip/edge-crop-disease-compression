"""Command-line entry point for model evaluation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_crop_disease_ai.evaluation.evaluator import main


if __name__ == "__main__":
    main()
