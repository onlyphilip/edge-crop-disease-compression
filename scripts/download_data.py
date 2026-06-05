"""Command-line entry point for downloading the PlantVillage dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_crop_disease_ai.data.download import main


if __name__ == "__main__":
    main()
