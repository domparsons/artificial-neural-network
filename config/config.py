from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FilePaths:
    data_file: str = str(PROJECT_ROOT / "data" / "FEHDataStudent.xlsx")
    compare_file: str = str(PROJECT_ROOT / "data" / "compare-data.xlsx")
    sheet_name: str = "Sheet1"
    columns: list = (
        "AREA",
        "BFIHOST",
        "FARL",
        "FPEXT",
        "LDP",
        "PROPWET",
        "RMED-1D",
        "SAAR",
        "Index flood",
    )


@dataclass
class Hyperparameters:
    epochs: int = 300
    initial_learning_rate: float = 0.1
    final_learning_rate: float = 0.01
    hidden_layer_size: int = 10
    output_size: int = 1
    momentum_rate: float = 0.8
