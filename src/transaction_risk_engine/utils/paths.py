from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    transaction_file: Path
    identity_file: Path
    joined_file: Path
    reports_dir: Path
