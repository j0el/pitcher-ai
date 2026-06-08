from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STATCAST_RAW_DIR = RAW_DIR / "statcast"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_STATCAST_GLOB = str(STATCAST_RAW_DIR / "*.parquet")
FEATURES_PATH = PROCESSED_DIR / "pitch_context_features.parquet"
MODEL_PATH = MODELS_DIR / "context_model.joblib"
METRICS_PATH = REPORTS_DIR / "context_model_metrics.json"
REPORT_PATH = REPORTS_DIR / "context_model_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "context_model_confusion_matrix.csv"
CLASS_CONFUSION_MATRIX_PATH = REPORTS_DIR / "context_model_pitch_class_confusion_matrix.csv"
CLASS_REPORT_PATH = REPORTS_DIR / "context_model_pitch_class_report.txt"
PITCH_TYPE_KEY_PATH = REPORTS_DIR / "pitch_type_key.csv"
SUMMARY_PATH = REPORTS_DIR / "baseline_summary.md"
