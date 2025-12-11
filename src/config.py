from pathlib import Path

# Root folder of the project (this file is in src/, so go one level up)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEBUG_SMALL_DATA = True     # podczas debugowania
DEBUG_NROWS = 1000000           # liczba rekordów do testu

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
