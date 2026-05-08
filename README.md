
# Purple Cap Analysis

This repository contains analysis and prediction experiments for the "Purple Cap" (leading wicket-taker) using match and delivery data.

Contents
- `ipl-predictor.ipynb`: Jupyter notebook with exploratory analysis and a simple predictor.
- `assets/`: raw CSV data used for analysis (`matches.csv`, `deliveries.csv`, and 2025 variants).
- `analysis_outputs/`: generated summary outputs (e.g. `wickets_by_over_*.csv`).
- `requirements.txt`: Python dependencies for the notebook and scripts.

Quickstart
1. Create and activate a Python virtual environment (example using venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Open the notebook `ipl-predictor.ipynb` in Jupyter or VS Code and run the cells to reproduce the analysis.

Data
- `assets/matches.csv` and `assets/deliveries.csv` are the primary inputs. There are also `*_2025.csv` variants included for subset analyses.
- Processed outputs are stored in `analysis_outputs/` (CSV summaries and derived tables).

Notes
- This repo is organized for interactive analysis. If you want to reproduce results programmatically, I can add small scripts (e.g., `scripts/run_analysis.py`).
- If you encounter binary extension import errors (e.g., `pydantic` on Windows), prefer running the notebook with the environment in `requirements.txt` or use the Python standard library alternatives.

Next steps
- Review this README and tell me any additional sections to add (examples, visualizations, CI, license).

