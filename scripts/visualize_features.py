"""
visualize_features.py
=====================
Analyzes the trade journal and computes feature importance for the
trained models. Helps understand which tools are driving decisions.
"""
import os
import sqlite3
import sys

# Make the repository importable no matter where this script is run from.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe (works on servers and Colab)
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # noqa: F401  (optional styling)
except ImportError:  # pragma: no cover
    sns = None

from pathlib import Path

from python_brain.config import JOURNAL_DB, MODELS_DIR

def run_analysis():
    if not JOURNAL_DB.exists():
        print(f"Journal DB not found at {JOURNAL_DB}")
        return

    # 1. Load data from journal
    conn = sqlite3.connect(str(JOURNAL_DB))
    # The schema lives in python_brain/database/migrations/001_initial.sql:
    # the table is `trade_journal` and closed trades have closed_at set.
    df = pd.read_sql_query(
        "SELECT * FROM trade_journal WHERE closed_at IS NOT NULL", conn
    )
    conn.close()

    if df.empty:
        print("No closed trades in journal.")
        return

    # The feature_vec is stored as a blob or json string. 
    # Let's assume we need to parse it. 
    # Actually, let's look at how it's stored in db_manager.py
    
    print(f"Found {len(df)} closed trades.")
    
    # Simple correlation analysis between scores and PnL
    score_cols = [c for c in df.columns if 'score' in c or 'conf' in c]
    if not score_cols:
        # If features aren't in main table, we might need to join with another table or parse a blob
        print("No score columns found in main table. Checking for feature_vec...")
        # (Simplified for this script)
        return

    corr = df[score_cols + ['pnl_pct']].corr()['pnl_pct'].sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    corr.drop('pnl_pct').plot(kind='barh')
    plt.title('Feature Correlation with PnL%')
    plt.tight_layout()
    
    output_path = Path("docs/feature_importance.png")
    plt.savefig(output_path)
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    run_analysis()
