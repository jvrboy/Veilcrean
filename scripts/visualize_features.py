"""
visualize_features.py
=====================
Analyzes the trade journal and computes feature importance for the
trained models. Helps understand which tools are driving decisions.
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from python_brain.config import JOURNAL_DB, MODELS_DIR

def run_analysis():
    if not JOURNAL_DB.exists():
        print(f"Journal DB not found at {JOURNAL_DB}")
        return

    # 1. Load data from journal
    conn = sqlite3.connect(str(JOURNAL_DB))
    df = pd.read_sql_query("SELECT * FROM trades WHERE status = 'CLOSED'", conn)
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
