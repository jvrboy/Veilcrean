"""
ai_critique.py
==============
Uses Gemini or Groq to provide a detailed technical critique of a
specific trade from the journal or a current market setup.
"""
import os
import sys

# Make the repository importable no matter where this script is run from.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import argparse
from pathlib import Path
from python_brain.config import JOURNAL_DB, LLM_CFG
from python_brain.analysis_tools.ai_reasoner import AIReasonerTool
from python_brain.database.db_manager import DBManager
import pandas as pd

def critique_trade(trade_id: str):
    db = DBManager(JOURNAL_DB)
    row = db.fetchone("SELECT * FROM trade_journal WHERE trade_id = ?", (trade_id,))
    if not row:
        print(f"Trade {trade_id} not found.")
        return

    # Convert row to dict
    trade = dict(row)
    
    # We'll use the AI tool's logic but with a more detailed "Critique" prompt
    reasoner = AIReasonerTool()
    if not reasoner.enabled:
        print("LLM not configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env")
        return

    print(f"Critiquing trade {trade_id} ({trade['symbol']} {trade['direction']})...")
    
    prompt = f"""
    You are an institutional trading mentor. Critique this trade setup:
    Symbol: {trade['symbol']}
    Direction: {trade['direction']}
    Entry: {trade['entry_price']}
    SL: {trade['sl']}
    TP: {trade['tp']}
    Confidence: {trade['confidence']}
    Regime: {trade['regime']}
    
    Based on this data, what were the strengths and weaknesses of this trade? 
    Give a rating from 1-10.
    """
    
    # Simple manual call to client
    try:
        if LLM_CFG.provider == "groq":
            res = reasoner.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_CFG.groq_model
            ).choices[0].message.content
        else:
            res = reasoner.client.generate_content(prompt).text
            
        print("\n=== AI CRITIQUE ===")
        print(res)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trade_id", help="The ID of the trade to critique")
    args = parser.parse_args()
    critique_trade(args.trade_id)
