"""
sentiment_agent.py
==================
Sub-agent focusing on high-impact news and overall market sentiment.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent

class SentimentAgent(BaseAgent):
    skill_role = "sentiment"
    name = "sentiment_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tech_report = context.get("technical_report", {})
        results = tech_report.get("tool_results", {})
        
        news = results.get("NewsFilterTool")
        ai   = results.get("AIReasonerTool")
        dxy  = results.get("DXYCorrelationTool")
        
        sentiment_score = 0.0
        if ai:   sentiment_score += ai.score * 0.5
        if dxy:  sentiment_score += dxy.score * 0.3
        
        # News is a multiplier or binary blocker
        blocked = False
        if news and news.metadata.get("news_alert"):
            blocked = True
            
        return {
            "sentiment_score": sentiment_score,
            "news_blocked": blocked,
            "sentiment_summary": "Bullish" if sentiment_score > 0.2 else "Bearish" if sentiment_score < -0.2 else "Neutral"
        }
