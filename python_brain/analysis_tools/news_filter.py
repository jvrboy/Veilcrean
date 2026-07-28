"""
news_filter.py
==============
Tool 9 — Economic Calendar / News Filter

Checks for upcoming high-impact news events.
Avoids trading 30 minutes before and after Red Folder events.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .base_tool import BaseTool, ToolResult
from ..config import RISK_CFG

class NewsFilterTool(BaseTool):
    name = "news_filter"

    def __init__(self):
        super().__init__()
        self.events = [] # Cache of events: [{"time": ts, "impact": "HIGH", "currency": "USD"}]

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        now = ctx.get("now") or datetime.now(timezone.utc)
        
        # In a real implementation, we'd fetch this from an API once an hour
        # For now, we return a neutral score unless an event is near.
        
        near_event = False
        impacted_currency = None
        
        for event in self.events:
            event_time = datetime.fromtimestamp(event['time'], tz=timezone.utc)
            diff = abs((event_time - now).total_seconds()) / 60.0
            
            if diff <= RISK_CFG.news_buffer_minutes and event['impact'] == 'HIGH':
                near_event = True
                impacted_currency = event['currency']
                break
        
        if near_event:
            result.score = 0.0
            result.confidence = 1.0 # High confidence that we shouldn't trade
            result.metadata = {"news_alert": f"High impact news for {impacted_currency} soon"}
            # The strategy logic should see this and potentially block trades
        else:
            result.score = 0.0
            result.confidence = 0.5
            
        result.features = {"near_high_impact_news": float(near_event)}
        
        return result

    def update_events(self, events: List[dict]):
        self.events = events
