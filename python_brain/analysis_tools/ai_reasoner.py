"""
ai_reasoner.py
==============
Tool 10 — AI Reasoning (Gemini / Groq)

Uses LLMs to provide a high-level "sanity check" and sentiment score
based on the output of all other technical tools.
"""
from __future__ import annotations
import os
import time
import json
from typing import Dict, Optional, List
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import LLM_CFG

class AIReasonerTool(BaseTool):
    name = "ai_reasoner"

    def __init__(self):
        super().__init__()
        self.last_run = 0
        self.last_result: Optional[ToolResult] = None
        self._init_clients()

    def _init_clients(self):
        self.enabled = False
        if LLM_CFG.provider == "groq" and LLM_CFG.groq_api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=LLM_CFG.groq_api_key)
                self.enabled = True
            except ImportError:
                pass
        elif LLM_CFG.provider == "gemini" and LLM_CFG.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=LLM_CFG.gemini_api_key)
                self.client = genai.GenerativeModel(LLM_CFG.gemini_model)
                self.enabled = True
            except ImportError:
                pass

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        # If disabled or not enough time passed, return a neutral score or last result
        if not self.enabled or not LLM_CFG.enabled:
            return ToolResult(tool_name=self.name, score=0.0, confidence=0.0)

        now = time.time()
        if self.last_result and (now - self.last_run) < 60: # Max once per minute
             return self.last_result

        # Build prompt from tool scores (passed in ctx)
        tool_scores = ctx.get("prev_scores", {})
        price = ctx.get("price", 0.0)
        symbol = ctx.get("symbol", "Unknown")
        
        prompt = f"""
        Analyze the following technical analysis data for {symbol} at price {price}.
        Technical Tool Scores (Scale -1 to 1):
        {json.dumps(tool_scores, indent=2)}
        
        Provide a concise market sentiment assessment. 
        Return a JSON object with:
        "sentiment_score": float between -1.0 (very bearish) and 1.0 (very bullish)
        "reasoning": string (max 50 words)
        "confidence": float between 0.0 and 1.0
        """

        try:
            if LLM_CFG.provider == "groq":
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=LLM_CFG.groq_model,
                    response_format={"type": "json_object"}
                )
                res_content = chat_completion.choices[0].message.content
            else: # Gemini
                response = self.client.generate_content(prompt)
                res_content = response.text
                # Try to extract JSON if it's wrapped in markdown
                if "```json" in res_content:
                    res_content = res_content.split("```json")[1].split("```")[0]

            data = json.loads(res_content)
            
            result = ToolResult(
                tool_name=self.name,
                score=float(data.get("sentiment_score", 0.0)),
                confidence=float(data.get("confidence", 0.5)),
                features={
                    "ai_sentiment": float(data.get("sentiment_score", 0.0)),
                    "ai_conf": float(data.get("confidence", 0.5))
                },
                metadata={"reasoning": data.get("reasoning", "")}
            )
            self.last_run = now
            self.last_result = result
            return result

        except Exception as e:
            return ToolResult(tool_name=self.name, errors=[f"AI Reasoner failed: {e}"])
