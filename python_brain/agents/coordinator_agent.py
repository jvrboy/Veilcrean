"""
coordinator_agent.py
====================
The "Super Agent" that coordinates the Analyst, Strategist, and Risk Officer.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from .analyst_agent import AnalystAgent
from .strategist_agent import StrategistAgent
from .risk_officer import RiskOfficerAgent
from .sentiment_agent import SentimentAgent
from .market_regime_agent import MarketRegimeAgent
from .portfolio_agent import PortfolioAgent
from .execution_agent import ExecutionAgent
from .scalper_circuit_breaker import CircuitBreakerAgent
from .research_agent import ResearchAgent
from .monitoring_agent import MonitoringAgent
from .simulation_agent import SimulationAgent
from .optimization_agent import OptimizationAgent
from .broker_agent import BrokerAgent
from .data_integration_agent import DataIntegrationAgent
from .synthetic_specialist_agent import SyntheticSpecialistAgent
from .skill_agent import SkillAgent
import numpy as np

class CoordinatorAgent(BaseAgent):
    skill_role = "coordinator"
    name = "coordinator"

    def __init__(self, decision_engine):
        super().__init__()
        self.analyst   = AnalystAgent()
        self.strategist = StrategistAgent(decision_engine)
        self.risk_officer = RiskOfficerAgent()
        self.sentiment = SentimentAgent()
        self.regime    = MarketRegimeAgent()
        self.portfolio = PortfolioAgent()
        self.execution = ExecutionAgent()
        self.breaker   = CircuitBreakerAgent()
        self.research  = ResearchAgent()
        self.monitor   = MonitoringAgent()
        self.sim       = SimulationAgent()
        self.opt       = OptimizationAgent()
        self.broker    = BrokerAgent()
        self.data_int  = DataIntegrationAgent()
        self.synthetic = SyntheticSpecialistAgent()
        self.skills    = SkillAgent()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 0. Data Integration
        data_info = self.data_int.run(context)
        context.update(data_info)

        # 1. Analyst Agent - Get technicals
        analysis = self.analyst.run(context)
        context.update(analysis)

        # 1b. Skill Agent - Run project trader skills for all sub-agent roles
        skills = self.skills.run(context)
        context.update(skills)
        
        # 2. Synthetic Specialist - Focus on DSI
        synthetic = self.synthetic.run(context)
        context.update(synthetic)
        
        # 3. Research Agent - Historical context
        research = self.research.run(context)
        context.update(research)
        
        # 3. Simulation Agent - What if?
        simulation = self.sim.run(context)
        context.update(simulation)
        
        # 4. Optimization Agent - Dynamic tuning
        optimization = self.opt.run(context)
        context.update(optimization)
        
        # 5. Portfolio Agent - Check account health
        portfolio = self.portfolio.run(context)
        context.update(portfolio)
        
        # 4. Sentiment Agent - Get news/external data
        sentiment = self.sentiment.run(context)
        context.update(sentiment)
        
        # 5. Strategist Agent - Get ML decision
        decision = self.strategist.run(context)
        
        # Apply Synthetic Bias for DSI
        if synthetic.get("synthetic_bias", 0) != 0:
            decision["confidence"] = min(0.99, decision["confidence"] + 0.1)
            
        # Apply research bias
        if research.get("research_bias", 0) != 0:
            if np.sign(research["research_bias"]) == np.sign(decision.get("confidence", 0)):
                decision["confidence"] = min(0.99, decision["confidence"] + 0.1)
        
        # Apply optimization results
        if optimization.get("optimal_threshold"):
            context["dynamic_threshold"] = optimization["optimal_threshold"]
            
        context["decision"] = decision
        
        # 6. Regime Agent - Refine based on market type
        regime_data = self.regime.run(context)
        context.update(regime_data)
        
        # 7. Risk Officer - Validate safety
        risk_check = self.risk_officer.run(context)
        context.update(risk_check)
        
        # 8. Circuit Breaker - Check for death spirals
        breaker_check = self.breaker.run(context)
        context.update(breaker_check)

        # 9. Broker Agent - Platform selection
        broker_info = self.broker.run(context)
        context.update(broker_info)

        # 10. Execution Agent - Final formatting
        exec_check = self.execution.run(context)
        context.update(exec_check)
        
        # 10. Monitoring Agent - Dashboard update
        monitor_check = self.monitor.run(context)
        context.update(monitor_check)
        
        return {
            "decision": decision,
            "risk_ok": risk_check.get("risk_ok", False) and not sentiment.get("news_blocked", False) and not breaker_check.get("breaker_tripped", False),
            "risk_reason": risk_check.get("reason", "") if risk_check.get("risk_ok") else "Breaker/Blocked",
            "technical_report": analysis.get("technical_report"),
            "sentiment": sentiment,
            "regime": regime_data,
            "portfolio": portfolio,
            "execution": exec_check,
            "breaker": breaker_check,
            "monitoring": monitor_check,
            "simulation": simulation,
            "optimization": optimization,
            "broker": broker_info,
            "synthetic": synthetic,
            "skills": skills.get("skill_report", {})
        }
