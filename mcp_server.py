"""
Veilcrean MCP Server
====================
Model Context Protocol (MCP) server exposing Veilcrean's 153 analysis
tools and NeuroSense cognitive engine to external AI agents.

Tools exposed:
  - analyze_market:       Run all analysis tools on a symbol
  - get_tool_list:        List all available analysis tools
  - cognitive_reason:     NeuroSense semantic reasoning about market conditions
  - pattern_recall:       Recall similar past market setups from episodic memory
  - adaptive_threshold:   Get Q-learning recommended confidence threshold
  - brain_introspect:     Full self-report of the cognitive brain state
  - brain_think:          Run one cognitive cycle
  - brain_sleep:          Consolidate memories and derive new knowledge
  - learn_fact:           Teach the brain a new market fact
  - ask_question:         Ask the brain a yes/no question with explanation
  - free_associate:       Get creative associations around a concept
  - train_classifier:     Train a neural pattern classifier
  - classify_pattern:     Classify a pattern using a trained classifier
  - record_trade:         Record a completed trade for learning feedback
"""
import sys
import json
import asyncio
import os
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# MCP Server framework (minimal stdio-based implementation)
# ---------------------------------------------------------------------------
class MCPServer:
    """Minimal MCP server that registers tools and serves them over stdio."""

    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, Any] = {}

    def register_tool(self, name: str, description: str,
                      input_schema: dict, func):
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": func,
        }

    def list_tools(self) -> List[dict]:
        return [
            {"name": t["name"], "description": t["description"],
             "inputSchema": t["inputSchema"]}
            for t in self.tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = await tool["handler"](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    async def run(self):
        """Stdio loop: read JSON-RPC requests, write JSON-RPC responses."""
        print(f"Veilcrean MCP Server '{self.name}' starting...", file=sys.stderr)
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_stdio(protocol)

        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode())
            except json.JSONDecodeError:
                continue

            method = request.get("method", "")
            req_id = request.get("id")

            if method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id,
                            "result": {"tools": self.list_tools()}}
            elif method == "tools/call":
                params = request.get("params", {})
                result = await self.call_tool(
                    params.get("name", ""), params.get("arguments", {}))
                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            elif method == "initialize":
                response = {"jsonrpc": "2.0", "id": req_id,
                            "result": {"protocolVersion": "2024-11-05",
                                       "serverInfo": {"name": self.name,
                                                      "version": "3.1.0"}}}
            else:
                response = {"jsonrpc": "2.0", "id": req_id,
                            "error": {"code": -32601,
                                      "message": f"Unknown method: {method}"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_brain = None
_cognitive_tool = None
_pattern_memory_tool = None
_adaptive_threshold_tool = None


def _get_brain():
    global _brain
    if _brain is None:
        from neurosense import Brain
        _brain = Brain(name="veilcrean-mcp")
        _seed_brain(_brain)
    return _brain


def _seed_brain(brain):
    brain.read(
        "A trend is a regime. A range is a regime. "
        "A breakout is a regime. A trend has momentum. "
        "A range has low volatility. A breakout has high volatility. "
        "Momentum can drive price. Volatility can expand range. "
        "A trend can break. A range can break. "
        "Consolidation can precede breakout. "
        "Liquidity can fuel breakout. "
        "Support can hold price. Resistance can reject price. "
        "A breakout can follow consolidation. "
        "High volatility can create opportunity. "
        "Low volatility can indicate consolidation."
    )


def _get_cognitive_tool():
    global _cognitive_tool
    if _cognitive_tool is None:
        from python_brain.analysis_tools.cognitive_reasoner import CognitiveReasonerTool
        _cognitive_tool = CognitiveReasonerTool()
    return _cognitive_tool


def _get_pattern_memory_tool():
    global _pattern_memory_tool
    if _pattern_memory_tool is None:
        from python_brain.analysis_tools.pattern_memory import PatternMemoryTool
        _pattern_memory_tool = PatternMemoryTool()
    return _pattern_memory_tool


def _get_adaptive_threshold_tool():
    global _adaptive_threshold_tool
    if _adaptive_threshold_tool is None:
        from python_brain.analysis_tools.adaptive_threshold import AdaptiveThresholdTool
        _adaptive_threshold_tool = AdaptiveThresholdTool()
    return _adaptive_threshold_tool


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def handle_analyze_market(symbol: str = "EURUSD",
                                 tool_scores: Optional[dict] = None,
                                 price: float = 0.0,
                                 regime: str = "UNKNOWN"):
    """Run all analysis tools on a symbol and return combined results."""
    scores = tool_scores or {
        "market_structure": 0.3,
        "momentum_volume": 0.5,
        "volatility_bands": 0.2,
        "liquidity": 0.4,
    }

    results = {}

    # Cognitive reasoner
    cog = _get_cognitive_tool()
    cog_result = cog.analyze({}, prev_scores=scores, price=price,
                             symbol=symbol)
    results["cognitive_reasoner"] = {
        "score": cog_result.score,
        "confidence": cog_result.confidence,
        "reasoning": cog_result.metadata.get("reasoning", ""),
    }

    # Pattern memory
    pm = _get_pattern_memory_tool()
    pm_result = pm.analyze({}, prev_scores=scores, regime=regime)
    results["pattern_memory"] = {
        "score": pm_result.score,
        "confidence": pm_result.confidence,
        "recalled_count": pm_result.features.get("recalled_count", 0),
        "memory_size": pm_result.features.get("memory_size", 0),
    }

    # Adaptive threshold
    at = _get_adaptive_threshold_tool()
    at_result = at.analyze({}, regime=regime, confidence=0.65)
    results["adaptive_threshold"] = {
        "score": at_result.score,
        "recommended_threshold": at_result.features.get(
            "recommended_threshold", 0.65),
        "regime": at_result.features.get("regime", regime),
        "agent_updates": at_result.features.get("agent_updates", 0),
    }

    # Combined sentiment
    all_scores = [r["score"] for r in results.values()]
    combined = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "symbol": symbol,
        "price": price,
        "regime": regime,
        "combined_sentiment": combined,
        "tool_results": results,
        "input_scores": scores,
    }


async def handle_get_tool_list():
    """List all 153 available analysis tools."""
    try:
        from python_brain.analysis_tools import ALL_TOOLS, __all__
        return {
            "total_tools": len(ALL_TOOLS),
            "tools": __all__,
        }
    except Exception as e:
        return {"error": str(e), "total_tools": 0, "tools": []}


async def handle_cognitive_reason(market_facts: str = ""):
    """Feed market facts to the cognitive brain and get reasoning output."""
    brain = _get_brain()
    if market_facts:
        brain.read(market_facts)
    can_do = brain.reason("market", "can")
    has = brain.reason("market", "has")
    return {
        "market_can": can_do,
        "market_has": has,
        "knowledge_facts": len(brain.knowledge),
        "introspect": brain.introspect(),
    }


async def handle_pattern_recall(query: str = "", regime: str = "unknown"):
    """Recall similar past market setups from episodic memory."""
    pm = _get_pattern_memory_tool()
    if not pm.enabled:
        return {"error": "Pattern memory not available (neurosense not installed)"}

    recalled = pm.memory.recall(query or regime, top=5)
    return {
        "query": query or regime,
        "recalled_count": len(recalled),
        "memory_size": len(pm.memory),
        "episodes": [
            {"summary": ep.summary, "importance": ep.importance,
             "kind": ep.kind, "recalls": ep.recalls}
            for ep in recalled
        ],
    }


async def handle_adaptive_threshold(regime: str = "UNKNOWN"):
    """Get the Q-learning recommended confidence threshold for a regime."""
    at = _get_adaptive_threshold_tool()
    recommended = at.get_recommended_threshold(regime)
    return {
        "regime": regime,
        "recommended_threshold": recommended,
        "agent_updates": at.agent.total_updates if at.enabled else 0,
        "exploration_rate": at.agent.epsilon if at.enabled else 0.0,
    }


async def handle_brain_introspect():
    """Get a full self-report of the cognitive brain's current state."""
    brain = _get_brain()
    return {"introspection": brain.introspect()}


async def handle_brain_think():
    """Run one cognitive cycle and return the brain's thought."""
    brain = _get_brain()
    return {"thought": brain.think()}


async def handle_brain_sleep():
    """Consolidate memories and derive new knowledge (like sleep)."""
    brain = _get_brain()
    return {"result": brain.sleep()}


async def handle_learn_fact(subject: str, relation: str, obj: str):
    """Teach the brain a new fact."""
    brain = _get_brain()
    fact = brain.learn_fact(subject, relation, obj)
    return {"learned": str(fact), "total_facts": len(brain.knowledge)}


async def handle_ask_question(subject: str, relation: str, obj: str):
    """Ask the brain a yes/no question with confidence and explanation."""
    brain = _get_brain()
    answer, conf, why = brain.ask(subject, relation, obj)
    return {"answer": answer, "confidence": conf, "explanation": why}


async def handle_free_associate(concept: str, top: int = 8):
    """Get creative associations around a concept."""
    brain = _get_brain()
    associations = brain.free_associate(concept, top=top)
    return {"concept": concept, "associations": associations}


async def handle_train_classifier(name: str, input_size: int,
                                   classes: List[str],
                                   training_data: List[List[float]],
                                   labels: List[str],
                                   epochs: int = 200):
    """Train a neural pattern classifier."""
    import numpy as np
    brain = _get_brain()
    brain.build_classifier(name, input_size=input_size, classes=classes)
    X = np.array(training_data, dtype=float)
    history = brain.train_classifier(name, X, labels, epochs=epochs)
    return {
        "name": name,
        "final_loss": history[-1] if history else 0.0,
        "epochs": len(history),
    }


async def handle_classify_pattern(name: str, features: List[float]):
    """Classify a pattern using a trained classifier."""
    import numpy as np
    brain = _get_brain()
    if name not in brain._classifiers:
        return {"error": f"Classifier '{name}' not found. Train it first."}
    label, prob = brain.classify(name, np.array(features, dtype=float))
    return {"label": label, "probability": prob}


async def handle_record_trade(tool_scores: dict, regime: str,
                               outcome: str, pnl: float):
    """Record a completed trade for pattern memory and adaptive threshold learning."""
    pm = _get_pattern_memory_tool()
    pm.record_trade(tool_scores, regime, outcome, pnl)

    at = _get_adaptive_threshold_tool()
    threshold = at.get_recommended_threshold(regime)
    at.record_outcome(regime, threshold, pnl)

    return {
        "recorded": True,
        "memory_size": len(pm.memory) if pm.enabled else 0,
        "threshold_agent_updates": at.agent.total_updates if at.enabled else 0,
    }


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

def create_server() -> MCPServer:
    server = MCPServer("Veilcrean-Analyst")

    server.register_tool(
        "analyze_market",
        "Run all analysis tools (cognitive reasoner, pattern memory, "
        "adaptive threshold) on a symbol and return combined results.",
        {"type": "object",
         "properties": {
             "symbol": {"type": "string", "default": "EURUSD"},
             "tool_scores": {"type": "object", "description": "Pre-computed tool scores"},
             "price": {"type": "number", "default": 0.0},
             "regime": {"type": "string", "default": "UNKNOWN"},
         }},
        handle_analyze_market,
    )

    server.register_tool(
        "get_tool_list",
        "List all 153 available analysis tools.",
        {"type": "object", "properties": {}},
        handle_get_tool_list,
    )

    server.register_tool(
        "cognitive_reason",
        "Feed market facts to the NeuroSense cognitive brain and get "
        "reasoning output about what the market can do and has.",
        {"type": "object",
         "properties": {
             "market_facts": {"type": "string",
                              "description": "Market facts in plain English, "
                              "e.g. 'market is_a trend. market has momentum.'"},
         }},
        handle_cognitive_reason,
    )

    server.register_tool(
        "pattern_recall",
        "Recall similar past market setups from episodic memory.",
        {"type": "object",
         "properties": {
             "query": {"type": "string", "description": "Keywords to search for"},
             "regime": {"type": "string", "default": "unknown"},
         }},
        handle_pattern_recall,
    )

    server.register_tool(
        "adaptive_threshold",
        "Get the Q-learning recommended confidence threshold for a market regime.",
        {"type": "object",
         "properties": {
             "regime": {"type": "string", "default": "UNKNOWN",
                        "description": "TRENDING, RANGING, VOLATILE, CHOPPY, BREAKOUT"},
         }},
        handle_adaptive_threshold,
    )

    server.register_tool(
        "brain_introspect",
        "Get a full self-report of the cognitive brain's current state.",
        {"type": "object", "properties": {}},
        handle_brain_introspect,
    )

    server.register_tool(
        "brain_think",
        "Run one cognitive cycle and return the brain's thought.",
        {"type": "object", "properties": {}},
        handle_brain_think,
    )

    server.register_tool(
        "brain_sleep",
        "Consolidate memories and derive new knowledge (like sleep).",
        {"type": "object", "properties": {}},
        handle_brain_sleep,
    )

    server.register_tool(
        "learn_fact",
        "Teach the brain a new fact (subject, relation, object).",
        {"type": "object",
         "properties": {
             "subject": {"type": "string"},
             "relation": {"type": "string", "description": "is_a, has, can, etc."},
             "obj": {"type": "string"},
         },
         "required": ["subject", "relation", "obj"]},
        handle_learn_fact,
    )

    server.register_tool(
        "ask_question",
        "Ask the brain a yes/no question with confidence and explanation.",
        {"type": "object",
         "properties": {
             "subject": {"type": "string"},
             "relation": {"type": "string"},
             "obj": {"type": "string"},
         },
         "required": ["subject", "relation", "obj"]},
        handle_ask_question,
    )

    server.register_tool(
        "free_associate",
        "Get creative associations around a concept using spreading "
        "activation and learned word co-occurrence.",
        {"type": "object",
         "properties": {
             "concept": {"type": "string"},
             "top": {"type": "integer", "default": 8},
         },
         "required": ["concept"]},
        handle_free_associate,
    )

    server.register_tool(
        "train_classifier",
        "Train a neural pattern classifier on labeled data.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "input_size": {"type": "integer"},
             "classes": {"type": "array", "items": {"type": "string"}},
             "training_data": {"type": "array",
                               "items": {"type": "array", "items": {"type": "number"}}},
             "labels": {"type": "array", "items": {"type": "string"}},
             "epochs": {"type": "integer", "default": 200},
         },
         "required": ["name", "input_size", "classes", "training_data", "labels"]},
        handle_train_classifier,
    )

    server.register_tool(
        "classify_pattern",
        "Classify a pattern using a trained classifier.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "features": {"type": "array", "items": {"type": "number"}},
         },
         "required": ["name", "features"]},
        handle_classify_pattern,
    )

    server.register_tool(
        "record_trade",
        "Record a completed trade for pattern memory and adaptive "
        "threshold learning. Call after each trade closes.",
        {"type": "object",
         "properties": {
             "tool_scores": {"type": "object"},
             "regime": {"type": "string"},
             "outcome": {"type": "string", "description": "win or loss"},
             "pnl": {"type": "number"},
         },
         "required": ["tool_scores", "regime", "outcome", "pnl"]},
        handle_record_trade,
    )

    return server


async def main():
    server = create_server()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
