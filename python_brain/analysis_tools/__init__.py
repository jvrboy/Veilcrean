"""The 8 analysis tools. Each is a stateless object that takes a buffer
snapshot and returns a *ToolResult* with a score in [-1, 1] and metadata."""
from .base_tool            import BaseTool, ToolResult
from .market_structure     import MarketStructureTool
from .supply_demand        import SupplyDemandTool
from .liquidity            import LiquidityTool
from .momentum_volume      import MomentumVolumeTool
from .key_levels           import KeyLevelsTool
from .session_time         import SessionTimeTool
from .candlestick          import CandlestickTool
from .mtf_alignment        import MTFAlignmentTool

ALL_TOOLS = [
    MarketStructureTool,
    SupplyDemandTool,
    LiquidityTool,
    MomentumVolumeTool,
    KeyLevelsTool,
    SessionTimeTool,
    CandlestickTool,
    MTFAlignmentTool,
]

__all__ = [
    "BaseTool", "ToolResult",
    "MarketStructureTool", "SupplyDemandTool", "LiquidityTool",
    "MomentumVolumeTool", "KeyLevelsTool", "SessionTimeTool",
    "CandlestickTool", "MTFAlignmentTool", "ALL_TOOLS",
]
