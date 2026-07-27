//+------------------------------------------------------------------+
//|                                           TradeExecutor.mqh        |
//|                  Veilcrean — Receives and executes trade commands   |
//+------------------------------------------------------------------+
#property copyright "Veilcrean"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

class CTradeExecutor
{
private:
   string  m_symbol;
   long    m_magic;
   double  m_max_slippage;
   double  m_max_spread_pts;
   CTrade  m_trade;
   CPositionInfo m_pos;

   //--- helpers for parsing Python JSON commands -----------------
   string  GetJsonString(const string json, const string key);
   double  GetJsonNumber(const string json, const string key);

   //--- order helpers --------------------------------------------
   bool    OpenPosition(const string direction, double lots, double sl, double tp, double confidence);
   bool    ClosePosition(ulong ticket, double percent=100.0);
   bool    ModifySlTp(ulong ticket, double sl, double tp);
   bool    TrailStop(ulong ticket, double trail_pts);
   bool    CheckSpreadOk();
   string  ResultJson(const string action, bool success, const string msg, ulong ticket=0);

public:
   bool    Init(string symbol, long magic, double max_slippage, double max_spread);
   void    ManageOpenPositions();
   string  HandleCommand(const string json_cmd);
};

//+------------------------------------------------------------------+
//| Init                                                              |
//+------------------------------------------------------------------+
bool CTradeExecutor::Init(string symbol, long magic, double max_slippage, double max_spread)
{
   m_symbol         = symbol;
   m_magic          = magic;
   m_max_slippage   = max_slippage;
   m_max_spread_pts = max_spread;

   m_trade.SetExpertMagicNumber(m_magic);
   m_trade.SetDeviationInPoints((int)m_max_slippage);
   m_trade.SetTypeFilling(ORDER_FILLING_FOK);

   return true;
}

//+------------------------------------------------------------------+
//| Very small JSON value extractor (assumes well-formed input)        |
//+------------------------------------------------------------------+
string CTradeExecutor::GetJsonString(const string json, const string key)
{
   string needle = "\"" + key + "\":\"";
   int p = StringFind(json, needle);
   if(p < 0) return "";
   p += StringLen(needle);
   int e = StringFind(json, "\"", p);
   if(e < 0) return "";
   return StringSubstr(json, p, e - p);
}

double CTradeExecutor::GetJsonNumber(const string json, const string key)
{
   string needle = "\"" + key + "\":";
   int p = StringFind(json, needle);
   if(p < 0) return 0.0;
   p += StringLen(needle);
   // skip whitespace
   while(p < StringLen(json) && (StringGetCharacter(json,p) == ' ' || StringGetCharacter(json,p) == '\t')) p++;
   string num = "";
   while(p < StringLen(json))
   {
      ushort c = StringGetCharacter(json, p);
      if((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') { num += ShortToString(c); p++; }
      else break;
   }
   return StringToDouble(num);
}

//+------------------------------------------------------------------+
//| Spread sanity check                                                |
//+------------------------------------------------------------------+
bool CTradeExecutor::CheckSpreadOk()
{
   MqlTick tick;
   if(!SymbolInfoTick(m_symbol, tick)) return false;
   double spread_pts = (tick.ask - tick.bid) / _Point;
   return (spread_pts <= m_max_spread_pts);
}

//+------------------------------------------------------------------+
//| Open a new position                                                |
//+------------------------------------------------------------------+
bool CTradeExecutor::OpenPosition(const string direction, double lots, double sl, double tp, double confidence)
{
   if(!CheckSpreadOk()) { Print("VEILCREAN EXEC: spread too wide, abort"); return false; }
   if(lots <= 0)        { Print("VEILCREAN EXEC: invalid lot size"); return false; }

   MqlTick tick;
   if(!SymbolInfoTick(m_symbol, tick)) return false;

   double price = (direction == "BUY") ? tick.ask : tick.bid;
   ENUM_ORDER_TYPE type = (direction == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   string comment = StringFormat("VEIL_%.2f", confidence);
   bool ok = m_trade.PositionOpen(m_symbol, type, lots, price, sl, tp, comment);
   if(!ok)
      PrintFormat("VEILCREAN EXEC: open failed err=%d msg=%s",
                  GetLastError(), m_trade.ResultRetcodeDescription());
   return ok;
}

//+------------------------------------------------------------------+
//| Close a position (full or partial)                                 |
//+------------------------------------------------------------------+
bool CTradeExecutor::ClosePosition(ulong ticket, double percent=100.0)
{
   if(!m_pos.SelectByTicket(ticket)) return false;
   if(m_pos.Symbol() != m_symbol) return false;
   if(percent >= 99.999) return m_trade.PositionClose(ticket);
   // partial close
   double lots = NormalizeDouble(m_pos.Volume() * percent / 100.0, 2);
   if(lots < m_pos.Volume() * 0.01) lots = m_pos.Volume() * 0.01; // min 1%
   return m_trade.PositionClosePartial(ticket, lots);
}

//+------------------------------------------------------------------+
//| Modify SL/TP of an open position                                   |
//+------------------------------------------------------------------+
bool CTradeExecutor::ModifySlTp(ulong ticket, double sl, double tp)
{
   if(!m_pos.SelectByTicket(ticket)) return false;
   if(m_pos.Symbol() != m_symbol) return false;
   return m_trade.PositionModify(ticket, sl, tp);
}

//+------------------------------------------------------------------+
//| Trailing stop helper                                               |
//+------------------------------------------------------------------+
bool CTradeExecutor::TrailStop(ulong ticket, double trail_pts)
{
   if(!m_pos.SelectByTicket(ticket)) return false;
   if(m_pos.Symbol() != m_symbol) return false;

   MqlTick tick;
   if(!SymbolInfoTick(m_symbol, tick)) return false;

   double new_sl = 0.0;
   double trail  = trail_pts * _Point;

   if(m_pos.PositionType() == POSITION_TYPE_BUY)
   {
      new_sl = NormalizeDouble(tick.bid - trail, _Digits);
      if(new_sl > m_pos.StopLoss() + _Point && new_sl < tick.bid)
         return m_trade.PositionModify(ticket, new_sl, m_pos.TakeProfit());
   }
   else if(m_pos.PositionType() == POSITION_TYPE_SELL)
   {
      new_sl = NormalizeDouble(tick.ask + trail, _Digits);
      if((new_sl < m_pos.StopLoss() - _Point || m_pos.StopLoss() == 0) && new_sl > tick.ask)
         return m_trade.PositionModify(ticket, new_sl, m_pos.TakeProfit());
   }
   return false;
}

//+------------------------------------------------------------------+
//| Trailing / management on every tick                                |
//+------------------------------------------------------------------+
void CTradeExecutor::ManageOpenPositions()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
      // Basic trailing at 50 pts — can be tuned or driven from Python
      TrailStop(ticket, 50.0);
   }
}

//+------------------------------------------------------------------+
//| Build a structured result JSON to send back to Python              |
//+------------------------------------------------------------------+
string CTradeExecutor::ResultJson(const string action, bool success, const string msg, ulong ticket=0)
{
   return StringFormat(
      "{\"type\":\"EXEC_RESULT\",\"action\":\"%s\",\"success\":%s,\"msg\":\"%s\",\"ticket\":%d,\"timestamp\":%d}",
      action, (success ? "true" : "false"), msg, (long)ticket, (long)TimeCurrent());
}

//+------------------------------------------------------------------+
//| Main command dispatcher                                            |
//|   Expects JSON like:                                               |
//|     {"action":"OPEN","direction":"BUY","symbol":"EURUSD",          |
//|      "lot_size":0.05,"sl":1.0820,"tp":1.0890,"confidence":0.87}   |
//+------------------------------------------------------------------+
string CTradeExecutor::HandleCommand(const string json_cmd)
{
   string action    = GetJsonString(json_cmd, "action");
   string direction = GetJsonString(json_cmd, "direction");
   string symbol    = GetJsonString(json_cmd, "symbol");
   double lots      = GetJsonNumber(json_cmd, "lot_size");
   double sl        = GetJsonNumber(json_cmd, "sl");
   double tp        = GetJsonNumber(json_cmd, "tp");
   double conf      = GetJsonNumber(json_cmd, "confidence");
   ulong ticket     = (ulong)GetJsonNumber(json_cmd, "ticket");

   // If a symbol is specified, only execute if it matches ours
   if(symbol != "" && symbol != m_symbol)
      return ResultJson(action, false, "symbol mismatch");

   if(action == "OPEN")
   {
      bool ok = OpenPosition(direction, lots, sl, tp, conf);
      return ResultJson("OPEN", ok, ok ? "opened" : "open failed",
                        ok ? m_trade.ResultOrder() : 0);
   }
   else if(action == "CLOSE")
   {
      bool ok = ClosePosition(ticket, 100.0);
      return ResultJson("CLOSE", ok, ok ? "closed" : "close failed", ticket);
   }
   else if(action == "PARTIAL_CLOSE")
   {
      double pct = GetJsonNumber(json_cmd, "percent");
      bool ok = ClosePosition(ticket, pct);
      return ResultJson("PARTIAL_CLOSE", ok, ok ? "partial closed" : "partial failed", ticket);
   }
   else if(action == "MODIFY")
   {
      bool ok = ModifySlTp(ticket, sl, tp);
      return ResultJson("MODIFY", ok, ok ? "modified" : "modify failed", ticket);
   }
   else if(action == "FLATTEN_ALL")
   {
      // Emergency close everything on this symbol
      int closed = 0;
      for(int i=PositionsTotal()-1;i>=0;i--)
      {
         ulong t = PositionGetTicket(i);
         if(t == 0) continue;
         if(PositionGetString(POSITION_SYMBOL) == m_symbol)
            if(ClosePosition(t, 100.0)) closed++;
      }
      return ResultJson("FLATTEN_ALL", true, StringFormat("closed %d positions", closed));
   }
   else if(action == "PING")
   {
      return "{\"type\":\"PONG\",\"timestamp\":" + IntegerToString((long)TimeCurrent()) + "}";
   }

   return ResultJson(action, false, "unknown action");
}
//+------------------------------------------------------------------+
