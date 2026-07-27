//+------------------------------------------------------------------+
//|                                            DataCollector.mqh       |
//|                     Veilcrean — Data collection routines           |
//+------------------------------------------------------------------+
#property copyright "Veilcrean"
#property strict

#include <Trade\SymbolInfo.mqh>

// All timeframes Veilcrean reasons about.
#define VEIL_TF_COUNT 9
static ENUM_TIMEFRAMES VEIL_TIMEFRAMES[VEIL_TF_COUNT] = {
   PERIOD_M1,
   PERIOD_M5,
   PERIOD_M15,
   PERIOD_M30,
   PERIOD_H1,
   PERIOD_H4,
   PERIOD_D1,
   PERIOD_W1,
   PERIOD_MN1
};
static const string VEIL_TF_NAMES[VEIL_TF_COUNT] = {
   "M1","M5","M15","M30","H1","H4","D1","W1","MN1"
};

class CDataCollector
{
private:
   string            m_symbol;
   int               m_history;
   datetime          m_last_bar_time[VEIL_TF_COUNT];

   //--- helpers ----------------------------------------------------
   string            TfNameToString(ENUM_TIMEFRAMES tf) const;
   string            CandleToJson(MqlRates r) const;
   string            CandlesToJson(ENUM_TIMEFRAMES tf) const;
   string            TickToJson() const;
   string            AccountToJson() const;
   string            PositionsToJson() const;

public:
   bool              Init(string symbol, int history_bars);
   bool              IsNewBar();
   string            BuildPacket(const string symbol, const string trigger);
   string            BuildAccountPacket(const string symbol, const string trigger);

   CDataCollector()  { ArrayInitialize(m_last_bar_time, 0); }
};

//+------------------------------------------------------------------+
//| Init                                                              |
//+------------------------------------------------------------------+
bool CDataCollector::Init(string symbol, int history_bars)
{
   m_symbol  = symbol;
   m_history = history_bars;
   return true;
}

//+------------------------------------------------------------------+
//| Returns true if a new bar has opened on the chart's primary TF    |
//+------------------------------------------------------------------+
bool CDataCollector::IsNewBar()
{
   datetime t = iTime(m_symbol, PERIOD_CURRENT, 0);
   if(t == 0) return false;
   int idx = -1;
   // map PERIOD_CURRENT to our array index by name string
   string cur = TfNameToString(PERIOD_CURRENT);
   for(int i=0;i<VEIL_TF_COUNT;i++)
      if(VEIL_TF_NAMES[i] == cur) { idx = i; break; }
   if(idx < 0) return false;
   if(m_last_bar_time[idx] != t)
   {
      m_last_bar_time[idx] = t;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Convert an ENUM_TIMEFRAMES to its short name string               |
//+------------------------------------------------------------------+
string CDataCollector::TfNameToString(ENUM_TIMEFRAMES tf) const
{
   for(int i=0;i<VEIL_TF_COUNT;i++)
      if(VEIL_TIMEFRAMES[i] == tf) return VEIL_TF_NAMES[i];
   return "UNK";
}

//+------------------------------------------------------------------+
//| Single candle → JSON object                                       |
//+------------------------------------------------------------------+
string CDataCollector::CandleToJson(MqlRates r) const
{
   return StringFormat("{\"O\":%.%df,\"H\":%.%df,\"L\":%.%df,\"C\":%.%df,\"V\":%.%df,\"t\":%d}",
                       _Digits, r.open, _Digits, r.high, _Digits, r.low,
                       _Digits, r.close, 0, (long)r.tick_volume,
                       (long)r.time);
}

//+------------------------------------------------------------------+
//| All candles for one TF → JSON array                               |
//+------------------------------------------------------------------+
string CDataCollector::CandlesToJson(ENUM_TIMEFRAMES tf) const
{
   MqlRates rates[];
   int copied = CopyRates(m_symbol, tf, 0, m_history, rates);
   if(copied <= 0) return "[]";

   string out = "[";
   for(int i=0;i<copied;i++)
   {
      if(i>0) out += ",";
      out += CandleToJson(rates[i]);
   }
   out += "]";
   return out;
}

//+------------------------------------------------------------------+
//| Latest tick → JSON object                                         |
//+------------------------------------------------------------------+
string CDataCollector::TickToJson() const
{
   MqlTick tick;
   if(!SymbolInfoTick(m_symbol, tick)) return "{}";
   double spread_pts = (tick.ask - tick.bid) / _Point;
   return StringFormat(
      "{\"bid\":%.%df,\"ask\":%.%df,\"spread\":%.1f,\"volume\":%d}",
      _Digits, tick.bid, _Digits, tick.ask, spread_pts, (long)tick.volume);
}

//+------------------------------------------------------------------+
//| Account info → JSON object                                        |
//+------------------------------------------------------------------+
string CDataCollector::AccountToJson() const
{
   return StringFormat(
      "{\"balance\":%.2f,\"equity\":%.2f,\"margin_free\":%.2f,\"margin_used\":%.2f,\"profit\":%.2f,\"leverage\":%d}",
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_PROFIT),
      (int)AccountInfoInteger(ACCOUNT_LEVERAGE));
}

//+------------------------------------------------------------------+
//| Open positions on this symbol → JSON array                        |
//+------------------------------------------------------------------+
string CDataCollector::PositionsToJson() const
{
   string out = "[";
   int total = PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;

      string type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      out += StringFormat(
         "%s{\"ticket\":%d,\"type\":\"%s\",\"lots\":%.2f,\"open\":%.%df,\"sl\":%.%df,\"tp\":%.%df,\"profit\":%.2f}",
         (i>0 ? "," : ""), (long)ticket, type,
         PositionGetDouble(POSITION_VOLUME),
         _Digits, PositionGetDouble(POSITION_PRICE_OPEN),
         _Digits, PositionGetDouble(POSITION_SL),
         _Digits, PositionGetDouble(POSITION_TP),
         PositionGetDouble(POSITION_PROFIT));
   }
   out += "]";
   return out;
}

//+------------------------------------------------------------------+
//| Build a full market data packet                                   |
//+------------------------------------------------------------------+
string CDataCollector::BuildPacket(const string symbol, const string trigger)
{
   // Build candles object
   string candles = "{";
   for(int i=0;i<VEIL_TF_COUNT;i++)
   {
      if(i>0) candles += ",";
      candles += StringFormat("\"%s\":%s", VEIL_TF_NAMES[i], CandlesToJson(VEIL_TIMEFRAMES[i]));
   }
   candles += "}";

   // Assemble the whole packet (we use double quoting for JSON compatibility)
   string packet = StringFormat(
      "{\"type\":\"MARKET_DATA\","
      "\"symbol\":\"%s\","
      "\"trigger\":\"%s\","
      "\"timestamp\":%d,"
      "\"tick\":%s,"
      "\"candles\":%s,"
      "\"account\":%s,"
      "\"positions\":%s"
      "}",
      symbol, trigger, (long)TimeCurrent(),
      TickToJson(), candles, AccountToJson(), PositionsToJson());

   return packet;
}

//+------------------------------------------------------------------+
//| Build a smaller packet used for account / trade-event updates     |
//+------------------------------------------------------------------+
string CDataCollector::BuildAccountPacket(const string symbol, const string trigger)
{
   return StringFormat(
      "{\"type\":\"ACCOUNT_UPDATE\","
      "\"symbol\":\"%s\","
      "\"trigger\":\"%s\","
      "\"timestamp\":%d,"
      "\"account\":%s,"
      "\"positions\":%s"
      "}",
      symbol, trigger, (long)TimeCurrent(), AccountToJson(), PositionsToJson());
}
//+------------------------------------------------------------------+
