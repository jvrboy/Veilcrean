//+------------------------------------------------------------------+
//|                                             Heartbeat.mqh          |
//|                  Veilcrean — Heartbeat + liveness payload          |
//+------------------------------------------------------------------+
#property copyright "Veilcrean"
#property strict

class CHeartbeat
{
private:
   int     m_interval_sec;
   datetime m_last_sent;
public:
   void Init(int interval_sec)
   {
      m_interval_sec = (interval_sec < 1) ? 1 : interval_sec;
      m_last_sent    = 0;
   }
   bool ShouldSend(const datetime now)
   {
      if(m_last_sent == 0) { m_last_sent = now; return true; }
      if((int)(now - m_last_sent) >= m_interval_sec) { m_last_sent = now; return true; }
      return false;
   }
   string BuildPayload(const string symbol, const double balance)
   {
      return StringFormat(
         "{\"type\":\"HEARTBEAT\",\"symbol\":\"%s\",\"balance\":%.2f,\"timestamp\":%d}",
         symbol, balance, (long)TimeCurrent());
   }
};
//+------------------------------------------------------------------+
