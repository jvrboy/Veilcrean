//+------------------------------------------------------------------+
//|                                              Veilcrean_EA.mq5    |
//|                              Veilcrean — Adaptive AI Trading Bot |
//|                                                                     |
//|  ROLE: Data collector + trade executor.                            |
//|  The EA is DUMB by design. All thinking happens in the Python       |
//|  brain. This file only:                                             |
//|    1. Collects OHLCV data for all timeframes                       |
//|    2. Streams tick/account data to Python over ZMQ                 |
//|    3. Receives trade commands and executes them                    |
//|    4. Manages open positions (SL/TP, partial close, trailing)      |
//|    5. Sends heartbeats so Python knows we're alive                 |
//+------------------------------------------------------------------+
#property copyright   "Veilcrean"
#property link        "https://github.com/veilcrean"
#property version     "1.00"
#property description "Veilcrean EA — Data + Execution layer"
#property strict

// We embed the support includes via #include so the file is self-contained
// in source form. Compiled with the supporting .mqh files in the same dir.
#include "DataCollector.mqh"
#include "SocketLib.mqh"
#include "TradeExecutor.mqh"
#include "Heartbeat.mqh"

//--- Inputs ---------------------------------------------------------+
input string  InpZmqAddress      = "tcp://127.0.0.1:5555";  // ZMQ endpoint
input int     InpZmqPullPort     = 5556;                    // Python → EA port (PULL)
input int     InpHeartbeatSec    = 5;                       // Heartbeat interval (s)
input int     InpTickPushMs      = 250;                     // Tick push throttling (ms)
input int     InpCandleHistory   = 200;                     // Candles per TF to send
input bool    InpDebugLog        = true;                    // Verbose logging
input double  InpMaxSlippage     = 3.0;                     // Max slippage (points)
input double  InpMaxSpreadPts    = 30.0;                    // Max spread (points)

//--- Globals --------------------------------------------------------+
CDataCollector   g_collector;
CTradeExecutor   g_executor;
CHeartbeat       g_heartbeat;
SocketContext    g_zmq_ctx;
SocketPublisher  g_zmq_pub;        // EA → Python
SocketSubscriber g_zmq_sub;        // Python → EA

datetime g_last_tick_push    = 0;
datetime g_last_candle_push  = 0;
datetime g_last_heartbeat    = 0;

string   g_symbol;
long     g_magic = 7772025;        // unique magic number for Veilcrean

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = _Symbol;

   // 1. Init the data collector for the chart's symbol
   if(!g_collector.Init(g_symbol, InpCandleHistory))
   {
      Print("VEILCREAN: DataCollector init failed");
      return(INIT_FAILED);
   }

   // 2. Init trade executor with safety parameters
   if(!g_executor.Init(g_symbol, g_magic, InpMaxSlippage, InpMaxSpreadPts))
   {
      Print("VEILCREAN: TradeExecutor init failed");
      return(INIT_FAILED);
   }

   // 3. Init ZMQ sockets
   if(!g_zmq_ctx.Create())
   {
      Print("VEILCREAN: ZMQ context failed");
      return(INIT_FAILED);
   }
   if(!g_zmq_pub.Create(g_zmq_ctx, InpZmqAddress))
   {
      Print("VEILCREAN: ZMQ publisher failed at ", InpZmqAddress);
      return(INIT_FAILED);
   }
   string pull_addr = StringFormat("tcp://127.0.0.1:%d", InpZmqPullPort);
   if(!g_zmq_sub.Create(g_zmq_ctx, pull_addr))
   {
      Print("VEILCREAN: ZMQ subscriber failed at ", pull_addr);
      return(INIT_FAILED);
   }

   // 4. Init heartbeat
   g_heartbeat.Init(InpHeartbeatSec);

   Print("VEILCREAN: EA online on ", g_symbol, " | PUB=", InpZmqAddress,
         " | SUB=", pull_addr);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_zmq_pub.Destroy();
   g_zmq_sub.Destroy();
   g_zmq_ctx.Destroy();
   Print("VEILCREAN: EA shut down (reason=", reason, ")");
}

//+------------------------------------------------------------------+
//| Expert tick function — main loop                                   |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime now = TimeCurrent();

   // 1. Listen for any trade commands from Python (non-blocking)
   PollCommands();

   // 2. Update trade management (trailing stop, partial TP, etc.)
   g_executor.ManageOpenPositions();

   // 3. Throttled tick push (avoid flooding)
   if((now - g_last_tick_push) * 1000 >= InpTickPushMs)
   {
      PushMarketData("TICK");
      g_last_tick_push = now;
   }

   // 4. Push a candle snapshot whenever a new bar opens on the chart TF
   if(g_collector.IsNewBar())
   {
      PushMarketData("CANDLE");
      g_last_candle_push = now;
   }

   // 5. Heartbeat on its own cadence
   if(g_heartbeat.ShouldSend(now))
   {
      string hb = g_heartbeat.BuildPayload(g_symbol, AccountInfoDouble(ACCOUNT_BALANCE));
      g_zmq_pub.Send(hb);
      g_last_heartbeat = now;
      if(InpDebugLog) Print("VEILCREAN: heartbeat sent");
   }
}

//+------------------------------------------------------------------+
//| Build and send a market data packet                               |
//+------------------------------------------------------------------+
void PushMarketData(const string trigger)
{
   string packet = g_collector.BuildPacket(g_symbol, trigger);
   if(packet == "")
   {
      if(InpDebugLog) Print("VEILCREAN: collector returned empty packet");
      return;
   }
   g_zmq_pub.Send(packet);
   if(InpDebugLog) PrintFormat("VEILCREAN: pushed %s packet (%d bytes)",
                               trigger, StringLen(packet));
}

//+------------------------------------------------------------------+
//| Poll for trade commands from the Python brain                     |
//+------------------------------------------------------------------+
void PollCommands()
{
   while(g_zmq_sub.PollAvailable())
   {
      string msg = g_zmq_sub.ReceiveString();
      if(msg == "") continue;

      if(InpDebugLog) Print("VEILCREAN: command received: ", msg);

      string response = g_executor.HandleCommand(msg);
      g_zmq_pub.Send(response);
   }
}

//+------------------------------------------------------------------+
//| OnTrade — fired when a trade event happens (notify Python)        |
//+------------------------------------------------------------------+
void OnTrade()
{
   string packet = g_collector.BuildAccountPacket(g_symbol, "TRADE_EVENT");
   g_zmq_pub.Send(packet);
}

//+------------------------------------------------------------------+
//| OnTimer — fallback heartbeat if no tick arrives                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   // No-op currently; OnTick is the workhorse.
}
//+------------------------------------------------------------------+
