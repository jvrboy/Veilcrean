//+------------------------------------------------------------------+
//|                                              SocketLib.mqh        |
//|           Veilcrean — ZeroMQ wrapper (uses libzmq via DLL)         |
//|                                                                     |
//|  Implementation note:                                               |
//|    This wrapper talks to libzmq.dll (or libzmq-v140-mt-4_3_5.dll)  |
//|    via #import. The DLL must be present in MQL5/Libraries.         |
//|    See docs/MQ5_ZMQ_SETUP.md for installation instructions.        |
//+------------------------------------------------------------------+
#property copyright "Veilcrean"
#property strict

#import "libzmq.dll"
long  zmq_ctx_new();
int   zmq_ctx_term(long context);
long  zmq_socket(long context, int type);
int   zmq_bind(long socket, const string endpoint);
int   zmq_connect(long socket, const string endpoint);
int   zmq_send(long socket, const uchar &data[], int len, int flags);
int   zmq_recv(long socket, uchar &data[], int len, int flags);
int   zmq_setsockopt(long socket, int option, const int &value, int len);
int   zmq_close(long socket);
int   zmq_poll(void &items[], int nitems, long timeout);
#import

// ZMQ socket types
#define ZMQ_PUB    1
#define ZMQ_SUB    2
#define ZMQ_PULL   7
#define ZMQ_PUSH   8

// ZMQ send/recv flags
#define ZMQ_DONTWAIT 1
#define ZMQ_NOBLOCK  1
#define ZMQ_SNDMORE  2

// ZMQ socket options
#define ZMQ_RCVTIMEO 27
#define ZMQ_SNDTIMEO 28
#define ZMQ_SUBSCRIBE 6

// poll item struct for non-blocking receive
struct pollitem_t
{
   long  socket;
   short events;
   short revents;
};

//+------------------------------------------------------------------+
//| Context wrapper                                                   |
//+------------------------------------------------------------------+
class SocketContext
{
private:
   long m_ctx;
public:
   bool   Create()   { m_ctx = zmq_ctx_new(); return (m_ctx != 0); }
   void   Destroy()  { if(m_ctx != 0) { zmq_ctx_term(m_ctx); m_ctx = 0; } }
   long   Handle()   { return m_ctx; }
};

//+------------------------------------------------------------------+
//| Generic socket wrapper                                            |
//+------------------------------------------------------------------+
class SocketBase
{
protected:
   long   m_socket;
public:
   virtual bool Create(SocketContext &ctx, const string endpoint) = 0;
   void          Destroy() { if(m_socket != 0) { zmq_close(m_socket); m_socket = 0; } }
   long          Handle()  { return m_socket; }

   // Receive a JSON-ish string (non-blocking)
   string ReceiveString()
   {
      uchar buf[65536];
      ArrayInitialize(buf, 0);
      int n = zmq_recv(m_socket, buf, ArraySize(buf), ZMQ_DONTWAIT);
      if(n <= 0) return "";
      return StringSubstr(CharArrayToString(buf, 0, n), 0, n);
   }

   // Send a string
   int Send(const string msg)
   {
      uchar buf[];
      StringToCharArray(msg, buf, 0, StringLen(msg));
      return zmq_send(m_socket, buf, ArraySize(buf), 0);
   }

   // Returns true if data is available
   bool PollAvailable(int timeout_ms=1)
   {
      pollitem_t items[1];
      items[0].socket = m_socket;
      items[0].events = 1; // POLLIN
      items[0].revents = 0;
      int rc = zmq_poll(items, 1, timeout_ms);
      return (rc > 0 && (items[0].revents & 1) != 0);
   }
};

//+------------------------------------------------------------------+
//| Publisher — EA pushes data to Python                              |
//+------------------------------------------------------------------+
class SocketPublisher : public SocketBase
{
public:
   bool Create(SocketContext &ctx, const string endpoint) override
   {
      m_socket = zmq_socket(ctx.Handle(), ZMQ_PUB);
      if(m_socket == 0) return false;
      // Set short send timeout to avoid blocking the EA
      int tmo = 1000;
      zmq_setsockopt(m_socket, ZMQ_SNDTIMEO, tmo, sizeof(int));
      return (zmq_bind(m_socket, endpoint) == 0);
   }
};

//+------------------------------------------------------------------+
//| Subscriber — EA receives commands from Python                     |
//+------------------------------------------------------------------+
class SocketSubscriber : public SocketBase
{
public:
   bool Create(SocketContext &ctx, const string endpoint) override
   {
      m_socket = zmq_socket(ctx.Handle(), ZMQ_PULL);
      if(m_socket == 0) return false;
      int tmo = 1000;
      zmq_setsockopt(m_socket, ZMQ_RCVTIMEO, tmo, sizeof(int));
      return (zmq_connect(m_socket, endpoint) == 0);
   }
};
//+------------------------------------------------------------------+
