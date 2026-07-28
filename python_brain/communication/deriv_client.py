"""
deriv_client.py
===============
A direct communication layer for the Deriv API.
Allows the bot to fetch data and execute trades without MT5.
"""
import asyncio
import json
import time
from typing import Optional, Callable, Dict, Any
import websockets
from ..config import DERIV_CFG
from ..utils.logger import get_logger

log = get_logger("deriv_client")

class DerivClient:
    def __init__(self, app_id: int, api_token: str):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.authenticated = False
        self.is_running = False
        self.on_tick_callback: Optional[Callable] = None

    async def connect(self):
        log.info(f"Connecting to Deriv API (App ID: {self.app_id})...")
        self.ws = await websockets.connect(self.ws_url)
        self.is_running = True
        log.info("Connected to Deriv.")
        await self.authorize()

    async def authorize(self):
        if not self.api_token:
            log.warning("No Deriv API token provided. Trade execution will be disabled.")
            return

        request = {"authorize": self.api_token}
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)

        if "error" in data:
            log.error(f"Deriv Authorization Failed: {data['error']['message']}")
            self.authenticated = False
        else:
            log.info(f"Deriv Authorized: {data['authorize']['email']}")
            self.authenticated = True

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        self.on_tick_callback = callback
        request = {"ticks": symbol}
        await self.ws.send(json.dumps(request))
        log.info(f"Subscribed to ticks for {symbol}")

    async def get_historical_data(self, symbol: str, granularity: int, start_time: int, end_time: int):
        """Fetch historical candles in chunks if necessary."""
        all_candles = []
        current_end = end_time
        
        while current_end > start_time:
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 5000, # Max allowed usually
                "end": current_end,
                "start": start_time,
                "granularity": granularity,
                "style": "candles"
            }
            await self.ws.send(json.dumps(request))
            response = await self.ws.recv()
            data = json.loads(response)
            
            if "error" in data:
                log.error(f"Error fetching history: {data['error']['message']}")
                break
                
            candles = data.get("candles", [])
            if not candles:
                break
                
            all_candles = candles + all_candles
            # Update current_end to the start of the earliest candle fetched
            current_end = candles[0]["epoch"] - 1
            
            log.info(f"Fetched {len(candles)} candles for {symbol}, remaining: {current_end - start_time}s")
            await asyncio.sleep(0.5) # Rate limiting
            
        return all_candles

    async def execute_trade(self, symbol: str, action: str, amount: float, duration: int = 60):
        if not self.authenticated:
            log.error("Cannot execute trade: Not authorized.")
            return None

        # Deriv uses 'proposal' to get a price and then 'buy'
        contract_type = "CALL" if action == "BUY" else "PUT"
        
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": "s",
            "symbol": symbol
        }
        
        await self.ws.send(json.dumps(proposal_request))
        proposal_res = await self.ws.recv()
        proposal_data = json.loads(proposal_res)
        
        if "error" in proposal_data:
            log.error(f"Proposal Error: {proposal_data['error']['message']}")
            return None

        proposal_id = proposal_data["proposal"]["id"]
        
        buy_request = {
            "buy": proposal_id,
            "price": amount
        }
        
        await self.ws.send(json.dumps(buy_request))
        buy_res = await self.ws.recv()
        return json.loads(buy_res)

    async def start_listening(self):
        while self.is_running:
            try:
                response = await self.ws.recv()
                data = json.loads(response)
                if "tick" in data and self.on_tick_callback:
                    await self.on_tick_callback(data["tick"])
            except Exception as e:
                log.error(f"WebSocket Error: {e}")
                break

    async def disconnect(self):
        self.is_running = False
        if self.ws:
            await self.ws.close()
            log.info("Disconnected from Deriv.")
