import os
import asyncio
import logging
import yaml
import pandas as pd
import ccxt.pro as ccxt
from bot_core.model_handler import HybridModel
from bot_core.risk_manager import RiskManager
from bot_core.state_manager import StateManager
from bot_core.telegram_handler import TelegramHandler

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger("TradingBot")

class ProductionTrader:
    def __init__(self, config):
        self.config = config
        self.exchange = self.connect_exchange()
        self.risk_manager = RiskManager(config['risk'])
        self.model = HybridModel(config['model'])
        self.state = StateManager()
        self.telegram = TelegramHandler(config['telegram'])
        self.is_running = True

    def connect_exchange(self):
        exchange = ccxt.binance({
            'apiKey': self.config['exchange']['api_key'],
            'secret': self.config['exchange']['api_secret'],
            'options': {'defaultType': 'future'},
        })
        if self.config['exchange']['is_testnet']:
            exchange.set_sandbox_mode(True)
        logger.info(f"Connected to Binance. Testnet: {self.config['exchange']['is_testnet']}")
        return exchange

    async def run(self):
        logger.info("Starting trading bot...")
        await self.telegram.send_message("🤖 Trading Bot Started!")
        
        # Create two main tasks that run forever
        trade_execution_task = asyncio.create_task(self.trade_cycle())
        position_monitoring_task = asyncio.create_task(self.monitor_open_positions())
        
        await asyncio.gather(trade_execution_task, position_monitoring_task)

    async def trade_cycle(self):
        """Main loop to check for new trading signals."""
        while self.is_running:
            try:
                open_positions = len(self.state.get_open_trades())
                if open_positions >= self.config['risk']['max_concurrent_trades']:
                    logger.info(f"Max concurrent trades ({open_positions}) reached. Waiting.")
                    await asyncio.sleep(60)
                    continue

                for pair in self.config['strategy']['pairs']:
                    await self.evaluate_pair(pair)
                
                logger.info("Trade cycle finished. Waiting for next candle...")
                await asyncio.sleep(60) # Wait a minute before the next cycle

            except Exception as e:
                logger.error(f"An error occurred in the main trade cycle: {e}", exc_info=True)
                await self.telegram.send_message(f"🚨 CRITICAL ERROR in trade cycle: {e}")
                await asyncio.sleep(60)

    async def evaluate_pair(self, pair: str):
        """Fetches data, gets a prediction, and executes a trade if signal is strong."""
        try:
            logger.info(f"Evaluating {pair}...")
            # 1. Fetch Data
            ohlcv = await self.exchange.fetch_ohlcv(pair, self.config['strategy']['timeframe'], limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # 2. Get Prediction
            features = self.model.process_features(df)
            if features.empty:
                logger.warning(f"Not enough data to generate features for {pair}")
                return
            
            latest_features = features.iloc[-1:]
            prob_long, prob_short = self.model.predict(latest_features)
            
            logger.info(f"{pair} -> Long Prob: {prob_long:.2f}, Short Prob: {prob_short:.2f}")

            # 3. Decision Making
            if prob_long > self.config['strategy']['min_probability']:
                await self.execute_trade(pair, 'long', prob_long, latest_features)
            elif prob_short > self.config['strategy']['min_probability']:
                await self.execute_trade(pair, 'short', prob_short, latest_features)

        except Exception as e:
            logger.error(f"Failed to evaluate {pair}: {e}", exc_info=True)

    async def execute_trade(self, pair: str, side: str, confidence: float, features: pd.DataFrame):
        """Places a market order and records the trade in the database."""
        try:
            balance = await self.exchange.fetch_balance()
            portfolio_value = float(balance['total']['USDT'])
            
            amount, leverage = self.risk_manager.calculate_position_size(
                portfolio_value=portfolio_value,
                confidence=confidence
            )
            
            await self.exchange.set_leverage(leverage, pair)
            
            order_side = 'buy' if side == 'long' else 'sell'
            
            logger.info(f"Placing {side} order for {amount} of {pair} at {leverage}x leverage.")
            
            # Create the market order
            order = await self.exchange.create_order(pair, 'market', order_side, amount)
            
            # Record the trade in our database
            trade_id = self.state.add_trade(
                pair=pair,
                side=side,
                entry_price=float(order['price']),
                amount=float(order['amount']),
                leverage=leverage,
                atr=features['volatility_atr'].iloc[-1]
            )
            
            logger.info(f"✅ Successfully opened {side} trade for {pair}. Trade ID: {trade_id}")
            await self.telegram.send_message(
                f"📈 NEW TRADE [{side.upper()}]\n"
                f"Pair: {pair}\n"
                f"Entry: ${order['price']}\n"
                f"Size: {order['amount']}\n"
                f"Leverage: {leverage}x"
            )

        except Exception as e:
            logger.error(f"Failed to execute trade for {pair}: {e}", exc_info=True)
            await self.telegram.send_message(f"❌ Failed to execute trade for {pair}: {e}")

    async def monitor_open_positions(self):
        """Continuously monitors open trades and manages trailing stop-loss."""
        while self.is_running:
            try:
                open_trades = self.state.get_open_trades()
                if not open_trades:
                    await asyncio.sleep(10) # Wait longer if no trades
                    continue

                tickers = await self.exchange.fetch_tickers([trade['pair'] for trade in open_trades])
                
                for trade in open_trades:
                    current_price = tickers[trade['pair']]['last']
                    await self.manage_trailing_stop(trade, current_price)

                await asyncio.sleep(15) # Check prices every 15 seconds

            except Exception as e:
                logger.error(f"An error occurred in the position monitor: {e}", exc_info=True)
                await self.telegram.send_message(f"🚨 CRITICAL ERROR in position monitor: {e}")
                await asyncio.sleep(60)

    async def manage_trailing_stop(self, trade: dict, current_price: float):
        """Adjusts and triggers the trailing stop loss for a single trade."""
        try:
            sl_price = self.risk_manager.calculate_trailing_sl(trade, current_price)
            self.state.update_sl_price(trade['id'], sl_price) # Update db with new SL

            should_close = False
            if trade['side'] == 'long' and current_price < sl_price:
                should_close = True
                profit_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
            elif trade['side'] == 'short' and current_price > sl_price:
                should_close = True
                profit_pct = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
            
            if should_close:
                logger.info(f"Trailing Stop Hit for {trade['pair']}! Closing position.")
                order_side = 'sell' if trade['side'] == 'long' else 'buy'
                
                # Close position with a market order
                await self.exchange.create_order(trade['pair'], 'market', order_side, trade['amount'], {'reduceOnly': True})
                
                self.state.close_trade(trade['id'], close_price=current_price)
                
                logger.info(f"✅ Position for {trade['pair']} closed. Profit: {profit_pct:.2f}%")
                await self.telegram.send_message(
                    f"💰 TRADE CLOSED\n"
                    f"Pair: {trade['pair']}\n"
                    f"Reason: Trailing Stop Hit\n"
                    f"Profit: {profit_pct:.2f}%"
                )

        except Exception as e:
            logger.error(f"Failed to manage trailing stop for {trade['pair']} (ID: {trade['id']}): {e}")

    async def shutdown(self):
        self.is_running = False
        logger.info("Shutting down bot...")
        if self.exchange:
            await self.exchange.close()
        logger.info("Exchange connection closed. Goodbye!")


async def main():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("FATAL: config.yaml not found. Please create it.")
        return

    bot = ProductionTrader(config)
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())