class RiskManager:
    def __init__(self, config):
        self.config = config

    def calculate_dynamic_leverage(self, confidence: float) -> int:
        leverage = 1
        for threshold, lev in sorted(self.config['leverage_tiers'].items(), reverse=True):
            if confidence >= threshold:
                leverage = lev
                break
        return leverage

    def calculate_position_size(self, portfolio_value: float, confidence: float) -> tuple[float, int]:
        leverage = self.calculate_dynamic_leverage(confidence)
        trade_capital = portfolio_value * self.config['portfolio_pct_per_trade']
        position_size = trade_capital * leverage
        # Note: CCXT amount is in base currency, so you'd need to divide by price.
        # For simplicity, many futures exchanges allow quote currency sizing.
        # This implementation assumes the amount can be specified directly.
        # A more robust solution would be: position_size / current_price.
        return position_size, leverage

    def calculate_trailing_sl(self, trade: dict, current_price: float) -> float:
        atr = trade['atr']
        sl_distance = atr * self.config['trailing_sl_atr_multiplier']
        
        new_sl_price = trade.get('sl_price', 0) # Get current SL or 0 if none
        
        if trade['side'] == 'long':
            # Initial SL is set below entry
            if new_sl_price == 0:
                new_sl_price = trade['entry_price'] - sl_distance
            # Trail the stop up
            potential_new_sl = current_price - sl_distance
            if potential_new_sl > new_sl_price:
                new_sl_price = potential_new_sl
        
        elif trade['side'] == 'short':
            # Initial SL is set above entry
            if new_sl_price == 0:
                new_sl_price = trade['entry_price'] + sl_distance
            # Trail the stop down
            potential_new_sl = current_price + sl_distance
            if potential_new_sl < new_sl_price:
                new_sl_price = potential_new_sl
                
        return new_sl_price