import ccxt
import pandas as pd
import os
from datetime import datetime

# --- Configuration ---
PAIRS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '15m' 
YEARS_OF_DATA = 6
# ---------------------

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Connecting to Binance...")
exchange = ccxt.binance()

# Calculate the starting timestamp
since = exchange.parse8601(f'{datetime.now().year - YEARS_OF_DATA}-01-01T00:00:00Z')

for pair in PAIRS:
    try:
        print(f"Downloading historical data for {pair}...")
        
        # Fetch OHLCV (Open, High, Low, Close, Volume) data
        ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, since, limit=100000)
        
        # Convert to a pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to a readable format (optional, but good practice)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Define the filename
        filename = f"data/{pair.replace('/', '_')}_{TIMEFRAME}.csv"
        
        # Save to a CSV file
        df.to_csv(filename, index=False)
        
        print(f"✅ Successfully saved data for {pair} to {filename}")
        
    except Exception as e:
        print(f"❌ Could not download data for {pair}. Error: {e}")

print("\nAll data downloaded!")