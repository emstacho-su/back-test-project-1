"""
RSI + Moving Average Convergence Strategy
==========================================
A multi-indicator system combining momentum (RSI) with trend confirmation (MA).

Strategy Logic:
- LONG when: RSI > 50 AND Fast MA > Slow MA (bullish momentum + uptrend)
- SHORT when: RSI < 50 AND Fast MA < Slow MA (bearish momentum + downtrend)
- EXIT LONG when: RSI < 30 OR Fast MA crosses below Slow MA
- EXIT SHORT when: RSI > 70 OR Fast MA crosses above Slow MA

Author: Your Backtesting Project
Data: EUR/JPY 15-minute bars (resampled from 1-minute MetaTrader data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# SETUP: Add backtesting.py to Python path
# ============================================================================
# This allows us to import from the local backtesting.py-master folder
# Adjust this path if your folder structure is different

# For Windows with your specific path:
BACKTESTING_PATH = Path(r"C:\Users\estac\OneDrive - Syracuse University\trading\backtesting project #1\backtesting.py-master")
sys.path.insert(0, str(BACKTESTING_PATH))

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================
# We define our own indicator functions since backtesting.py has limited built-ins
# These return pandas Series that can be used with the self.I() wrapper

def SMA(series, period):
    """
    Simple Moving Average
    
    Why SMA over EMA for beginners:
    - Easier to understand and debug
    - Less sensitive to recent price spikes
    - Good baseline before trying EMA variants
    """
    return pd.Series(series).rolling(window=period).mean()


def RSI(series, period=14):
    """
    Relative Strength Index (RSI)
    
    RSI measures momentum by comparing average gains vs average losses.
    - RSI > 70: Overbought (potential reversal down)
    - RSI < 30: Oversold (potential reversal up)
    - RSI > 50: Bullish momentum
    - RSI < 50: Bearish momentum
    
    We use the smoothed/Wilder's RSI calculation (industry standard).
    """
    series = pd.Series(series)
    
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # First average: simple mean for initial value
    # Subsequent: exponential moving average (Wilder's smoothing)
    # Wilder's smoothing uses alpha = 1/period
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# ============================================================================
# STRATEGY CLASS
# ============================================================================

class RSI_MA_Strategy(Strategy):
    """
    RSI + Moving Average Convergence Strategy
    
    Parameters (can be optimized later):
    - rsi_period: Lookback for RSI calculation (default 14)
    - fast_ma: Fast moving average period (default 50)
    - slow_ma: Slow moving average period (default 200)
    - rsi_overbought: RSI level considered overbought (default 70)
    - rsi_oversold: RSI level considered oversold (default 30)
    """
    
    # Strategy parameters - these can be optimized using bt.optimize()
    rsi_period = 14
    fast_ma = 50
    slow_ma = 200
    rsi_overbought = 70
    rsi_oversold = 30
    rsi_bull_threshold = 50  # RSI above this = bullish momentum
    
    def init(self):
        """
        Initialize indicators.
        
        The self.I() wrapper is CRITICAL - it:
        1. Ensures indicators are calculated only on data available at each bar
        2. Prevents look-ahead bias
        3. Handles the indicator array alignment automatically
        """
        # Price data for indicator calculation
        close = self.data.Close
        
        # RSI indicator
        self.rsi = self.I(RSI, close, self.rsi_period)
        
        # Moving averages for trend confirmation
        self.sma_fast = self.I(SMA, close, self.fast_ma)
        self.sma_slow = self.I(SMA, close, self.slow_ma)
    
    def next(self):
        """
        Called on every bar to make trading decisions.
        
        Key principle: self.data.Close[-1] is the CURRENT bar's close
        self.data.Close[-2] is the PREVIOUS bar's close
        
        We only see data up to the current bar - no future information.
        """
        # Current indicator values
        rsi_now = self.rsi[-1]
        sma_fast_now = self.sma_fast[-1]
        sma_slow_now = self.sma_slow[-1]
        
        # Skip if indicators aren't ready (NaN during warmup period)
        if pd.isna(rsi_now) or pd.isna(sma_fast_now) or pd.isna(sma_slow_now):
            return
        
        # ===== ENTRY LOGIC =====
        
        # Trend direction from MAs
        uptrend = sma_fast_now > sma_slow_now
        downtrend = sma_fast_now < sma_slow_now
        
        # Momentum from RSI
        bullish_momentum = rsi_now > self.rsi_bull_threshold
        bearish_momentum = rsi_now < self.rsi_bull_threshold
        
        # LONG ENTRY: Bullish momentum + Uptrend confirmation
        if not self.position:  # Only enter if flat
            if bullish_momentum and uptrend:
                self.buy()
            elif bearish_momentum and downtrend:
                self.sell()  # Short entry
        
        # ===== EXIT LOGIC =====
        
        elif self.position.is_long:
            # Exit long if: RSI oversold OR trend reversal
            exit_signal = (
                rsi_now < self.rsi_oversold or  # RSI oversold
                crossover(self.sma_slow, self.sma_fast)  # Death cross
            )
            if exit_signal:
                self.position.close()
        
        elif self.position.is_short:
            # Exit short if: RSI overbought OR trend reversal
            exit_signal = (
                rsi_now > self.rsi_overbought or  # RSI overbought
                crossover(self.sma_fast, self.sma_slow)  # Golden cross
            )
            if exit_signal:
                self.position.close()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_metatrader_csv(file_path):
    """
    Load a single MetaTrader format CSV file.
    
    MetaTrader format (no header):
    Date, Time, Open, High, Low, Close, Volume
    2025.12.01,00:00,180.413000,180.418000,180.399000,180.400000,0
    """
    df = pd.read_csv(
        file_path,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
        parse_dates={'Datetime': ['Date', 'Time']},  # Combine date+time columns
        date_format='%Y.%m.%d %H:%M'
    )
    df.set_index('Datetime', inplace=True)
    return df


def load_multiple_csvs(folder_path, pattern="*.csv"):
    """
    Load and combine multiple CSV files from a folder.
    
    Useful for combining monthly files (2025/01, 2025/02, etc.)
    """
    folder = Path(folder_path)
    all_files = sorted(folder.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    dfs = []
    for f in all_files:
        print(f"Loading: {f.name}")
        dfs.append(load_metatrader_csv(f))
    
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='first')]  # Remove duplicates
    combined.sort_index(inplace=True)
    
    return combined


def resample_ohlcv(df, timeframe='15T'):
    """
    Resample 1-minute bars to a higher timeframe.
    
    Timeframe options:
    - '15T' or '15min' = 15 minutes
    - '30T' or '30min' = 30 minutes  
    - '1H' or '60min' = 1 hour
    - '4H' = 4 hours
    - '1D' = 1 day
    
    Why resample?
    - Reduces noise in the data
    - Aligns with your target trading timeframe (15m-1h)
    - Faster backtesting
    """
    resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    
    # Path to your data (adjust if needed)
    # Use the existing `data/eur-jpy Data` folder inside the repo workspace
    DATA_PATH = Path(r"C:\Users\estac\OneDrive - Syracuse University\trading\backtesting project #1\data\eur-jpy Data")
    
    # Choose which data to load:
    # Option 1: Single file (2024 full year)
    # data_file = DATA_PATH / "2024" / "csv" / "DAT_MT_EURJPY_M1_2024.csv"
    # df_1m = load_metatrader_csv(data_file)
    
    # Option 2: Multiple monthly files (2025)
    data_folder = DATA_PATH / "2025" / "csv"
    df_1m = load_multiple_csvs(data_folder)
    
    print(f"\nLoaded {len(df_1m):,} 1-minute bars")
    print(f"Date range: {df_1m.index[0]} to {df_1m.index[-1]}")
    
    # ===== RESAMPLE TO TARGET TIMEFRAME =====
    
    TIMEFRAME = '15T'  # 15 minutes - change to '1H' for hourly
    df = resample_ohlcv(df_1m, TIMEFRAME)
    
    print(f"\nResampled to {len(df):,} {TIMEFRAME} bars")
    print(f"Sample data:")
    print(df.head())
    
    # ===== RUN BACKTEST =====
    
    bt = Backtest(
        df,
        RSI_MA_Strategy,
        cash=10000,           # Starting capital
        commission=0.00007,   # ~0.7 pips spread for EUR/JPY
        margin=1/30,          # 30:1 leverage (typical for forex)
        trade_on_close=False, # Execute on next bar's open (realistic)
        exclusive_orders=True # Only one position at a time
    )
    
    # Run the backtest
    stats = bt.run()
    
    # ===== DISPLAY RESULTS =====
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(stats)
    
    # Key metrics to focus on
    print("\n" + "="*60)
    print("KEY PERFORMANCE METRICS")
    print("="*60)
    print(f"Total Return:     {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:     {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate:         {stats['Win Rate [%]']:.2f}%")
    print(f"Total Trades:     {stats['# Trades']}")
    print(f"Profit Factor:    {stats['Profit Factor']:.2f}")
    
    # ===== GENERATE INTERACTIVE PLOT =====
    
    print("\nGenerating interactive plot...")
    bt.plot(
    filename=r'C:\Users\estac\eurjpy_backtest.html',
    open_browser=True
)
    
    print("\nPlot saved to: eurjpy_backtest_results.html")
