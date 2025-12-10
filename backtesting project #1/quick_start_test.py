"""
Quick Start Test Script
========================
Run this FIRST to verify your setup is working correctly.

This script:
1. Tests the backtesting.py import
2. Loads a sample of your data
3. Runs a simple SMA crossover backtest
4. Shows you a plot

If this works, you're ready for the full RSI+MA strategy!
"""

import pandas as pd
from pathlib import Path
import sys

# ============================================================================
# STEP 1: Setup paths
# ============================================================================

# Path to your backtesting.py library
BACKTESTING_PATH = Path(r"C:\Users\estac\OneDrive - Syracuse University\trading\backtesting project #1\backtesting.py-master")
sys.path.insert(0, str(BACKTESTING_PATH))

# Test the import
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    print("✅ backtesting.py imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nMake sure the path above points to your backtesting.py-master folder")
    sys.exit(1)

# ============================================================================
# STEP 2: Load your data
# ============================================================================

DATA_PATH = Path(r"C:\Users\estac\OneDrive - Syracuse University\trading\backtesting project #1\eur-jpy Data")

# Try loading one file
test_file = DATA_PATH / "2024" / "csv" / "DAT_MT_EURJPY_M1_2024.csv"

if not test_file.exists():
    # Try 2025 data instead
    test_file = DATA_PATH / "2025" / "csv" / "DAT_MT_EURJPY_M1_202501.csv"

print(f"\nLoading: {test_file}")

try:
    df = pd.read_csv(
        test_file,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
        parse_dates={'Datetime': ['Date', 'Time']},
        date_format='%Y.%m.%d %H:%M'
    )
    df.set_index('Datetime', inplace=True)
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Resample to 15-minute bars
# ============================================================================

df_15m = df.resample('15T').agg({
    'Open': 'first',
    'High': 'max', 
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"\n✅ Resampled to {len(df_15m):,} 15-minute bars")

# ============================================================================
# STEP 4: Define a simple test strategy
# ============================================================================

def SMA(series, period):
    return pd.Series(series).rolling(window=period).mean()


class SimpleSmaCross(Strategy):
    """Simple SMA crossover - just to test that everything works"""
    fast = 10
    slow = 30
    
    def init(self):
        self.sma_fast = self.I(SMA, self.data.Close, self.fast)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow)
    
    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.sell()


# ============================================================================
# STEP 5: Run backtest
# ============================================================================

print("\n" + "="*50)
print("Running test backtest...")
print("="*50)

bt = Backtest(
    df_15m,
    SimpleSmaCross,
    cash=10000,
    commission=0.0001,
    exclusive_orders=True
)

stats = bt.run()

print("\n✅ Backtest completed!")
print(f"\nReturn: {stats['Return [%]']:.2f}%")
print(f"Trades: {stats['# Trades']}")
print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")

# ============================================================================
# STEP 6: Generate plot
# ============================================================================

print("\n" + "="*50)
print("Generating plot (will open in browser)...")
print("="*50)

bt.plot(
    filename='quick_test_results.html',
    open_browser=True
)

print("\n🎉 SUCCESS! Your setup is working!")
print("\nNext steps:")
print("1. Close the plot browser tab")
print("2. Run the full RSI+MA strategy: python rsi_ma_strategy.py")
