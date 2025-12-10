"""
Parameter Optimization Script
==============================
Use this AFTER your basic strategy is working to find optimal parameters.

WARNING: Be careful with optimization - it's easy to overfit!
- More parameters = higher risk of overfitting
- Always test optimized parameters on out-of-sample data
- Simpler is often better

This script tests combinations of:
- RSI period (10, 14, 21)
- Fast MA (20, 50, 100)
- Slow MA (100, 150, 200)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths
BACKTESTING_PATH = Path(r"C:\Users\estac\OneDrive - Syracuse University\trading\backtesting project #1\backtesting.py-master")
sys.path.insert(0, str(BACKTESTING_PATH))

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ============================================================================
# INDICATORS (same as main strategy)
# ============================================================================

def SMA(series, period):
    return pd.Series(series).rolling(window=period).mean()

def RSI(series, period=14):
    series = pd.Series(series)
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gains / avg_losses
    return 100 - (100 / (1 + rs))

# ============================================================================
# STRATEGY (with optimizable parameters)
# ============================================================================

class RSI_MA_Strategy(Strategy):
    # These are the parameters we'll optimize
    rsi_period = 14
    fast_ma = 50
    slow_ma = 200
    
    def init(self):
        close = self.data.Close
        self.rsi = self.I(RSI, close, self.rsi_period)
        self.sma_fast = self.I(SMA, close, self.fast_ma)
        self.sma_slow = self.I(SMA, close, self.slow_ma)
    
    def next(self):
        if pd.isna(self.rsi[-1]) or pd.isna(self.sma_slow[-1]):
            return
        
        rsi_now = self.rsi[-1]
        uptrend = self.sma_fast[-1] > self.sma_slow[-1]
        downtrend = self.sma_fast[-1] < self.sma_slow[-1]
        
        if not self.position:
            if rsi_now > 50 and uptrend:
                self.buy()
            elif rsi_now < 50 and downtrend:
                self.sell()
        elif self.position.is_long:
            if rsi_now < 30 or crossover(self.sma_slow, self.sma_fast):
                self.position.close()
        elif self.position.is_short:
            if rsi_now > 70 or crossover(self.sma_fast, self.sma_slow):
                self.position.close()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(timeframe='15T'):
    """Load and prepare EUR/JPY data"""
    DATA_PATH = Path(r"C:\Users\estac\OneDrive - Syracuse University\trading\backtesting project #1\eur-jpy Data")
    
    # Load 2024 data (full year for optimization)
    data_file = DATA_PATH / "2024" / "csv" / "DAT_MT_EURJPY_M1_2024.csv"
    
    if not data_file.exists():
        # Fall back to 2025 monthly files
        folder = DATA_PATH / "2025" / "csv"
        files = sorted(folder.glob("*.csv"))
        dfs = []
        for f in files[:6]:  # First 6 months
            df = pd.read_csv(f, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                           parse_dates={'Datetime': ['Date', 'Time']}, date_format='%Y.%m.%d %H:%M')
            df.set_index('Datetime', inplace=True)
            dfs.append(df)
        df_1m = pd.concat(dfs)
    else:
        df_1m = pd.read_csv(data_file, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                          parse_dates={'Datetime': ['Date', 'Time']}, date_format='%Y.%m.%d %H:%M')
        df_1m.set_index('Datetime', inplace=True)
    
    # Resample
    df = df_1m.resample(timeframe).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    return df

# ============================================================================
# MAIN OPTIMIZATION
# ============================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load_data('15T')
    print(f"Loaded {len(df):,} bars")
    
    # Setup backtest
    bt = Backtest(
        df,
        RSI_MA_Strategy,
        cash=10000,
        commission=0.00007,
        margin=1/30,
        exclusive_orders=True
    )
    
    # ===== OPTIMIZATION =====
    print("\n" + "="*60)
    print("RUNNING PARAMETER OPTIMIZATION")
    print("="*60)
    print("This may take a few minutes...\n")
    
    # Define parameter ranges to test
    # Keep ranges small to avoid overfitting
    stats, heatmap = bt.optimize(
        rsi_period=range(10, 22, 4),      # 10, 14, 18
        fast_ma=range(20, 80, 20),         # 20, 40, 60
        slow_ma=range(100, 220, 40),       # 100, 140, 180
        maximize='Sharpe Ratio',           # Optimize for risk-adjusted returns
        constraint=lambda p: p.fast_ma < p.slow_ma,  # Fast MA must be < Slow MA
        return_heatmap=True
    )
    
    # ===== RESULTS =====
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print("\nBest Parameters Found:")
    print(f"  RSI Period: {stats._strategy.rsi_period}")
    print(f"  Fast MA:    {stats._strategy.fast_ma}")
    print(f"  Slow MA:    {stats._strategy.slow_ma}")
    
    print("\nPerformance with Best Parameters:")
    print(f"  Return:       {stats['Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Win Rate:     {stats['Win Rate [%]']:.2f}%")
    print(f"  # Trades:     {stats['# Trades']}")
    
    # ===== HEATMAP VISUALIZATION =====
    print("\n" + "="*60)
    print("GENERATING HEATMAP")
    print("="*60)
    
    # The heatmap shows how Sharpe Ratio varies with parameters
    # This helps you understand which parameters are robust
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Reshape heatmap for visualization
        # Group by RSI period for multiple heatmaps
        for rsi_val in [10, 14, 18]:
            try:
                subset = heatmap.xs(rsi_val, level='rsi_period')
                if len(subset) > 0:
                    pivot = subset.unstack()
                    
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
                    plt.title(f'Sharpe Ratio Heatmap (RSI Period = {rsi_val})')
                    plt.xlabel('Slow MA')
                    plt.ylabel('Fast MA')
                    plt.tight_layout()
                    plt.savefig(f'heatmap_rsi_{rsi_val}.png', dpi=150)
                    print(f"Saved: heatmap_rsi_{rsi_val}.png")
            except:
                pass
                
    except ImportError:
        print("Install matplotlib and seaborn for heatmap visualization:")
        print("  pip install matplotlib seaborn")
    
    # ===== PLOT BEST RESULT =====
    print("\nGenerating plot with best parameters...")
    bt.plot(
        filename='optimized_backtest.html',
        open_browser=True
    )
    
    # ===== WARNINGS =====
    print("\n" + "="*60)
    print("⚠️  IMPORTANT WARNINGS ABOUT OPTIMIZATION")
    print("="*60)
    print("""
1. OVERFITTING RISK: These 'optimal' parameters are fit to historical data.
   They may not work as well in the future.

2. OUT-OF-SAMPLE TESTING: You should test these parameters on data that
   was NOT used in optimization (e.g., use 2024 for optimization, 
   test on 2025 data).

3. SIMPLICITY: Often, default parameters (RSI 14, MA 50/200) work 
   just as well and are more robust.

4. NUMBER OF TRADES: If the strategy only makes a few trades,
   the results may not be statistically significant.
    """)
