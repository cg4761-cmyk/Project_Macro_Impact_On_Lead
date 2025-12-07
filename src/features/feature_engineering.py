"""
Feature Engineering for LOPBDY and LMPBDS03 Data
Creates technical indicators and target variables
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_lopbdy_data(data_path: str = None) -> pd.DataFrame:
    """
    Load LOPBDY data from Excel file.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the LOPBDY Excel file. If None, uses default path.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with LOPBDY data
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data_raw" / "LOPBDY LME Comdty.xlsx"
    
    # Load Excel file
    df = pd.read_excel(data_path)
    
    # Delete first 6 rows
    if len(df) > 6:
        df = df.iloc[6:].reset_index(drop=True)
    
    # Assume first column is date and second is price, adjust as needed
    # Common column names: Date, Price, Close, etc.
    return df


def create_features(df: pd.DataFrame, price_col: str = None) -> pd.DataFrame:
    """
    Create features from LOPBDY price data.
    
    Features created:
    - MA7: 7-period moving average
    - MA30: 30-period moving average
    - Rolling volatility: rolling standard deviation of returns
    - EMA5, EMA15, EMA30: Exponential moving averages (5, 15, 30 periods)
    - MACD5, MACD15, MACD30: Moving Average Convergence Divergence indicators
    - RSI: Relative Strength Index (14-period)
    - BB_upper, BB_middle, BB_lower: Bollinger Bands (20-period, 2 std dev)
    
    Targets:
    - return_7d: (p7 - p1) / p1 (7-day forward return)
    - target: Binary classification (1 if return_7d > 0, else 0)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    price_col : str, optional
        Name of the price column. If None, will try to infer.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added features and target
    """
    df = df.copy()
    
    # Try to identify price column if not specified
    if price_col is None:
        # Common column names for price data
        possible_cols = ['Close', 'Price', 'LOPBDY', 'Value', 'Px_Last']
        price_col = None
        for col in possible_cols:
            if col in df.columns:
                price_col = col
                break
        
        # If still not found, use the second column (assuming first is date)
        if price_col is None and len(df.columns) >= 2:
            price_col = df.columns[1]
    
    if price_col is None or price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Ensure price column is numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    
    # Sort by date if date column exists
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
    date_col = None
    if date_cols:
        date_col = date_cols[0]
        df = df.sort_values(by=date_col)
        df = df.reset_index(drop=True)
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Set date as index for date-based operations
        df_indexed = df.set_index(date_col)
    else:
        df_indexed = df.copy()
    
    # Calculate returns (daily returns)
    df['returns'] = df[price_col].pct_change()
    
    # Feature 1: MA7 - 7-period moving average
    df['MA7'] = df[price_col].rolling(window=7, min_periods=1).mean()
    
    # Feature 2: MA30 - 30-period moving average
    df['MA30'] = df[price_col].rolling(window=30, min_periods=1).mean()
    
    # Feature 3: Rolling volatility (standard deviation of returns)
    # Common windows: 7, 30 days. Using 30 days for volatility
    df['rolling_volatility'] = df['returns'].rolling(window=30, min_periods=1).std()
    
    # Feature 4-6: EMA5, EMA15, EMA30 - Exponential Moving Averages
    df['EMA5'] = df[price_col].ewm(span=5, adjust=False, min_periods=1).mean()
    df['EMA15'] = df[price_col].ewm(span=15, adjust=False, min_periods=1).mean()
    df['EMA30'] = df[price_col].ewm(span=30, adjust=False, min_periods=1).mean()
    
    # Feature 7-9: MACD5, MACD15, MACD30 - Moving Average Convergence Divergence
    # MACD = Fast EMA - Slow EMA, Signal = EMA of MACD
    # MACD5: fast=5, slow=26, signal=9
    ema5_fast = df[price_col].ewm(span=5, adjust=False, min_periods=1).mean()
    ema26_slow = df[price_col].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD5'] = ema5_fast - ema26_slow
    
    # MACD15: fast=15, slow=26, signal=9
    ema15_fast = df[price_col].ewm(span=15, adjust=False, min_periods=1).mean()
    df['MACD15'] = ema15_fast - ema26_slow
    
    # MACD30: fast=30, slow=26, signal=9
    ema30_fast = df[price_col].ewm(span=30, adjust=False, min_periods=1).mean()
    df['MACD30'] = ema30_fast - ema26_slow
    
    # Feature 10: RSI - Relative Strength Index (typically 14 periods)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df[price_col], period=14)
    
    # Feature 11-13: Bollinger Bands (typically 20 periods, 2 std dev)
    bb_period = 20
    bb_std = 2
    bb_middle = df[price_col].rolling(window=bb_period, min_periods=1).mean()
    bb_std_dev = df[price_col].rolling(window=bb_period, min_periods=1).std()
    df['BB_upper'] = bb_middle + (bb_std_dev * bb_std)
    df['BB_middle'] = bb_middle
    df['BB_lower'] = bb_middle - (bb_std_dev * bb_std)
    
    # Target: return after 7 days - (p7 - p1) / p1
    # Calculate return from current date to 7 days later
    if date_col is not None:
        # Create a mapping: for each date, find price 7 days later
        df_indexed = df.set_index(date_col)
        # For each date, get the price 7 days later
        price_7d_later = []
        for date in df[date_col]:
            date_7d_later = date + pd.Timedelta(days=6)
            if date_7d_later in df_indexed.index:
                price_7d_later.append(df_indexed.loc[date_7d_later, price_col])
            else:
                price_7d_later.append(np.nan)
        df['return_7d'] = (np.array(price_7d_later) - df[price_col]) / df[price_col]
    else:
        # Fallback to row-based shift if no date column
        df['price_7d_forward'] = df[price_col].shift(-7)
        df['return_7d'] = (df['price_7d_forward'] - df[price_col]) / df[price_col]
        df = df.drop(['price_7d_forward'], axis=1)
    
    # Binary target: 1 if return_7d > 0, else 0
    df['target'] = (df['return_7d'] > 0).astype(int)
    
    # Delete last 10 rows
    if len(df) > 10:
        df = df.iloc[:-10].reset_index(drop=True)

    if len(df) > 30:
        df = df.iloc[29:].reset_index(drop=True)
    
    return df


def load_lmpbds03_data(data_path: str = None) -> pd.DataFrame:
    """
    Load LMPBDS03 (future) data from Excel file.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the LMPBDS03 Excel file. If None, uses default path.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with LMPBDS03 data
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data_raw" / "LMPBDS03 LME Comdty.xlsx"
    
    # Load Excel file
    df = pd.read_excel(data_path)
    
    # Delete first 6 rows (similar to LOPBDY)
    if len(df) > 6:
        df = df.iloc[6:].reset_index(drop=True)
    
    return df


def create_future_features(df_future: pd.DataFrame, df_spot: pd.DataFrame = None,
                          future_price_col: str = None, spot_price_col: str = None,
                          date_col: str = None) -> pd.DataFrame:
    """
    Create features from LMPBDS03 (future) price data.
    
    Features created:
    - spread: future - spot
    - future_return: daily return of future price
    - future_MA7: 7-period moving average of future price
    - future_MA30: 30-period moving average of future price
    - future_rolling_volatility: rolling standard deviation of future returns
    - future_return_7d: 7-day forward return of future price
    
    Parameters:
    -----------
    df_future : pd.DataFrame
        DataFrame with future price data
    df_spot : pd.DataFrame, optional
        DataFrame with spot price data (LOPBDY) to calculate spread
    future_price_col : str, optional
        Name of the future price column. If None, will try to infer.
    spot_price_col : str, optional
        Name of the spot price column. If None, will try to infer.
    date_col : str, optional
        Name of the date column. If None, will try to infer.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with future features
    """
    df = df_future.copy()
    
    # Try to identify columns
    if date_col is None:
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
        if date_cols:
            date_col = date_cols[0]
    
    if future_price_col is None:
        possible_cols = ['Close', 'Price', 'LMPBDS03', 'Value', 'Px_Last']
        future_price_col = None
        for col in possible_cols:
            if col in df.columns:
                future_price_col = col
                break
        
        if future_price_col is None and len(df.columns) >= 2:
            # Skip date column if it's the first
            future_price_col = df.columns[1] if date_col != df.columns[0] else df.columns[1]
    
    if future_price_col is None or future_price_col not in df.columns:
        raise ValueError(f"Future price column '{future_price_col}' not found. Available columns: {df.columns.tolist()}")
    
    # Ensure price column is numeric
    df[future_price_col] = pd.to_numeric(df[future_price_col], errors='coerce')
    
    # Sort by date if date column exists
    if date_col and date_col in df.columns:
        df = df.sort_values(by=date_col)
        df = df.reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df_indexed = df.set_index(date_col)
    else:
        df_indexed = df.copy()
        date_col = None
    
    # Calculate spread (future - spot) if spot data provided
    if df_spot is not None:
        # Find spot price column if not provided
        spot_date_cols = [col for col in df_spot.columns if 'date' in col.lower() or 'Date' in col]
        
        if spot_price_col is None:
            spot_possible_cols = ['Close', 'Price', 'LOPBDY', 'Value', 'Px_Last']
            for col in spot_possible_cols:
                if col in df_spot.columns:
                    spot_price_col = col
                    break
            
            if spot_price_col is None and len(df_spot.columns) >= 2:
                spot_price_col = df_spot.columns[1] if (not spot_date_cols or spot_date_cols[0] != df_spot.columns[0]) else df_spot.columns[1]
        
        # Check if spot_price_col is valid
        if spot_price_col and spot_price_col in df_spot.columns:
            # Get spot date column
            spot_date_col = spot_date_cols[0] if spot_date_cols else None
            
            if spot_date_col:
                df_spot[spot_date_col] = pd.to_datetime(df_spot[spot_date_col], errors='coerce')
                df_spot[spot_price_col] = pd.to_numeric(df_spot[spot_price_col], errors='coerce')
                
                # Merge spot prices by date
                if date_col:
                    df_spot_indexed = df_spot.set_index(spot_date_col)
                    spot_prices = []
                    for date in df[date_col]:
                        if date in df_spot_indexed.index:
                            spot_prices.append(df_spot_indexed.loc[date, spot_price_col])
                        else:
                            spot_prices.append(np.nan)
                    df['spot_price'] = spot_prices
                    df['spread'] = df[future_price_col] - df['spot_price']
    
    # Calculate future returns (daily returns)
    df['future_return'] = df[future_price_col].pct_change()
    
    # Feature 1: future_MA7 - 7-period moving average
    df['future_MA7'] = df[future_price_col].rolling(window=7, min_periods=1).mean()
    
    # Feature 2: future_MA30 - 30-period moving average
    df['future_MA30'] = df[future_price_col].rolling(window=30, min_periods=1).mean()
    
    # Feature 3: future_rolling_volatility (standard deviation of returns)
    df['future_rolling_volatility'] = df['future_return'].rolling(window=30, min_periods=1).std()
    
    # Feature 4: future_return_7d - 7-day forward return
    if date_col:
        price_7d_later = []
        for date in df[date_col]:
            date_7d_later = date + pd.Timedelta(days=7)
            if date_7d_later in df_indexed.index:
                price_7d_later.append(df_indexed.loc[date_7d_later, future_price_col])
            else:
                price_7d_later.append(np.nan)
        df['future_return_7d'] = (np.array(price_7d_later) - df[future_price_col]) / df[future_price_col]
    else:
        df['price_7d_forward'] = df[future_price_col].shift(-7)
        df['future_return_7d'] = (df['price_7d_forward'] - df[future_price_col]) / df[future_price_col]
        df = df.drop(['price_7d_forward'], axis=1)
    
    # Shift future_return_7d down by 7 rows to align with actual dates
    df['future_return_7d'] = df['future_return_7d'].shift(7)
    
    # Select only relevant columns for output
    output_cols = []
    if date_col:
        output_cols.append(date_col)
    if 'spread' in df.columns:
        output_cols.extend(['spread', 'spot_price'])
    output_cols.extend(['future_return', 'future_MA7', 'future_MA30', 
                       'future_rolling_volatility', 'future_return_7d'])

    # Delete last 10 rows
    if len(df) > 10:
        df = df.iloc[:-10].reset_index(drop=True)
    
    return df[output_cols]


def merge_features(lopbdy_features: pd.DataFrame, lmpbds03_features: pd.DataFrame,
                   date_col: str = None) -> pd.DataFrame:
    """
    Merge LOPBDY features with LMPBDS03 features by date.
    
    Parameters:
    -----------
    lopbdy_features : pd.DataFrame
        DataFrame with LOPBDY features
    lmpbds03_features : pd.DataFrame
        DataFrame with LMPBDS03 features
    date_col : str, optional
        Name of the date column. If None, will try to infer.
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with all features aligned by date
    """
    # Find date columns
    if date_col is None:
        lopbdy_date_cols = [col for col in lopbdy_features.columns if 'date' in col.lower() or 'Date' in col]
        lmpbds03_date_cols = [col for col in lmpbds03_features.columns if 'date' in col.lower() or 'Date' in col]
        
        if lopbdy_date_cols and lmpbds03_date_cols:
            date_col = lopbdy_date_cols[0]
            # Rename if different
            if lmpbds03_date_cols[0] != date_col:
                lmpbds03_features = lmpbds03_features.rename(columns={lmpbds03_date_cols[0]: date_col})
        elif lopbdy_date_cols:
            date_col = lopbdy_date_cols[0]
        elif lmpbds03_date_cols:
            date_col = lmpbds03_date_cols[0]
    
    if date_col is None:
        # If no date column, merge on index
        return pd.concat([lopbdy_features, lmpbds03_features], axis=1)
    
    # Ensure date columns are datetime
    lopbdy_features[date_col] = pd.to_datetime(lopbdy_features[date_col], errors='coerce')
    lmpbds03_features[date_col] = pd.to_datetime(lmpbds03_features[date_col], errors='coerce')
    
    # Merge on date column
    merged = pd.merge(lopbdy_features, lmpbds03_features, on=date_col, how='outer', suffixes=('', '_future'))
    
    # Sort by date
    merged = merged.sort_values(by=date_col).reset_index(drop=True)
    
    return merged


def process_lmpbds03_features(data_path: str = None, spot_data: pd.DataFrame = None,
                              future_price_col: str = None, spot_price_col: str = None,
                              date_col: str = None, save_path: str = None) -> pd.DataFrame:
    """
    Complete pipeline: load LMPBDS03 data and create features.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the LMPBDS03 Excel file
    spot_data : pd.DataFrame, optional
        Spot price data (LOPBDY) for calculating spread
    future_price_col : str, optional
        Name of the future price column
    spot_price_col : str, optional
        Name of the spot price column
    date_col : str, optional
        Name of the date column
    save_path : str, optional
        Path to save the processed data. If None, doesn't save.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with future features
    """
    # Load data
    df = load_lmpbds03_data(data_path)
    
    # Create features
    df_features = create_future_features(df, spot_data, future_price_col, 
                                        spot_price_col, date_col)
    
    # Save if path provided
    if save_path:
        df_features.to_csv(save_path, index=False)
        print(f"Future features saved to {save_path}")
    
    return df_features


def process_lopbdy_features(data_path: str = None, price_col: str = None, 
                            save_path: str = None) -> pd.DataFrame:
    """
    Complete pipeline: load data and create features.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the LOPBDY Excel file
    price_col : str, optional
        Name of the price column
    save_path : str, optional
        Path to save the processed data. If None, doesn't save.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and target
    """
    # Load data
    df = load_lopbdy_data(data_path)
    
    # Create features
    df_features = create_features(df, price_col)
    
    # Delete column 3 "Unnamed" if it exists
    # Check column 3 (0-indexed, so index 3 is the 4th column)
    if len(df_features.columns) > 3:
        col_name = df_features.columns[3]
        if 'Unnamed' in str(col_name) or str(col_name).startswith('Unnamed'):
            df_features = df_features.drop(columns=[col_name])
            print(f"Dropped column 3: {col_name}")
    
    # Also check for any other "Unnamed" columns
    unnamed_cols = [col for col in df_features.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df_features = df_features.drop(columns=unnamed_cols)
        print(f"Dropped additional Unnamed columns: {unnamed_cols}")
    
    # Save if path provided
    if save_path:
        df_features.to_csv(save_path, index=False)
        print(f"Features saved to {save_path}")
    
    return df_features


def process_and_merge_all_features(lopbdy_path: str = None, lmpbds03_path: str = None,
                                   save_path: str = None) -> pd.DataFrame:
    """
    Complete pipeline: process both LOPBDY and LMPBDS03, create features, and merge.
    
    Parameters:
    -----------
    lopbdy_path : str, optional
        Path to the LOPBDY Excel file
    lmpbds03_path : str, optional
        Path to the LMPBDS03 Excel file
    save_path : str, optional
        Path to save the merged features. If None, doesn't save.
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with all features from both datasets
    """
    # Process LOPBDY features
    print("Processing LOPBDY features...")
    lopbdy_features = process_lopbdy_features(data_path=lopbdy_path)
    
    # Load LOPBDY raw data for spot prices (for spread calculation)
    lopbdy_raw = load_lopbdy_data(lopbdy_path)
    
    # Process LMPBDS03 features
    print("Processing LMPBDS03 features...")
    lmpbds03_features = process_lmpbds03_features(
        data_path=lmpbds03_path,
        spot_data=lopbdy_raw
    )
    
    # Merge features by date
    print("Merging features...")
    merged_features = merge_features(lopbdy_features, lmpbds03_features)
    
    
    # Delete column 3 "Unnamed" if it exists
    # Check column 3 (0-indexed, so index 3 is the 4th column)
    if len(merged_features.columns) > 3:
        col_name = merged_features.columns[3]
        if 'Unnamed' in str(col_name) or str(col_name).startswith('Unnamed'):
            merged_features = merged_features.drop(columns=[col_name])
            print(f"Dropped column 3: {col_name}")
    
    # Also check for any other "Unnamed" columns
    unnamed_cols = [col for col in merged_features.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        merged_features = merged_features.drop(columns=unnamed_cols)
        print(f"Dropped additional Unnamed columns: {unnamed_cols}")
    
    # Save if path provided
    if save_path:
        merged_features.to_csv(save_path, index=False)
        print(f"All features saved to {save_path}")
    
    return merged_features


if __name__ == "__main__":
    # Process and save LOPBDY features
    print("="*60)
    print("PROCESSING LOPBDY FEATURES")
    print("="*60)
    df_lopbdy = process_lopbdy_features(
        save_path="data_processed/lopbdy_features.csv"
    )
    
    print("\n" + "="*60)
    print("LOPBDY FEATURES SUMMARY")
    print("="*60)
    print(f"\nTotal rows: {len(df_lopbdy)}")
    print(f"\nColumns ({len(df_lopbdy.columns)} total): {df_lopbdy.columns.tolist()}")
    
    # Show LOPBDY features
    lopbdy_cols = ['MA7', 'MA30', 'rolling_volatility', 'EMA5', 'EMA15', 'EMA30',
                   'MACD5', 'MACD15', 'MACD30', 'RSI', 'BB_upper', 'BB_middle',
                   'BB_lower', 'return_7d', 'target']
    existing_lopbdy_cols = [col for col in lopbdy_cols if col in df_lopbdy.columns]
    if existing_lopbdy_cols:
        print("\nLOPBDY Feature Statistics:")
        print(df_lopbdy[existing_lopbdy_cols].describe())
    
    # Show target distribution
    if 'target' in df_lopbdy.columns:
        print("\nTarget Distribution:")
        print(df_lopbdy['target'].value_counts())
        print(f"\nTarget percentage: {df_lopbdy['target'].mean()*100:.2f}% positive (1)")
    
    print("\n" + "="*60)
    print("PROCESSING AND MERGING ALL FEATURES")
    print("="*60)
    # Process and merge all features
    df_all = process_and_merge_all_features(
        save_path="data_processed/all_features.csv"
    )
    
    print("\n" + "="*60)
    print("MERGED FEATURES SUMMARY")
    print("="*60)
    print(f"\nTotal rows: {len(df_all)}")
    print(f"\nColumns: {df_all.columns.tolist()}")
    
    # Show LOPBDY features in merged
    lopbdy_cols_merged = [col for col in existing_lopbdy_cols if col in df_all.columns]
    if lopbdy_cols_merged:
        print("\nLOPBDY Features in merged dataset:")
        print(df_all[lopbdy_cols_merged].describe())
    
    # Show LMPBDS03 features
    future_cols = ['spread', 'future_return', 'future_MA7', 'future_MA30', 
                   'future_rolling_volatility', 'future_return_7d']
    future_cols = [col for col in future_cols if col in df_all.columns]
    if future_cols:
        print("\nLMPBDS03 Features:")
        print(df_all[future_cols].describe())
    
    print("\nFirst 10 rows of merged features:")
    print(df_all.head(10))

