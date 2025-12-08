"""
Macroeconomic Feature Engineering
Creates features from all_raw_data.csv for lead price prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_all_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load all_raw_data.csv and flip it from recent-to-past to past-to-recent.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the all_raw_data.csv file. If None, uses default path.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with data in chronological order (past to recent)
    """
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data_raw" / "all_raw_data.csv"
    
    df = pd.read_csv(data_path)
    
    # Flip dataset to make dates from past to recent (ascending order)
    # Original data is from recent to past, so reverse it
    df = df.iloc[::-1].reset_index(drop=True)
    
    return df


def create_macro_features(data_path: Optional[str] = None,
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create macroeconomic features from all_raw_data.csv.
    
    Features created:
    - dxy_return_7d: 7-day % change of DXY
    - yield_change_7d: 7-day difference of 10-yr Treasury yield
    - dxy_ma_diff_20_5: EMA20(DXY) - EMA5(DXY)
    - copper_return_7d: 7-day % change of Copper
    - wti_return_7d: 7-day % change of WTI
    - lead_to_copper_ratio: lead_close / copper_close
    - copper_ma_diff_20_5: EMA20(Copper) - EMA5(Copper)
    - copper_vol_7d: rolling std of 7-day copper returns
    - wti_vol_7d: rolling std of 7-day WTI returns
    - corr_lead_copper_30d: 30-day correlation between lead and copper returns
    - corr_lead_dxy_30d: 30-day correlation between lead and DXY returns
    - IDF: 0.5 * copper_return_7d + 0.5 * wti_return_7d
    - divergence_lc: lead_return_7d - copper_return_7d
    - oil_dxy_interaction: wti_return_7d * dxy_return_7d
    - joint_momentum: copper_return_7d + lead_return_7d
    - regime_dxy_yield: sign(dxy_return_7d) * sign(yield_change_7d)
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the all_raw_data.csv file
    save_path : str, optional
        Path to save the processed features
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all macroeconomic features
    """
    # Load and flip data
    df = load_all_raw_data(data_path)
    
    # Use exact column names as provided
    date_col = 'Date'
    lead_close = 'LOPBDY_Price'
    copper_close = 'copper_price_close'
    dxy_close = 'usd_index_close'
    wti_close = 'wti_crude_oil_close'
    yield_col = 'us_10yr_yield_close'
    
    # Verify columns exist
    required_cols = [date_col, lead_close, copper_close, dxy_close, wti_close, yield_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
    
    # Convert date to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Create features dictionary
    features = {date_col: df[date_col]}
    
    # 1. dxy_return_7d: 7-day % change of DXY
    dxy_pct_change = df[dxy_close].pct_change(periods=7)
    features['dxy_return_7d'] = dxy_pct_change
    
    # 2. yield_change_7d: 7-day difference of 10-yr Treasury yield
    yield_diff = df[yield_col].diff(periods=7)
    features['yield_change_7d'] = yield_diff
    
    # 3. dxy_ma_diff_20_5: EMA20(DXY) - EMA5(DXY)
    dxy_ema20 = df[dxy_close].ewm(span=20, adjust=False, min_periods=1).mean()
    dxy_ema5 = df[dxy_close].ewm(span=5, adjust=False, min_periods=1).mean()
    features['dxy_ma_diff_20_5'] = dxy_ema20 - dxy_ema5
    
    # 4. copper_return_7d: 7-day % change of Copper
    copper_pct_change = df[copper_close].pct_change(periods=7)
    features['copper_return_7d'] = copper_pct_change
    
    # 5. wti_return_7d: 7-day % change of WTI
    wti_pct_change = df[wti_close].pct_change(periods=7)
    features['wti_return_7d'] = wti_pct_change
    
    # 6. lead_to_copper_ratio: lead_close / copper_close
    ratio = df[lead_close] / df[copper_close]
    features['lead_to_copper_ratio'] = ratio
    
    # 7. copper_ma_diff_20_5: EMA20(Copper) - EMA5(Copper)
    copper_ema20 = df[copper_close].ewm(span=20, adjust=False, min_periods=1).mean()
    copper_ema5 = df[copper_close].ewm(span=5, adjust=False, min_periods=1).mean()
    features['copper_ma_diff_20_5'] = copper_ema20 - copper_ema5
    
    # 8. copper_vol_7d: rolling std of 7-day copper returns
    copper_daily_returns = df[copper_close].pct_change()
    copper_vol = copper_daily_returns.rolling(window=7, min_periods=1).std()
    features['copper_vol_7d'] = copper_vol
    
    # 9. wti_vol_7d: rolling std of 7-day WTI returns
    wti_daily_returns = df[wti_close].pct_change()
    wti_vol = wti_daily_returns.rolling(window=7, min_periods=1).std()
    features['wti_vol_7d'] = wti_vol
    
    # 10. corr_lead_copper_30d: 30-day correlation between lead and copper returns
    lead_returns = df[lead_close].pct_change()
    copper_returns = df[copper_close].pct_change()
    # Calculate rolling correlation manually
    returns_df = pd.DataFrame({'lead': lead_returns, 'copper': copper_returns})
    corr_values = []
    for i in range(len(returns_df)):
        if i < 29:
            corr_values.append(np.nan)
        else:
            window_lead = returns_df['lead'].iloc[i-29:i+1]
            window_copper = returns_df['copper'].iloc[i-29:i+1]
            corr = window_lead.corr(window_copper)
            corr_values.append(corr)
    features['corr_lead_copper_30d'] = pd.Series(corr_values, index=df.index)
    
    # 11. corr_lead_dxy_30d: 30-day correlation between lead and DXY returns
    dxy_returns = df[dxy_close].pct_change()
    # Calculate rolling correlation manually
    returns_df = pd.DataFrame({'lead': lead_returns, 'dxy': dxy_returns})
    corr_values = []
    for i in range(len(returns_df)):
        if i < 29:
            corr_values.append(np.nan)
        else:
            window_lead = returns_df['lead'].iloc[i-29:i+1]
            window_dxy = returns_df['dxy'].iloc[i-29:i+1]
            corr = window_lead.corr(window_dxy)
            corr_values.append(corr)
    features['corr_lead_dxy_30d'] = pd.Series(corr_values, index=df.index)
    
    # 12. IDF: 0.5 * copper_return_7d + 0.5 * wti_return_7d
    idf = 0.5 * features['copper_return_7d'] + 0.5 * features['wti_return_7d']
    features['IDF'] = idf
    
    # 13. divergence_lc: lead_return_7d - copper_return_7d
    lead_return_7d = df[lead_close].pct_change(periods=7)
    features['divergence_lc'] = lead_return_7d - features['copper_return_7d']
    
    # 14. oil_dxy_interaction: wti_return_7d * dxy_return_7d
    features['oil_dxy_interaction'] = features['wti_return_7d'] * features['dxy_return_7d']
    
    # 15. joint_momentum: copper_return_7d + lead_return_7d
    features['joint_momentum'] = features['copper_return_7d'] + lead_return_7d
    
    # 16. regime_dxy_yield: sign(dxy_return_7d) * sign(yield_change_7d)
    dxy_sign = np.sign(features['dxy_return_7d'])
    yield_sign = np.sign(features['yield_change_7d'])
    features['regime_dxy_yield'] = dxy_sign * yield_sign
    
    # Create DataFrame from features
    macro_features_df = pd.DataFrame(features)
    
    # Drop rows with insufficient rolling windows (rows with NaN from rolling operations)
    # Keep rows where at least some features are available
    # Drop rows where all feature columns (except date) are NaN
    feature_cols = [col for col in macro_features_df.columns if col != date_col]
    macro_features_df = macro_features_df.dropna(subset=feature_cols, how='all')
    
    # For rolling correlations, we need at least 30 days, so drop first 29 rows
    # Also drop rows with NaN in key features that require sufficient history
    if len(macro_features_df) > 30:
        # Keep rows starting from index 30 to ensure rolling windows are complete
        macro_features_df = macro_features_df.iloc[30:].reset_index(drop=True)
    
    # Final dropna to remove any remaining NaN values
    macro_features_df = macro_features_df.dropna()
    
    # Save if path provided
    if save_path:
        macro_features_df.to_csv(save_path, index=False)
    
    return macro_features_df


if __name__ == "__main__":
    # Create macroeconomic features
    macro_features_df = create_macro_features(
        save_path="data_processed/macro_features.csv"
    )

