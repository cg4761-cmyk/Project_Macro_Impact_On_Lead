"""
Merge all_features.csv with macro_features.csv
Renames first column of all_features.csv to "Date", then merges and drops all NA rows
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def merge_all_features(all_features_path: Optional[str] = None,
                      macro_features_path: Optional[str] = None,
                      save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Merge all_features.csv with macro_features.csv by date and drop all NA rows.
    Renames the first column of all_features.csv to "Date" before merging.
    
    Parameters:
    -----------
    all_features_path : str, optional
        Path to the all_features.csv file. If None, uses default path.
    macro_features_path : str, optional
        Path to the macro_features.csv file. If None, uses default path.
    save_path : str, optional
        Path to save the merged features. If None, doesn't save.
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with all features, with all NA rows dropped
    """
    # Set default paths
    if all_features_path is None:
        project_root = Path(__file__).parent.parent.parent
        all_features_path = project_root / "data_processed" / "all_features.csv"
    
    if macro_features_path is None:
        project_root = Path(__file__).parent.parent.parent
        macro_features_path = project_root / "data_processed" / "macro_features.csv"
    
    # Load datasets
    print(f"Loading all_features.csv from {all_features_path}")
    all_features_df = pd.read_csv(all_features_path)
    
    print(f"Loading macro_features.csv from {macro_features_path}")
    macro_features_df = pd.read_csv(macro_features_path)
    
    # Rename first column of all_features.csv to "Date"
    first_col = all_features_df.columns[0]
    if first_col != 'Date':
        print(f"Renaming first column '{first_col}' to 'Date' in all_features.csv")
        all_features_df = all_features_df.rename(columns={first_col: 'Date'})
    
    # Find date column in macro_features.csv
    date_col = 'Date'
    macro_date_cols = [col for col in macro_features_df.columns if 'date' in col.lower() or 'Date' in col]
    if macro_date_cols:
        macro_date_col = macro_date_cols[0]
        if macro_date_col != date_col:
            macro_features_df = macro_features_df.rename(columns={macro_date_col: date_col})
    else:
        # If no date column found, assume first column is date
        first_col_macro = macro_features_df.columns[0]
        if first_col_macro != date_col:
            macro_features_df = macro_features_df.rename(columns={first_col_macro: date_col})
    
    # Convert date columns to datetime
    all_features_df[date_col] = pd.to_datetime(all_features_df[date_col], errors='coerce')
    macro_features_df[date_col] = pd.to_datetime(macro_features_df[date_col], errors='coerce')
    
    # Merge by date using inner join to ensure both datasets have data
    print("Merging datasets by date...")
    merged_df = pd.merge(
        all_features_df, 
        macro_features_df, 
        on=date_col, 
        how='inner',
        suffixes=('', '_macro')
    )
    
    # Remove duplicate date columns if any (from suffixes)
    cols_to_drop = [col for col in merged_df.columns if col.endswith('_macro') and col != date_col]
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)
    
    # Sort by date
    merged_df = merged_df.sort_values(by=date_col).reset_index(drop=True)
    
    print(f"Before dropping NA: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Drop all rows with any NA values
    merged_df = merged_df.dropna()
    
    print(f"After dropping NA: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Save if path provided
    if save_path:
        merged_df.to_csv(save_path, index=False)
        print(f"Merged features saved to {save_path}")
    
    return merged_df


if __name__ == "__main__":
    # Merge all features
    merged_features_df = merge_all_features(
        save_path="data_processed/all_features_with_macro.csv"
    )
    
    print(f"\nFinal dataset shape: {merged_features_df.shape}")
    print(f"Columns: {len(merged_features_df.columns)}")
    print(f"\nFirst few column names: {merged_features_df.columns.tolist()[:10]}")

