"""
Example script to generate features from LOPBDY and LMPBDS03 data
"""

from src.features import process_and_merge_all_features, process_lopbdy_features

if __name__ == "__main__":
    # Option 1: Process and merge all features (recommended)
    print("=" * 60)
    print("PROCESSING AND MERGING ALL FEATURES")
    print("=" * 60)
    
    df_all = process_and_merge_all_features(
        lopbdy_path=None,  # Uses default: data_raw/LOPBDY LME Comdty.xlsx
        lmpbds03_path=None,  # Uses default: data_raw/LMPBDS03 LME Comdty.xlsx
        save_path="data_processed/all_features.csv"
    )
    
    print("\n" + "=" * 60)
    print("MERGED FEATURES SUMMARY")
    print("=" * 60)
    print(f"\nTotal rows: {len(df_all)}")
    print(f"\nColumns: {df_all.columns.tolist()}")
    
    # Show LOPBDY features
    lopbdy_cols = [col for col in ['MA7', 'MA30', 'rolling_volatility', 'return_7d'] 
                   if col in df_all.columns]
    if lopbdy_cols:
        print("\nLOPBDY Features:")
        print(df_all[lopbdy_cols].describe())
    
    # Show LMPBDS03 features
    future_cols = [col for col in ['spread', 'future_return', 'future_MA7', 'future_MA30', 
                   'future_rolling_volatility', 'future_return_7d'] 
                   if col in df_all.columns]
    if future_cols:
        print("\nLMPBDS03 Features:")
        print(df_all[future_cols].describe())
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 10 rows)")
    print("=" * 60)
    display_cols = lopbdy_cols + future_cols
    if display_cols:
        print(df_all[display_cols].head(10))
    
    # Option 2: Process only LOPBDY features (for comparison)
    print("\n" + "=" * 60)
    print("OPTION 2: LOPBDY FEATURES ONLY")
    print("=" * 60)
    
    df_lopbdy = process_lopbdy_features(
        data_path=None,
        price_col=None,
        save_path="data_processed/lopbdy_features.csv"
    )
    
    print(f"\nTotal rows: {len(df_lopbdy)}")
    print(f"\nColumns: {df_lopbdy.columns.tolist()}")
    print("\nFeature statistics:")
    print(df_lopbdy[['MA7', 'MA30', 'rolling_volatility', 'return_7d']].describe())

