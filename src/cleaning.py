# src/cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DLD transactions data based on the official DATA_DICTIONARY.md rules.
    Engineered specifically to support RQ1-RQ4 and the ML Mispricing model.
    """
    initial_rows = len(df)
    print(f"🧹 Starting cleaning pipeline. Rows before: {initial_rows:,}")
    
    # 1. Quality Flags: Keep only rows that pass EVERY check (flag == 0)
    # This automatically handles extreme outliers (P1-P99 band) and bad dates.
    df = df[df['quality_flags'] == 0].copy()
    print(f"   - After quality_flags filter: {len(df):,} rows")
    
    # 2. Property Type & Usage: Keep only built residential (Unit & Villa)
    # Excludes Land (plot area distorts psf) and whole buildings (wholesale).
    df = df[df['property_type_label'].isin(['Unit', 'Villa'])].copy()
    df = df[df['usage_label'] == 'Residential'].copy()
    print(f"   - After Property Type & Usage filter: {len(df):,} rows")

    # 3. Clean Free-Text Strings (CRITICAL FOR RQ1)
    # The dictionary warns of spelling/casing variations in building names.
    # " burj khalifa " and "Burj Khalifa" must be merged.
    text_cols = ['area_name_en', 'building_name', 'project_name']
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    
    # Handle missing building names gracefully
    df['building_name'] = df['building_name'].replace('Nan', 'Unknown Building')

    # 4. Calculate Price per Square Meter (CRITICAL FOR ALL RQs)
    # The dataset gives price_psf (square feet). Our RQs are based on square meters.
    # 1 sqm = 10.7639 sqft. We calculate this explicitly to avoid confusion.
    df['price_per_sqm'] = df['price_aed'] / df['area_sqm']
    
    # Safety check: remove any zero/negative areas that slipped through
    df = df[(df['area_sqm'] > 0) & (df['price_aed'] > 0)]

    # 5. Time Features (FOR RQ2 & RQ4)
    df['instance_date'] = pd.to_datetime(df['instance_date'])
    df['year'] = df['instance_date'].dt.year
    df['month'] = df['instance_date'].dt.month
    df['quarter'] = df['instance_date'].dt.quarter
    df['year_quarter'] = df['year'].astype(str) + "-Q" + df['quarter'].astype(str)

    # 6. Handle Missing Values
    # We DROP rows missing core metrics we need for charts.
    critical_cols = ['price_aed', 'area_sqm', 'price_per_sqm', 'area_name_en']
    df = df.dropna(subset=critical_cols)

    # We DO NOT drop missing rooms_bucket (bedrooms) or has_parking.
    # The dictionary says older rows lack bedroom counts. Dropping them 
    # would erase early 2000s data. We keep them as NaN and handle them 
    # later in RQ3 (using medians) or ML (using imputation).
    df['rooms_bucket'] = pd.to_numeric(df['rooms_bucket'], errors='coerce')
    df['has_parking'] = df['has_parking'].astype(bool)

    print(f"✅ Cleaning complete. Final rows: {len(df):,} (Removed {initial_rows - len(df):,} rows)")
    print(f"   - Date range: {df['instance_date'].min().date()} to {df['instance_date'].max().date()}")
    
    return df

if __name__ == "__main__":
    import sys
    sys.path.append('../') 
    from loader import load_raw_data
    
    print("--- Testing Loader and Cleaner ---")
    raw_df = load_raw_data()
    clean_df = clean_transactions(raw_df)
    
    # Save the processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clean_transactions.parquet"
    
    clean_df.to_parquet(output_path, index=False)
    print(f"💾 Saved cleaned data to {output_path}")