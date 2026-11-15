# src/loader.py
import pandas as pd
from pathlib import Path

def load_raw_data(file_path: str = "data/raw/dld_sales_transactions.parquet") -> pd.DataFrame:
    """
    Loads the raw Dubai Land Department (DLD) sales transactions dataset.
    
    Args:
        file_path (str): Relative path to the Parquet file from the project root.
        
    Returns:
        pd.DataFrame: The raw, uncleaned dataframe.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"❌ Data not found at: {path.absolute()}\n"
            "Please download 'dld_sales_transactions.parquet' from Hugging Face "
            "and place it in the data/raw/ folder."
        )
    
    print(f"⏳ Loading raw data from {path}...")
    
    # Load the parquet file (lightning fast due to columnar storage)
    df = pd.read_parquet(path)
    
    print(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns.")
    
    # Quick sanity check based on the data dictionary
    if 'instance_date' in df.columns:
        print(f"📅 Date range: {df['instance_date'].min()} to {df['instance_date'].max()}")
        
    return df