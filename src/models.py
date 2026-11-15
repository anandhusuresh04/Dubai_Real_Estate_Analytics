# src/models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = ROOT_DIR / 'visualizations'
VIS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_find_undervalued(df: pd.DataFrame):
    """
    Trains a Random Forest to predict fair price_per_sqm, then finds undervalued deals.
    OPTIMIZED FOR SPEED: Trains on a 150k random sample.
    APPLIES SANITY FILTERS: Removes parking spaces and data entry errors.
    """
    print("🤖 Initializing Machine Learning Valuation Engine...")
    
    target = 'price_per_sqm'
    features = [
        'area_name_en', 'property_type_label', 'rooms_bucket', 
        'has_parking', 'reg_type_label', 'year'
    ]
    
    # 🚨 THE ULTIMATE SANITY FILTER
    ml_df = df.dropna(subset=[target]).copy()
    
    # 1. Must be a real home (not a parking spot/storage unit)
    ml_df = ml_df[ml_df['area_sqm'] >= 30]
    
    # 2. Must have a realistic total price (no 50k AED villas)
    ml_df = ml_df[ml_df['price_aed'] >= 200000]
    
    # 3. Must have a realistic price per sqm (no 10 AED/sqm typos)
    ml_df = ml_df[ml_df['price_per_sqm'] >= 3000]
    
    X = ml_df[features]
    y = ml_df[target]
    
    cat_cols = ['area_name_en', 'property_type_label', 'reg_type_label']
    num_cols = ['rooms_bucket', 'year']
    bool_col = ['has_parking']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('bool', 'passthrough', bool_col)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 🚨 SPEED FIX: Sample 150k rows for fast training
    print("✂️ Sampling 150,000 rows for fast training...")
    X_train_sample = X_train.sample(n=150000, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    print("⏳ Training Random Forest model... (this will take 10-15 seconds)")
    model.fit(X_train_sample, y_train_sample)
    
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"✅ Model Trained. MAE: {mae:.2f} AED/sqm | R-squared: {r2:.2f}")
    
    # Find Undervalued Deals
    print("🔍 Scanning transactions for undervalued deals...")
    ml_df['expected_psm'] = model.predict(X)
    ml_df['discount_pct'] = ((ml_df['expected_psm'] - ml_df['price_per_sqm']) / ml_df['expected_psm']) * 100
    
    # Filter for deals sold at least 20% below expected value
    undervalued = ml_df[ml_df['discount_pct'] >= 20].copy()
    undervalued = undervalued.sort_values(by='discount_pct', ascending=False)
    
    display_cols = ['instance_date', 'area_name_en', 'building_name', 'rooms_bucket', 
                    'property_type_label', 'price_aed', 'price_per_sqm', 'expected_psm', 'discount_pct']
    top_deals = undervalued[display_cols].head(15)
    
    return top_deals, mae, r2, model


def plot_feature_importance(model):
    """
    Plots the top 15 most important features driving price per sqm.
    """
    print("📊 Extracting feature importances...")
    # Get feature names after preprocessing
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['regressor'].feature_importances_
    
    # Create dataframe
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(15)
    
    # Clean up feature names for display
    imp_df['Feature'] = imp_df['Feature'].str.replace('cat__area_name_en_', 'Community: ')
    imp_df['Feature'] = imp_df['Feature'].str.replace('cat__reg_type_label_', 'Type: ')
    imp_df['Feature'] = imp_df['Feature'].str.replace('num__', '')
    imp_df['Feature'] = imp_df['Feature'].str.replace('bool__', '')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Top 15 Drivers of Dubai Real Estate Price per SQM (ML Feature Importance)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'fig05_ml_feature_importance.png', bbox_inches='tight')
    plt.show()
    print(f"📊 Chart saved to {VIS_DIR / 'fig05_ml_feature_importance.png'}")