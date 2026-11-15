# src/analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define the root directory for saving visualizations safely
ROOT_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = ROOT_DIR / 'visualizations'
VIS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_unit_size_illusion(df: pd.DataFrame):
    """
    RQ0 (The Hook): Are prices actually rising, or are units just shrinking?
    We plot Median Total Price vs Median Price/SQM vs Median Size (SQM) over time.
    """
    print("🔍 Running RQ0: The Unit Size Illusion...")
    
    # Group by year and calculate medians
    trend = df.groupby('year').agg(
        median_price=('price_aed', 'median'),
        median_psm=('price_per_sqm', 'median'),
        median_sqm=('area_sqm', 'median')
    ).reset_index()
    
    # Filter out low volume years and incomplete years
    volume = df.groupby('year').size()
    trend = trend[trend['year'].isin(volume[volume >= 1000].index)]
    trend = trend[trend['year'] < 2026]
    
    # Normalize to 2010 = 100 so we can compare the lines on one chart
    base_year = trend[trend['year'] == 2010].iloc[0]
    trend['price_index'] = (trend['median_price'] / base_year['median_price']) * 100
    trend['psm_index'] = (trend['median_psm'] / base_year['median_psm']) * 100
    trend['sqm_index'] = (trend['median_sqm'] / base_year['median_sqm']) * 100
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the three lines
    sns.lineplot(data=trend, x='year', y='price_index', marker='o', color='black', linewidth=2.5, label='Median Total Price')
    sns.lineplot(data=trend, x='year', y='psm_index', marker='s', color='red', linewidth=2.5, label='Median Price per SQM')
    sns.lineplot(data=trend, x='year', y='sqm_index', marker='^', color='blue', linewidth=2.5, label='Median Unit Size (SQM)')
    
    ax.axhline(100, color='grey', linestyle='--', alpha=0.5, label='2010 Baseline')
    ax.set_title('RQ0: The Unit Size Illusion (Indexed to 2010 = 100)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Transaction Year', fontsize=12)
    ax.set_ylabel('Index (2010 = 100)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'fig00_unit_size_illusion.png', bbox_inches='tight')
    plt.show()
    print(f"📊 Chart saved to {VIS_DIR / 'fig00_unit_size_illusion.png'}")
    
    return trend[['year', 'median_price', 'median_psm', 'median_sqm']]

# ==========================================
# RQ1: Market Maturity (Off-Plan vs Ready)
# ==========================================
def analyze_market_maturity(df: pd.DataFrame):
    print("🔍 Running RQ1: Off-Plan vs. Ready Ratio (Market Maturity)...")
    
    yearly_counts = df.groupby(['year', 'reg_type_label']).size().unstack(fill_value=0)
    yearly_counts['Total'] = yearly_counts['Ready'] + yearly_counts['Off-Plan']
    yearly_counts['Off_Plan_Pct'] = (yearly_counts['Off-Plan'] / yearly_counts['Total']) * 100
    yearly_counts['Ready_Pct'] = (yearly_counts['Ready'] / yearly_counts['Total']) * 100
    
    trend = yearly_counts[yearly_counts['Total'] >= 1000].reset_index()
    trend = trend[trend['year'] < 2026]
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=trend, x='year', y='Off_Plan_Pct', marker='o', color='darkorange', linewidth=2.5, ax=ax1, label='Off-Plan Sales (%)')
    sns.lineplot(data=trend, x='year', y='Ready_Pct', marker='s', color='teal', linewidth=2.5, ax=ax1, label='Ready (Resale) Sales (%)')
    ax1.axvspan(2008, 2009, color='grey', alpha=0.2, label='2008 Financial Crisis')
    ax1.axvspan(2020, 2021, color='blue', alpha=0.1, label='COVID-19')
    ax1.set_title('RQ1: Is Dubai Maturing? Off-Plan vs. Ready Transaction Ratio', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Transaction Year', fontsize=12)
    ax1.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'fig01_market_maturity.png', bbox_inches='tight')
    plt.show()
    return trend[['year', 'Off_Plan_Pct', 'Ready_Pct']]


# ==========================================
# RQ2: Off-Plan Discount (Stress Signal)
# ==========================================
def analyze_offplan_discount(df: pd.DataFrame):
    print("🔍 Running RQ2: Off-Plan vs Ready Price Premium/Discount...")
    
    grouped = df.groupby(['year', 'area_name_en', 'reg_type_label']).agg(
        median_psm=('price_per_sqm', 'median'),
        volume=('transaction_id', 'count')
    ).reset_index()
    grouped = grouped[grouped['volume'] >= 10]
    
    pivot_df = grouped.pivot_table(index=['year', 'area_name_en'], columns='reg_type_label', values='median_psm').reset_index()
    pivot_df = pivot_df.dropna(subset=['Off-Plan', 'Ready'])
    pivot_df['offplan_premium_pct'] = ((pivot_df['Off-Plan'] - pivot_df['Ready']) / pivot_df['Ready']) * 100
    
    city_trend = pivot_df.groupby('year').agg(
        median_premium=('offplan_premium_pct', 'median'),
        communities=('area_name_en', 'nunique')
    ).reset_index()
    city_trend = city_trend[(city_trend['year'] >= 2008) & (city_trend['year'] < 2026)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=city_trend, x='year', y='median_premium', marker='o', color='purple', linewidth=2.5, ax=ax, label='Median Off-Plan Premium (%)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Parity (Off-Plan = Ready)')
    ax.axvspan(2008, 2009, color='grey', alpha=0.2, label='2008 Financial Crisis')
    ax.axvspan(2020, 2021, color='blue', alpha=0.1, label='COVID-19')
    ax.set_title('RQ2: The Off-Plan Premium/Discount (vs. Ready Prices)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Transaction Year', fontsize=12)
    ax.set_ylabel('Off-Plan Premium over Ready (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'fig02_offplan_discount.png', bbox_inches='tight')
    plt.show()

    print("\n--- 🚨 RQ2 STRESS SIGNAL: Communities with the LOWEST Off-Plan Premium in 2025 ---")
    recent_stress = pivot_df[pivot_df['year'] == 2025].sort_values(by='offplan_premium_pct').head(5)
    stress_display = recent_stress[['area_name_en', 'Ready', 'Off-Plan', 'offplan_premium_pct']].copy()
    stress_display.columns = ['Community', 'Ready Price/sqm', 'Off-Plan Price/sqm', 'Premium (%)']
    print(stress_display.to_string(index=False))
    
    return city_trend


# ==========================================
# RQ3: Hidden Value Premium (Parking & Bedrooms)
# ==========================================
def analyze_parking_and_bedroom_premiums(df: pd.DataFrame):
    print("🔍 Running RQ3: Parking & Bedroom Premiums...")
    
    ready_df = df[(df['reg_type_label'] == 'Ready') & (df['rooms_bucket'].notna())].copy()
    ready_df = ready_df[ready_df['property_type_label'] == 'Unit']
    
    # --- PART A: Parking Premium ---
    parking_group = ready_df.groupby(['area_name_en', 'has_parking']).agg(
        median_psm=('price_per_sqm', 'median'),
        volume=('transaction_id', 'count')
    ).reset_index()
    
    parking_group = parking_group[parking_group['volume'] >= 20]
    parking_pivot = parking_group.pivot(index='area_name_en', columns='has_parking', values='median_psm').dropna()
    
    if not parking_pivot.empty:
        parking_pivot.columns = ['No_Parking_PSM', 'Has_Parking_PSM']
        parking_pivot['Parking_Premium_Pct'] = ((parking_pivot['Has_Parking_PSM'] - parking_pivot['No_Parking_PSM']) / parking_pivot['No_Parking_PSM']) * 100
        top_parking = parking_pivot.sort_values(by='Parking_Premium_Pct', ascending=False).head(10).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        # 🚨 FIXED: Removed hue and legend to support newer Seaborn
        sns.barplot(data=top_parking, x='Parking_Premium_Pct', y='area_name_en', palette='Blues_r', ax=ax)
        ax.set_title('RQ3: Top 10 Communities by Parking Premium (% increase in price/sqm)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Parking Premium (%)', fontsize=12)
        ax.set_ylabel('Community', fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIS_DIR / 'fig03_parking_premium.png', bbox_inches='tight')
        plt.show()
    else:
        top_parking = pd.DataFrame()
    
    # --- PART B: Bedroom Premium ---
    br_df = ready_df[ready_df['rooms_bucket'].isin([1, 2])].copy()
    br_group = br_df.groupby(['area_name_en', 'rooms_bucket']).agg(
        median_psm=('price_per_sqm', 'median'),
        volume=('transaction_id', 'count')
    ).reset_index()
    
    br_group = br_group[br_group['volume'] >= 20]
    br_pivot = br_group.pivot(index='area_name_en', columns='rooms_bucket', values='median_psm').dropna()
    
    if not br_pivot.empty:
        br_pivot.columns = ['1BR_PSM', '2BR_PSM']
        br_pivot['1BR_Premium_Pct'] = ((br_pivot['1BR_PSM'] - br_pivot['2BR_PSM']) / br_pivot['2BR_PSM']) * 100
        top_family = br_pivot.sort_values(by='1BR_Premium_Pct', ascending=True).head(10).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        # 🚨 FIXED: Removed hue and legend to support newer Seaborn
        sns.barplot(data=top_family, x='1BR_Premium_Pct', y='area_name_en', palette='Purples_r', ax=ax)
        ax.set_title('RQ3: Communities where 2BRs command the highest premium over 1BRs', fontsize=15, fontweight='bold')
        ax.set_xlabel('1BR Premium over 2BR (%) [Negative = 2BR is more expensive per sqm]', fontsize=12)
        ax.set_ylabel('Community', fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIS_DIR / 'fig03_bedroom_premium.png', bbox_inches='tight')
        plt.show()
    else:
        top_family = pd.DataFrame()

    return top_parking, top_family


# ==========================================
# MAIN RUNNER
# ==========================================
def run_all_analysis(df: pd.DataFrame):
    """Runs all Research Questions."""
    print("\n" + "="*50)
    rq0_results = analyze_unit_size_illusion(df)
    print("\n" + "="*50)
    rq1_results = analyze_market_maturity(df)
    print("\n" + "="*50)
    rq2_results = analyze_offplan_discount(df)
    print("\n" + "="*50)
    rq3_parking, rq3_family = analyze_parking_and_bedroom_premiums(df)
    
    return rq0_results, rq1_results, rq2_results, rq3_parking, rq3_family