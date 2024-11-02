import pandas as pd
import numpy as np
from datetime import datetime
import json

def clean_fed_rates(fed_rates_df):
    """Clean and prepare FED-RATES data"""
    print("\nCleaning Fed Rates data...")

    fed_rates_df['Date'] = pd.to_datetime(fed_rates_df['Date'])
    fed_rates_df['New Rate (%)'] = pd.to_numeric(
        fed_rates_df['New Rate (%)'].astype(str).str.replace('%', ''), 
        errors='coerce'
    )

    # Remove any NaT dates or null rates
    fed_rates_df = fed_rates_df.dropna(subset=['Date', 'New Rate (%)'])
    fed_rates_df = fed_rates_df.sort_values('Date').reset_index(drop=True)

    return fed_rates_df

def clean_transactions(transactions_df):
    """Clean and prepare TRANSACTIONS-PT data"""
    print("\nCleaning Transactions data...")

    # Clean transaction dates
    transactions_df['DATE'] = pd.to_datetime(transactions_df['DATE'], format='%m/%d/%Y')

    # Clean numeric columns
    numeric_cols = ['PRICE', 'LOT AREA', 'LOT FRONTAGE', 'BASE FAR', 'BASE ZFA', 'PPZFA']
    for col in numeric_cols:
        transactions_df[col] = pd.to_numeric(
            transactions_df[col].astype(str).str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        )

    # Remove transactions with missing dates or prices
    transactions_df = transactions_df.dropna(subset=['DATE', 'PRICE'])

    return transactions_df

def create_expanded_rates(fed_rates_df, latest_transaction_date):
    """Create expanded rate periods with daily granularity"""
    print("\nCreating expanded rate periods...")

    expanded_rates = []

    # Create daily series for each rate period
    for i in range(len(fed_rates_df) - 1):
        start_date = fed_rates_df['Date'].iloc[i]
        end_date = fed_rates_df['Date'].iloc[i + 1]
        rate = fed_rates_df['New Rate (%)'].iloc[i]

        if pd.isna(start_date) or pd.isna(end_date) or pd.isna(rate):
            print(f"Skipping row {i} due to missing data")
            continue

        dates = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D')
        period_rates = pd.DataFrame({
            'Date': dates,
            'Rate': rate
        })
        expanded_rates.append(period_rates)

    # Add final period
    final_start = fed_rates_df['Date'].iloc[-1]
    final_rate = fed_rates_df['New Rate (%)'].iloc[-1]

    if final_start < latest_transaction_date and not pd.isna(final_start) and not pd.isna(final_rate):
        final_dates = pd.date_range(start=final_start, end=latest_transaction_date, freq='D')
        final_period = pd.DataFrame({
            'Date': final_dates,
            'Rate': final_rate
        })
        expanded_rates.append(final_period)

    return pd.concat(expanded_rates, ignore_index=True)

def identify_outliers(df, column, n_std=3):
    """Identify outliers using z-score method"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > n_std

def calculate_market_lag_effects(df):
    """Calculate market response to Fed rate changes and policy events"""
    print("\nAnalyzing market lag effects...")

    results = {}

    # Calculate baseline PPZFA for premium/discount calculations
    baseline_ppzfa = df['PPZFA'].median()

    # 1. Fed Rate Change Analysis
    # Group data into periods (e.g., 3 months) and calculate average metrics
    df['YearMonth'] = df['DATE'].dt.to_period('M')
    monthly_metrics = df.groupby('YearMonth').agg({
        'Rate': 'mean',
        'PPZFA': 'mean',
        'PRICE': ['count', 'mean']
    })

    # Flatten column names for easier access
    monthly_metrics.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                             for col in monthly_metrics.columns]
    monthly_metrics = monthly_metrics.reset_index()

    # Calculate lagged correlations for 3 and 6 month periods
    for lag_months in [3, 6]:
        # Lag the rate changes
        monthly_metrics[f'Rate_Lag_{lag_months}M'] = monthly_metrics['Rate_mean'].shift(lag_months)

        # Calculate correlations
        rate_ppzfa_corr = monthly_metrics['PPZFA_mean'].corr(monthly_metrics[f'Rate_Lag_{lag_months}M'])
        rate_volume_corr = monthly_metrics['PRICE_count'].corr(monthly_metrics[f'Rate_Lag_{lag_months}M'])

        results[f'{lag_months}M_lag'] = {
            'rate_ppzfa_correlation': rate_ppzfa_corr,
            'rate_volume_correlation': rate_volume_corr
        }

    # 2. Policy Event Analysis
    policy_events = {
        '421a_expiration': '2022-06-15',
        'covid_outbreak': '2018-01-01'
    }

    for event, date in policy_events.items():
        event_date = pd.to_datetime(date)

        # Analyze 6 months before and after
        pre_period = df[
            (df['DATE'] >= event_date - pd.DateOffset(months=6)) &
            (df['DATE'] < event_date)
        ]
        post_period = df[
            (df['DATE'] >= event_date) &
            (df['DATE'] < event_date + pd.DateOffset(months=6))
        ]

        # Calculate metrics
        if len(pre_period) > 0 and len(post_period) > 0:
            results[f'{event}_impact'] = {
                'pre_avg_ppzfa': pre_period['PPZFA'].mean(),
                'post_avg_ppzfa': post_period['PPZFA'].mean(),
                'pre_volume': len(pre_period),
                'post_volume': len(post_period),
                'ppzfa_change_pct': ((post_period['PPZFA'].mean() / pre_period['PPZFA'].mean()) - 1) * 100 if pre_period['PPZFA'].mean() != 0 else 0,
                'volume_change_pct': ((len(post_period) / len(pre_period)) - 1) * 100 if len(pre_period) > 0 else 0
            }
        else:
            results[f'{event}_impact'] = {
                'pre_avg_ppzfa': 0,
                'post_avg_ppzfa': 0,
                'pre_volume': 0,
                'post_volume': 0,
                'ppzfa_change_pct': 0,
                'volume_change_pct': 0
            }

    # 3. Geographic Response Analysis
    # Analyze how different boroughs respond to rate changes
    borough_responses = {}
    for borough in df['BOROUGH'].unique():
        borough_data = df[df['BOROUGH'] == borough]
        if len(borough_data) > 0:
            borough_monthly = borough_data.groupby('YearMonth').agg({
                'Rate': 'mean',
                'PPZFA': 'mean',
                'PRICE': ['count', 'mean']
            })

            # Flatten column names
            borough_monthly.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                                    for col in borough_monthly.columns]
            borough_monthly = borough_monthly.reset_index()

            # Calculate 3-month lag correlation for each borough
            borough_monthly['Rate_Lag_3M'] = borough_monthly['Rate_mean'].shift(3)

            borough_responses[borough] = {
                'rate_ppzfa_correlation': borough_monthly['PPZFA_mean'].corr(borough_monthly['Rate_Lag_3M']),
                'avg_ppzfa': borough_data['PPZFA'].mean(),
                'ppzfa_premium_discount': ((borough_data['PPZFA'].mean() / baseline_ppzfa) - 1) * 100
            }

    results['borough_responses'] = borough_responses

    return results

def analyze_time_periods(df):
    """Analyze trends across different time periods"""
    print("\nAnalyzing time periods...")

    # Add time period columns
    df['Year'] = df['DATE'].dt.year
    df['Quarter'] = df['DATE'].dt.quarter
    df['YearQuarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)

    # Annual analysis
    annual_stats = df.groupby('Year').agg({
        'PRICE': ['count', 'sum', 'mean'],
        'PPZFA': 'mean',
        'Rate': 'mean'
    }).round(2)

    # Quarterly analysis
    quarterly_stats = df.groupby('YearQuarter').agg({
        'PRICE': ['count', 'sum', 'mean'],
        'PPZFA': 'mean',
        'Rate': 'mean'
    }).round(2)

    return {
        'annual': annual_stats,
        'quarterly': quarterly_stats
    }

def analyze_geography(df):
    """Analyze geographic patterns"""
    print("\nAnalyzing geographic patterns...")

    # Check for required columns
    geography_cols = {
        'borough': next((col for col in df.columns if 'BOROUGH' in col.upper()), None),
        'neighborhood': next((col for col in df.columns if 'NEIGHBORHOOD' in col.upper()), None)
    }

    results = {}

    try:
        if geography_cols['borough']:
            # Borough level analysis
            borough_stats = df.groupby(geography_cols['borough']).agg({
                'PRICE': ['count', 'sum', 'mean'],
                'PPZFA': 'mean'
            }).round(2)
            results['borough'] = borough_stats
        else:
            print("Warning: No borough column found")
            results['borough'] = pd.DataFrame()

        if geography_cols['borough'] and geography_cols['neighborhood']:
            # Neighborhood level analysis
            neighborhood_stats = df.groupby([geography_cols['borough'], geography_cols['neighborhood']]).agg({
                'PRICE': ['count', 'sum', 'mean'],
                'PPZFA': 'mean'
            }).round(2)
            results['neighborhood'] = neighborhood_stats

            # Identify hotspots (top 10% by PPZFA)
            hotspot_threshold = df['PPZFA'].quantile(0.9)
            hotspots = df[df['PPZFA'] > hotspot_threshold].groupby(
                [geography_cols['borough'], geography_cols['neighborhood']]
            ).size()
            results['hotspots'] = hotspots
        else:
            print("Warning: Missing required columns for neighborhood analysis")
            results['neighborhood'] = pd.DataFrame()
            results['hotspots'] = pd.DataFrame()

    except Exception as e:
        print(f"Warning: Error in geographic analysis: {str(e)}")
        results = {
            'borough': pd.DataFrame(),
            'neighborhood': pd.DataFrame(),
            'hotspots': pd.DataFrame()
        }

    return results

def get_combined_zoning_info(row, zoning_columns):
    """Combine zoning information from multiple columns"""
    all_zonings = []
    for col in zoning_columns:
        if pd.notna(row[col]):
            all_zonings.append(str(row[col]))
    return ' | '.join(all_zonings) if all_zonings else 'UNKNOWN'

def analyze_zoning(df):
    """Analyze zoning categories"""
    print("\nAnalyzing zoning categories...")

    # Check if zoning columns exist
    zoning_columns = [col for col in df.columns if 'ZONING' in col.upper()]
    if not zoning_columns:
        print("Warning: No zoning columns found. Skipping zoning analysis.")
        return {'density': pd.DataFrame()}

    print(f"Using columns {zoning_columns} for zoning analysis")

    # Define density categories
    density_mapping = {
        'LOW': ['R1', 'R2', 'R3', 'R4', 'R5'],
        'MEDIUM': ['R6', 'R7', 'C2-4'],
        'HIGH': ['R8', 'R9', 'R10']
    }

    try:
        # Create combined zoning information
        df['COMBINED_ZONING'] = df.apply(
            lambda row: get_combined_zoning_info(row, zoning_columns), 
            axis=1
        )

        # Add density category based on combined zoning
        df['DENSITY_CATEGORY'] = df['COMBINED_ZONING'].apply(
            lambda x: next(
                (k for k, v in density_mapping.items() 
                 if any(z in x for z in v)), 
                'OTHER'
            )
        )

        # Analyze by density
        density_stats = df.groupby('DENSITY_CATEGORY', observed=False).agg({
            'PRICE': ['count', 'sum', 'mean'],
            'PPZFA': 'mean'
        }).round(2)

    except Exception as e:
        print(f"Warning: Error in zoning analysis: {str(e)}")
        print("Proceeding with basic analysis...")
        density_stats = pd.DataFrame()

    return {
        'density': density_stats
    }

def analyze_physical_characteristics(df):
    """Analyze physical characteristics"""
    print("\nAnalyzing physical characteristics...")

    # Lot frontage categories
    df['FRONTAGE_CATEGORY'] = pd.cut(
        df['LOT FRONTAGE'],
        bins=[0, 25, 45, float('inf')],
        labels=['≤25 feet', '25-45 feet', '>45 feet']
    )

    # Analyze by lot characteristics
    frontage_stats = df.groupby('FRONTAGE_CATEGORY', observed=True).agg({
        'PRICE': ['count', 'sum', 'mean'],
        'PPZFA': 'mean'
    }).round(2)

    return {
        'frontage': frontage_stats
    }

def analyze_sliver_law(df):
    """Analyze Sliver Law impacts"""
    print("\nAnalyzing Sliver Law impacts...")

    # Get zoning columns
    zoning_columns = [col for col in df.columns if 'ZONING' in col.upper()]
    if not zoning_columns:
        print("Warning: No zoning columns found. Skipping Sliver Law analysis.")
        return {'sliver': pd.DataFrame()}

    # Identify properties subject to Sliver Law
    sliver_districts = [
        'R7-2', 'R7D', 'R7X', 'R8', 'R9', 'R10',
        'C1-6', 'C1-7', 'C1-8', 'C1-9', 'C2-6', 'C2-7', 'C2-8',
        'C4-4D', 'C4-5D', 'C4-5X', 'C4-6A', 'C4-7A',
        'C5-1A', 'C5-2A', 'C6-2A', 'C6-3A', 'C6-3D', 'C6-3X', 'C6-4A', 'C6-4X'
    ]

    try:
        # Create combined zoning information if not already exists
        if 'COMBINED_ZONING' not in df.columns:
            df['COMBINED_ZONING'] = df.apply(
                lambda row: get_combined_zoning_info(row, zoning_columns), 
                axis=1
            )

        # Check if property is subject to Sliver Law based on combined zoning
        df['SLIVER_APPLICABLE'] = (
            (df['LOT FRONTAGE'] < 45) & 
            df['COMBINED_ZONING'].apply(
                lambda x: any(district in str(x) for district in sliver_districts)
            )
        )

        # Analyze Sliver Law impacts
        sliver_stats = df.groupby('SLIVER_APPLICABLE', observed=False).agg({
            'PRICE': ['count', 'sum', 'mean'],
            'PPZFA': 'mean'
        }).round(2)

    except Exception as e:
        print(f"Warning: Error in Sliver Law analysis: {str(e)}")
        print("Proceeding with basic analysis...")
        sliver_stats = pd.DataFrame()

    return {
        'sliver': sliver_stats
    }

def save_results(results, merged_df):
    """Save analysis results to files"""
    print("\nSaving results...")

    # Save merged data
    merged_df.to_csv('output/merged_data.csv', index=False)
    print("Merged data saved to 'output/merged_data.csv'")

    # Save analysis results
    for category, data in results.items():
        if isinstance(data, dict):
            for subcategory, content in data.items():
                if isinstance(content, pd.DataFrame):
                    filename = f'output/{category}_{subcategory}_analysis.csv'
                    content.to_csv(filename)
                    print(f"Saved {category} {subcategory} analysis to {filename}")
                elif isinstance(content, dict):
                    filename = f'output/{category}_{subcategory}_analysis.json'
                    with open(filename, 'w') as f:
                        json.dump(content, f, indent=2)
                    print(f"Saved {category} {subcategory} analysis to {filename}")
        elif isinstance(data, pd.DataFrame):
            filename = f'output/{category}_analysis.csv'
            data.to_csv(filename)
            print(f"Saved {category} analysis to {filename}")

def print_analysis_results(results, merged_df):
    """Print comprehensive analysis results"""
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)

    # Calculate baseline PPZFA for premium/discount calculations
    baseline_ppzfa = merged_df['PPZFA'].median()

    # 1. Market Overview
    print("\n1. MARKET OVERVIEW")
    print("-"*30)
    print(f"Total Transactions: {len(merged_df):,}")
    print(f"Total Volume: ${merged_df['PRICE'].sum()/1e9:.2f}B")
    print(f"Baseline PPZFA: ${baseline_ppzfa:.2f}")
    print(f"Average Rate: {merged_df['Rate'].mean():.2f}%")

    # 2. Lag Analysis Results
    print("\n2. MARKET LAG EFFECTS")
    print("-"*30)

    # Fed Rate Impact
    print("\na. Fed Rate Impact:")
    for period, metrics in results['lag_results'].items():
        if period.endswith('M_lag'):
            print(f"\n{period} Analysis:")
            print(f"PPZFA Response to Rate Changes: {metrics['rate_ppzfa_correlation']:.3f}")
            print(f"Volume Response to Rate Changes: {metrics['rate_volume_correlation']:.3f}")

    # Policy Events Impact
    print("\nb. Policy Event Impacts:")
    policy_events = ['421a_expiration_impact', 'covid_outbreak_impact']
    for event in policy_events:
        if event in results['lag_results']:
            metrics = results['lag_results'][event]
            event_name = event.replace('_impact', '').replace('_', ' ').title()
            print(f"\n{event_name}:")
            print(f"Pre-Event PPZFA: ${metrics['pre_avg_ppzfa']:.2f}")
            print(f"Post-Event PPZFA: ${metrics['post_avg_ppzfa']:.2f}")
            print(f"PPZFA Change: {metrics['ppzfa_change_pct']:.1f}%")
            print(f"Volume Change: {metrics['volume_change_pct']:.1f}%")

    # Geographic Analysis
    print("\n3. GEOGRAPHIC ANALYSIS")
    print("-"*30)
    if 'borough' in results['geography']:
        borough_stats = results['geography']['borough']
        for borough in borough_stats.index:
            print(f"\n{borough}:")
            count = borough_stats.loc[borough, ('PRICE', 'count')]
            avg_ppzfa = borough_stats.loc[borough, ('PPZFA', 'mean')]
            print(f"Transaction Count: {count:,}")
            print(f"Average PPZFA: ${avg_ppzfa:.2f}")
            premium_discount = ((avg_ppzfa / baseline_ppzfa) - 1) * 100
            print(f"Premium/Discount to Baseline: {premium_discount:+.1f}%")

    # Zoning Analysis
    print("\n4. ZONING ANALYSIS")
    print("-"*30)
    if 'density' in results['zoning']:
        density_stats = results['zoning']['density']
        for category in density_stats.index:
            print(f"\n{category}:")
            count = density_stats.loc[category, ('PRICE', 'count')]
            avg_ppzfa = density_stats.loc[category, ('PPZFA', 'mean')]
            print(f"Transaction Count: {count:,}")
            print(f"Average PPZFA: ${avg_ppzfa:.2f}")
            premium_discount = ((avg_ppzfa / baseline_ppzfa) - 1) * 100
            print(f"Premium/Discount to Baseline: {premium_discount:+.1f}%")

    # Physical Characteristics
    print("\n5. PHYSICAL CHARACTERISTICS")
    print("-"*30)
    if 'frontage' in results['physical']:
        frontage_stats = results['physical']['frontage']
        for category in frontage_stats.index:
            print(f"\n{category}:")
            count = frontage_stats.loc[category, ('PRICE', 'count')]
            avg_ppzfa = frontage_stats.loc[category, ('PPZFA', 'mean')]
            print(f"Transaction Count: {count:,}")
            print(f"Average PPZFA: ${avg_ppzfa:.2f}")
            premium_discount = ((avg_ppzfa / baseline_ppzfa) - 1) * 100
            print(f"Premium/Discount to Baseline: {premium_discount:+.1f}%")

    # Sliver Law Analysis
    print("\n6. SLIVER LAW ANALYSIS")
    print("-"*30)
    if 'sliver' in results['sliver_law']:
        sliver_stats = results['sliver_law']['sliver']
        for applicable in [True, False]:
            if applicable in sliver_stats.index:
                print(f"\nSliver Law Applicable: {applicable}")
                count = sliver_stats.loc[applicable, ('PRICE', 'count')]
                avg_ppzfa = sliver_stats.loc[applicable, ('PPZFA', 'mean')]
                print(f"Transaction Count: {count:,}")
                print(f"Average PPZFA: ${avg_ppzfa:.2f}")
                premium_discount = ((avg_ppzfa / baseline_ppzfa) - 1) * 100
                print(f"Premium/Discount to Baseline: {premium_discount:+.1f}%")

def main():
    """Main analysis function"""
    print("\nStarting analysis...")

    # 1. Data Loading and Preparation
    print("\nLoading and preparing data...")

    # Load FED-RATES data
    fed_rates_df = pd.read_csv('FED-RATES.csv')
    fed_rates_df = clean_fed_rates(fed_rates_df)

    # Load TRANSACTIONS-PT data
    transactions_df = pd.read_csv('TRANSACTIONS-PT.csv')
    transactions_df = clean_transactions(transactions_df)

    # Create expanded rate periods
    expanded_rates = create_expanded_rates(
        fed_rates_df,
        transactions_df['DATE'].max()
    )

    # Merge datasets
    print("\nMerging datasets...")
    merged_df = pd.merge(
        transactions_df,
        expanded_rates,
        left_on='DATE',
        right_on='Date',
        how='left'
    )

    # Identify and flag outliers
    print("\nIdentifying outliers...")
    merged_df['PRICE_OUTLIER'] = identify_outliers(merged_df, 'PRICE')
    merged_df['PPZFA_OUTLIER'] = identify_outliers(merged_df, 'PPZFA')

    # Calculate market lag effects
    print("\nCalculating market lag effects...")
    lag_effects = calculate_market_lag_effects(merged_df)

    # Perform analyses
    results = {
        'lag_results': lag_effects,
        'time_periods': analyze_time_periods(merged_df),
        'geography': analyze_geography(merged_df),
        'zoning': analyze_zoning(merged_df),
        'physical': analyze_physical_characteristics(merged_df),
        'sliver_law': analyze_sliver_law(merged_df)
    }

    # Save results
    save_results(results, merged_df)

    # Print detailed analysis results
    print_analysis_results(results, merged_df)

if __name__ == "__main__":
    main()
    print("\nStarting analysis...")

    # 1. Data Loading and Preparation
    print("\nLoading and preparing data...")

    # Load FED-RATES data
    fed_rates_df = pd.read_csv('FED-RATES.csv')
    fed_rates_df = clean_fed_rates(fed_rates_df)

    # Load TRANSACTIONS-PT data
    transactions_df = pd.read_csv('TRANSACTIONS-PT.csv')
    transactions_df = clean_transactions(transactions_df)

    # Create expanded rate periods
    expanded_rates = create_expanded_rates(
        fed_rates_df,
        transactions_df['DATE'].max()
    )

    # Merge datasets
    print("\nMerging datasets...")
    merged_df = pd.merge(
        transactions_df,
        expanded_rates,
        left_on='DATE',
        right_on='Date',
        how='left'
    )

    # Identify and flag outliers
    print("\nIdentifying outliers...")
    merged_df['PRICE_OUTLIER'] = identify_outliers(merged_df, 'PRICE')
    merged_df['PPZFA_OUTLIER'] = identify_outliers(merged_df, 'PPZFA')

    # Calculate market lag effects
    print("\nCalculating market lag effects...")
    lag_effects = calculate_market_lag_effects(merged_df)

    # Perform analyses
    results = {
        'lag_results': lag_effects,
        'time_periods': analyze_time_periods(merged_df),
        'geography': analyze_geography(merged_df),
        'zoning': analyze_zoning(merged_df),
        'physical': analyze_physical_characteristics(merged_df),
        'sliver_law': analyze_sliver_law(merged_df)
    }

    # Save all results
    save_results(results, merged_df)

    # Print detailed analysis results
    print_analysis_results(results, merged_df)

if __name__ == "__main__":
    main()
