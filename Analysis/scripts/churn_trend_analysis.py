#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script analyzes churn patterns over time to identify seasonal trends and the impact
of major business events on customer churn behavior.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import calendar
from scipy import stats
from sklearn.linear_model import LinearRegression

# Add the parent directory to the path to import from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_preprocessing import load_and_preprocess_data

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TRENDS_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'Analysis', 'images', 'churn_trends')
TRENDS_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Analysis', 'results', 'churn_trend_analysis_results')

# Create directories for saving results and images if they don't exist
os.makedirs(TRENDS_IMAGES_DIR, exist_ok=True)
os.makedirs(TRENDS_RESULTS_DIR, exist_ok=True)

def create_synthetic_dates(df, start_date='2020-01-01'):
    """
    Create synthetic dates for customers based on their tenure.
    This is needed because the original dataset doesn't contain actual dates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The customer churn dataset
    start_date : str
        The starting date for the oldest customers
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with synthetic dates added
    """
    # Make a copy to avoid modifying the original dataframe
    date_df = df.copy()
    
    # Convert start_date to datetime
    start_date = pd.to_datetime(start_date)
    
    # Use tenure to create a join_date (when the customer joined)
    # For customers with tenure = 0, set them to have joined on the last day
    max_tenure = date_df['tenure'].max()
    
    # Create a function to calculate join date
    def calculate_join_date(tenure):
        if tenure == 0:
            return start_date + pd.DateOffset(months=max_tenure)
        else:
            return start_date + pd.DateOffset(months=max_tenure - tenure)
    
    # Apply the function to create join_date
    date_df['join_date'] = date_df['tenure'].apply(calculate_join_date)
    
    # For customers who have churned, create a churn_date
    # Customers who haven't churned have a churn_date of NaT
    date_df['churn_date'] = np.where(
        date_df['Churn'] == 'Yes',
        date_df['join_date'] + pd.to_timedelta(date_df['tenure'] * 30, unit='D'),  # Convert months to days (approx 30 days per month)
        pd.NaT
    )
    
    # Create a last_active_date - either the churn_date or the analysis_date (today)
    analysis_date = start_date + pd.DateOffset(months=max_tenure + 1)
    date_df['last_active_date'] = date_df['churn_date'].fillna(analysis_date)
    
    return date_df

def add_business_events(start_date='2020-01-01', end_date='2022-01-01'):
    """
    Create a dataframe of major business events for analysis.
    
    Parameters:
    -----------
    start_date : str
        The starting date for the event timeline
    end_date : str
        The ending date for the event timeline
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing business events
    """
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Define business events
    events = [
        {
            'date': '2020-03-15',
            'event': 'COVID-19 Lockdown',
            'description': 'Start of COVID-19 lockdowns affecting customer behavior',
            'category': 'External'
        },
        {
            'date': '2020-05-01',
            'event': 'Price Increase',
            'description': '10% price increase across all service plans',
            'category': 'Pricing'
        },
        {
            'date': '2020-07-15',
            'event': 'New Competitor Entry',
            'description': 'Major competitor entered the market with promotional pricing',
            'category': 'Competition'
        },
        {
            'date': '2020-09-01',
            'event': 'Service Outage',
            'description': 'Major service outage affecting 30% of customers',
            'category': 'Service Issue'
        },
        {
            'date': '2020-11-15',
            'event': 'Black Friday Promotion',
            'description': 'Special discounts for new and existing customers',
            'category': 'Promotion'
        },
        {
            'date': '2021-01-10',
            'event': 'New Feature Launch',
            'description': 'Launch of enhanced service features',
            'category': 'Product'
        },
        {
            'date': '2021-03-01',
            'event': 'Customer Service Enhancement',
            'description': 'Improved customer service response times',
            'category': 'Service Improvement'
        },
        {
            'date': '2021-05-15',
            'event': 'System Upgrade',
            'description': 'Major system upgrade with temporary service disruptions',
            'category': 'Service Issue'
        },
        {
            'date': '2021-07-01',
            'event': 'Loyalty Program Launch',
            'description': 'New loyalty program with rewards for long-term customers',
            'category': 'Retention'
        },
        {
            'date': '2021-09-15',
            'event': 'Competitor Price Drop',
            'description': 'Major competitor reduced prices by 15%',
            'category': 'Competition'
        },
        {
            'date': '2021-11-25',
            'event': 'Holiday Promotion',
            'description': 'Special holiday discounts and package deals',
            'category': 'Promotion'
        }
    ]
    
    # Create a dataframe from the events
    events_df = pd.DataFrame(events)
    events_df['date'] = pd.to_datetime(events_df['date'])
    
    # Filter events within the date range
    events_df = events_df[(events_df['date'] >= start_date) & (events_df['date'] <= end_date)]
    
    return events_df

def aggregate_churn_by_time(date_df, freq='M'):
    """
    Aggregate churn data over time (monthly, quarterly, etc.).
    
    Parameters:
    -----------
    date_df : pandas.DataFrame
        Dataframe with date information
    freq : str
        Frequency for aggregation ('M' for month, 'Q' for quarter, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        Aggregated churn data
    """
    # Make a copy of the dataframe
    df_copy = date_df.copy()
    
    # Ensure date columns are datetime type
    df_copy['churn_date'] = pd.to_datetime(df_copy['churn_date'], errors='coerce')
    df_copy['last_active_date'] = pd.to_datetime(df_copy['last_active_date'], errors='coerce')
    
    # Convert join_date to period
    df_copy['join_period'] = df_copy['join_date'].dt.to_period(freq)
    
    # Convert churn_date to period (for those who churned)
    churned = df_copy['Churn'] == 'Yes'
    df_copy.loc[churned, 'churn_period'] = df_copy.loc[churned, 'churn_date'].dt.to_period(freq)
    
    # Count new customers by period
    new_customers = df_copy.groupby('join_period').size().reset_index(name='new_customers')
    new_customers['join_period'] = new_customers['join_period'].astype(str)
    
    # Count churned customers by period
    churned_customers = df_copy[churned].groupby('churn_period').size().reset_index(name='churned_customers')
    churned_customers['churn_period'] = churned_customers['churn_period'].astype(str)
    
    # Create a date range covering all periods
    all_periods = pd.period_range(
        start=min(df_copy['join_date']),
        end=max(df_copy['last_active_date']),
        freq=freq
    ).astype(str).tolist()
    
    # Create the final dataframe
    churn_over_time = pd.DataFrame({'period': all_periods})
    
    # Merge with new and churned customer counts
    churn_over_time = (
        churn_over_time
        .merge(new_customers, left_on='period', right_on='join_period', how='left')
        .merge(churned_customers, left_on='period', right_on='churn_period', how='left')
        .drop(['join_period', 'churn_period'], axis=1)
        .fillna(0)
    )
    
    # Calculate cumulative customers and active customers
    churn_over_time['new_customers'] = churn_over_time['new_customers'].astype(int)
    churn_over_time['churned_customers'] = churn_over_time['churned_customers'].astype(int)
    churn_over_time['cumulative_new'] = churn_over_time['new_customers'].cumsum()
    churn_over_time['cumulative_churned'] = churn_over_time['churned_customers'].cumsum()
    churn_over_time['active_customers'] = churn_over_time['cumulative_new'] - churn_over_time['cumulative_churned']
    
    # Calculate churn rate
    churn_over_time['churn_rate'] = np.where(
        churn_over_time['active_customers'] + churn_over_time['churned_customers'] > 0,
        churn_over_time['churned_customers'] / (churn_over_time['active_customers'] + churn_over_time['churned_customers']),
        0
    )
    
    # Convert period to datetime for easier plotting
    churn_over_time['date'] = pd.to_datetime(churn_over_time['period'])
    
    return churn_over_time

def plot_churn_trends(churn_over_time, events_df=None):
    """
    Plot churn trends over time and highlight business events.
    
    Parameters:
    -----------
    churn_over_time : pandas.DataFrame
        Aggregated churn data over time
    events_df : pandas.DataFrame, optional
        Business events data
    """
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    # Plot 1: Customers over time (new, churned, active)
    ax1 = axs[0]
    ax1.plot(churn_over_time['date'], churn_over_time['active_customers'], 'b-', label='Active Customers')
    ax1.plot(churn_over_time['date'], churn_over_time['new_customers'], 'g-', label='New Customers')
    ax1.plot(churn_over_time['date'], churn_over_time['churned_customers'], 'r-', label='Churned Customers')
    ax1.set_title('Customer Counts Over Time')
    ax1.set_ylabel('Number of Customers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Churn rate over time
    ax2 = axs[1]
    ax2.plot(churn_over_time['date'], churn_over_time['churn_rate'] * 100, 'r-')
    ax2.set_title('Churn Rate Over Time')
    ax2.set_ylabel('Churn Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Calculate moving average for smoother trend
    window_size = 3
    churn_over_time['churn_rate_ma'] = churn_over_time['churn_rate'].rolling(window=window_size, min_periods=1).mean()
    ax2.plot(churn_over_time['date'], churn_over_time['churn_rate_ma'] * 100, 'b-', label=f'{window_size}-Month Moving Average')
    ax2.legend()
    
    # Plot 3: Net customer growth
    ax3 = axs[2]
    churn_over_time['net_growth'] = churn_over_time['new_customers'] - churn_over_time['churned_customers']
    ax3.bar(churn_over_time['date'], churn_over_time['net_growth'], color=np.where(churn_over_time['net_growth'] >= 0, 'g', 'r'))
    ax3.set_title('Net Customer Growth')
    ax3.set_ylabel('Net Change in Customers')
    ax3.grid(True, alpha=0.3)
    
    # If events data is provided, add event markers
    if events_df is not None:
        for ax in axs:
            # Plot vertical lines for events
            for _, event in events_df.iterrows():
                ax.axvline(x=event['date'], color='gray', linestyle='--', alpha=0.7)
                
                # Add event label to the first plot only
                if ax == ax1:
                    ax.annotate(
                        event['event'],
                        xy=(event['date'], ax.get_ylim()[1] * 0.9),
                        xytext=(event['date'], ax.get_ylim()[1] * 0.9),
                        rotation=90,
                        ha='right',
                        fontsize=8
                    )
    
    # Format x-axis for dates
    plt.xlabel('Date')
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(TRENDS_IMAGES_DIR, 'churn_trends_over_time.png'))
    plt.close()

def analyze_seasonal_patterns(churn_over_time):
    """
    Analyze seasonal patterns in churn data.
    
    Parameters:
    -----------
    churn_over_time : pandas.DataFrame
        Aggregated churn data over time
        
    Returns:
    --------
    tuple
        Decomposition results and monthly aggregation
    """
    # Require at least 2 years of data for seasonal analysis
    if len(churn_over_time) < 24:
        # Create a copy with the index as the date for cleaner time series analysis
        churn_ts = churn_over_time.set_index('date')['churn_rate']
        
        # Plot monthly churn rates
        plt.figure(figsize=(12, 6))
        churn_ts.plot()
        plt.title('Monthly Churn Rate')
        plt.xlabel('Date')
        plt.ylabel('Churn Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(TRENDS_IMAGES_DIR, 'monthly_churn_rate.png'))
        plt.close()
        
        print("Warning: Less than 2 years of data available. Limited seasonal analysis performed.")
        return None, None
    
    # Create a copy with the index as the date for time series analysis
    churn_ts = churn_over_time.set_index('date')['churn_rate']
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(churn_ts, model='additive', period=12)
    
    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    ax1.set_ylabel('Churn Rate')
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    ax2.set_ylabel('Trend')
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonality')
    ax3.set_ylabel('Seasonal Effect')
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    ax4.set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(TRENDS_IMAGES_DIR, 'seasonal_decomposition.png'))
    plt.close()
    
    # Extract month from date and calculate monthly averages
    churn_over_time['month'] = churn_over_time['date'].dt.month
    monthly_churn = churn_over_time.groupby('month')['churn_rate'].mean().reset_index()
    
    # Add month names
    monthly_churn['month_name'] = monthly_churn['month'].apply(lambda x: calendar.month_name[x])
    
    # Sort by month
    monthly_churn = monthly_churn.sort_values('month')
    
    # Plot monthly averages
    plt.figure(figsize=(12, 6))
    sns.barplot(x='month_name', y='churn_rate', data=monthly_churn)
    plt.title('Average Churn Rate by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(TRENDS_IMAGES_DIR, 'average_churn_by_month.png'))
    plt.close()
    
    return decomposition, monthly_churn

def analyze_event_impact(churn_over_time, events_df, window_size=1):
    """
    Analyze the impact of business events on churn rates.
    
    Parameters:
    -----------
    churn_over_time : pandas.DataFrame
        Aggregated churn data over time
    events_df : pandas.DataFrame
        Business events data
    window_size : int
        Number of periods before and after an event to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Event impact analysis results
    """
    # Initialize results list
    impact_results = []
    
    # For each event
    for _, event in events_df.iterrows():
        event_date = event['date']
        event_name = event['event']
        event_category = event['category']
        
        # Find the period containing the event
        event_period = churn_over_time[churn_over_time['date'] >= event_date].iloc[0]
        event_index = churn_over_time[churn_over_time['date'] >= event_date].index[0]
        
        # Calculate pre-event average (excluding the event period)
        pre_indices = range(max(0, event_index - window_size), event_index)
        pre_event_rate = churn_over_time.iloc[pre_indices]['churn_rate'].mean() if pre_indices else None
        
        # Calculate post-event average (including the event period)
        post_indices = range(event_index, min(len(churn_over_time), event_index + window_size + 1))
        post_event_rate = churn_over_time.iloc[post_indices]['churn_rate'].mean() if post_indices else None
        
        # Calculate the change
        if pre_event_rate is not None and post_event_rate is not None:
            absolute_change = post_event_rate - pre_event_rate
            percent_change = (absolute_change / pre_event_rate) * 100 if pre_event_rate > 0 else float('inf')
            
            # Perform t-test to check if the change is statistically significant
            if len(pre_indices) > 1 and len(post_indices) > 1:
                pre_values = churn_over_time.iloc[pre_indices]['churn_rate'].values
                post_values = churn_over_time.iloc[post_indices]['churn_rate'].values
                t_stat, p_value = stats.ttest_ind(pre_values, post_values, equal_var=False)
                significant = p_value < 0.05
            else:
                t_stat = None
                p_value = None
                significant = None
        else:
            absolute_change = None
            percent_change = None
            t_stat = None
            p_value = None
            significant = None
        
        # Store results
        impact_results.append({
            'event_date': event_date,
            'event_name': event_name,
            'event_category': event_category,
            'pre_event_rate': pre_event_rate,
            'post_event_rate': post_event_rate,
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant
        })
    
    # Convert to DataFrame
    impact_df = pd.DataFrame(impact_results)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Only include events with valid percent change
    valid_impact = impact_df.dropna(subset=['percent_change'])
    
    # Sort by absolute impact
    valid_impact = valid_impact.sort_values('absolute_change')
    
    # Create a horizontal bar chart
    colors = ['r' if x >= 0 else 'g' for x in valid_impact['absolute_change']]
    plt.barh(valid_impact['event_name'], valid_impact['percent_change'], color=colors)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Impact of Business Events on Churn Rate')
    plt.xlabel('Percent Change in Churn Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(TRENDS_IMAGES_DIR, 'event_impact_analysis.png'))
    plt.close()
    
    # Save results to CSV
    impact_df.to_csv(os.path.join(TRENDS_RESULTS_DIR, 'event_impact_analysis.csv'), index=False)
    
    return impact_df

def generate_trend_analysis_report(churn_over_time, monthly_churn, event_impact_df):
    """
    Generate a comprehensive report of the churn trend analysis.
    
    Parameters:
    -----------
    churn_over_time : pandas.DataFrame
        Aggregated churn data over time
    monthly_churn : pandas.DataFrame
        Monthly churn rate aggregation
    event_impact_df : pandas.DataFrame
        Event impact analysis results
    """
    # Calculate overall average churn rate
    avg_churn_rate = churn_over_time['churn_rate'].mean() * 100
    
    # Find months with highest and lowest churn
    if monthly_churn is not None:
        highest_month = monthly_churn.loc[monthly_churn['churn_rate'].idxmax()]
        lowest_month = monthly_churn.loc[monthly_churn['churn_rate'].idxmin()]
    else:
        highest_month = None
        lowest_month = None
    
    # Find the most impactful events (both positive and negative)
    if event_impact_df is not None and not event_impact_df.empty:
        most_positive_event = event_impact_df.loc[event_impact_df['percent_change'].idxmin()] if 'percent_change' in event_impact_df.columns and not event_impact_df['percent_change'].isna().all() else None
        most_negative_event = event_impact_df.loc[event_impact_df['percent_change'].idxmax()] if 'percent_change' in event_impact_df.columns and not event_impact_df['percent_change'].isna().all() else None
    else:
        most_positive_event = None
        most_negative_event = None
    
    # Create report
    report = [
        "# Churn Trend Analysis Report\n\n",
        "## Overview\n",
        "This report presents the analysis of churn patterns over time, identifying seasonal trends and the impact of major business events on customer churn behavior.\n\n",
        "## Key Metrics\n",
        f"- Average Churn Rate: {avg_churn_rate:.2f}%\n",
        f"- Highest Monthly Churn Rate: {churn_over_time['churn_rate'].max() * 100:.2f}%\n",
        f"- Lowest Monthly Churn Rate: {churn_over_time['churn_rate'].min() * 100:.2f}%\n\n",
        "## Seasonal Patterns\n"
    ]
    
    if highest_month is not None and lowest_month is not None:
        report.extend([
            f"- Month with Highest Average Churn: {highest_month['month_name']} ({highest_month['churn_rate'] * 100:.2f}%)\n",
            f"- Month with Lowest Average Churn: {lowest_month['month_name']} ({lowest_month['churn_rate'] * 100:.2f}%)\n\n"
        ])
    
    # Add seasonal observations
    report.append("### Observed Seasonal Patterns:\n")
    
    # If we have more than one year of data, calculate quarter averages
    if len(churn_over_time) >= 12:
        churn_over_time['quarter'] = churn_over_time['date'].dt.quarter
        quarterly_churn = churn_over_time.groupby('quarter')['churn_rate'].mean().reset_index()
        
        report.append("#### Quarterly Churn Rates:\n")
        for _, row in quarterly_churn.iterrows():
            report.append(f"- Q{int(row['quarter'])}: {row['churn_rate'] * 100:.2f}%\n")
        
        # Identify if there's a clear seasonal pattern
        max_q = quarterly_churn.loc[quarterly_churn['churn_rate'].idxmax()]
        min_q = quarterly_churn.loc[quarterly_churn['churn_rate'].idxmin()]
        
        if min_q['churn_rate'] > 0:
            report.append(f"\nQuarter with highest churn (Q{int(max_q['quarter'])}) has {(max_q['churn_rate'] / min_q['churn_rate'] - 1) * 100:.1f}% higher churn rate than the quarter with lowest churn (Q{int(min_q['quarter'])}).\n\n")
        else:
            report.append(f"\nQuarter with highest churn (Q{int(max_q['quarter'])}) has {max_q['churn_rate'] * 100:.1f}% churn rate, while the quarter with lowest churn (Q{int(min_q['quarter'])}) has 0% churn rate.\n\n")
    
    # Add event impact analysis
    report.append("## Business Event Impact\n")
    
    if most_positive_event is not None and most_negative_event is not None:
        report.extend([
            "### Events with Largest Impact on Churn:\n",
            f"- Most Positive Impact (Churn Reduction): {most_positive_event['event_name']} on {most_positive_event['event_date'].strftime('%Y-%m-%d')} with {most_positive_event['percent_change']:.1f}% change in churn rate\n",
            f"- Most Negative Impact (Churn Increase): {most_negative_event['event_name']} on {most_negative_event['event_date'].strftime('%Y-%m-%d')} with {most_negative_event['percent_change']:.1f}% change in churn rate\n\n"
        ])
    
    # Add summary of all events
    if event_impact_df is not None and not event_impact_df.empty:
        report.append("### Summary of All Business Events Impact:\n")
        report.append("| Event | Date | Pre-Event Churn | Post-Event Churn | % Change | Significant? |\n")
        report.append("|-------|------|----------------|------------------|----------|-------------|\n")
        
        for _, row in event_impact_df.iterrows():
            report.append(
                f"| {row['event_name']} | {row['event_date'].strftime('%Y-%m-%d')} | "
                f"{row['pre_event_rate'] * 100:.2f}% | {row['post_event_rate'] * 100:.2f}% | "
                f"{row['percent_change']:.1f}% | {'Yes' if row['significant'] else 'No'} |\n"
            )
    
    # Add recommendations based on findings
    report.append("\n## Recommendations\n")
    
    # Seasonal recommendations
    if highest_month is not None:
        report.append(f"1. **Seasonal Retention Strategy**: Implement enhanced customer retention campaigns before and during {highest_month['month_name']} when churn rates are historically highest.\n")
    
    # Event-based recommendations
    if most_negative_event is not None:
        report.append(f"2. **Event Planning**: For future events similar to '{most_negative_event['event_name']}', develop mitigation strategies to reduce customer churn, such as proactive communication or special offers.\n")
    
    # General trend recommendations
    report.append("3. **Regular Monitoring**: Establish a monthly churn trend monitoring dashboard to detect unusual patterns early.\n")
    
    # Final recommendation based on overall analysis
    report.append("4. **Predictive Alert System**: Use the identified seasonal patterns to create an early warning system for periods with expected high churn.\n")
    
    # Write report to file
    with open(os.path.join(TRENDS_RESULTS_DIR, 'churn_trend_analysis_report.md'), 'w') as f:
        f.writelines(report)
    
    print("Churn trend analysis report generated successfully.")

def main():
    # Load and preprocess the data
    df = load_and_preprocess_data()
    
    # Create synthetic dates for analysis
    # Note: In a real-world scenario, you would use actual customer join and churn dates
    date_df = create_synthetic_dates(df, start_date='2020-01-01')
    
    # Add business events for analysis
    events_df = add_business_events(start_date='2020-01-01', end_date='2022-01-01')
    
    # Aggregate churn data by month
    churn_over_time = aggregate_churn_by_time(date_df, freq='M')
    
    # Plot churn trends over time
    plot_churn_trends(churn_over_time, events_df)
    
    # Analyze seasonal patterns
    decomposition, monthly_churn = analyze_seasonal_patterns(churn_over_time)
    
    # Analyze impact of business events
    event_impact_df = analyze_event_impact(churn_over_time, events_df, window_size=1)
    
    # Generate comprehensive report
    generate_trend_analysis_report(churn_over_time, monthly_churn, event_impact_df)
    
    print("Churn trend analysis completed successfully.")

if __name__ == "__main__":
    main() 